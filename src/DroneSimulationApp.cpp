#include "DroneSimulationApp.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <polyscope/view.h>

namespace {
const char* integratorLabel(IntegratorType type) {
    switch (type) {
        case IntegratorType::ExplicitEuler: return "explicit_euler";
        case IntegratorType::ImplicitEuler: return "implicit_euler";
        case IntegratorType::ImplicitMidpoint: return "irk_implicit_midpoint";
        case IntegratorType::Rk4:
        default: return "rk4";
    }
}
} // namespace

DroneSimulationApp::DroneSimulationApp()
    : dynamics_("model/drone_parameters.yaml"),
      state_(Eigen::VectorXd::Zero(DroneDynamics::kStateSize)) {
    state_(3) = 1.0;
    for (int i = 0; i < 4; ++i) {
        state_(13 + 4 * i) = 1.0;
    }
    thrust_.setZero();
    integratorType_ = dynamics_.params().integrator;
    const auto& settings = dynamics_.params().integratorSettings;
    dt_ = (settings.dt > 0.0) ? settings.dt : 0.002;
    substeps_ = std::max(1, settings.substeps);
    const int maxIt = std::max(1, settings.implicitMaxIterations);
    const double tol = (settings.implicitTolerance > 0.0) ? settings.implicitTolerance : 1e-6;
    const double fdEps = (settings.implicitFdEps > 0.0) ? settings.implicitFdEps : 1e-6;
    implicitEuler_.setConfig(maxIt, tol, fdEps);
    irk_.setConfig(maxIt, tol, fdEps);
    logIntegratorSettings();
    std::cout << "Hover thrust per rotor: " << hoverThrust() << " N" << std::endl;
    initRotorData();
}

bool DroneSimulationApp::initializeScene(const std::string& urdfPath) {
    if (!rig_.initialize(urdfPath)) {
        return false;
    }
    rig_.update(baseTransform(), jointAngles());
    return true;
}

void DroneSimulationApp::step() {
    updateController();

    auto dyn = [this](const Eigen::VectorXd& x) {
        return dynamics_.derivative(x, thrust_);
    };

    for (int i = 0; i < substeps_; ++i) {
        switch (integratorType_) {
            case IntegratorType::ExplicitEuler:
                state_ = explicitEuler_.step(state_, dt_, dyn);
                break;
            case IntegratorType::ImplicitEuler:
                state_ = implicitEuler_.step(state_, dt_, dyn);
                break;
            case IntegratorType::ImplicitMidpoint:
                state_ = irk_.step(state_, dt_, dyn);
                break;
            case IntegratorType::Rk4:
            default:
                state_ = rk4_.step(state_, dt_, dyn);
                break;
        }
        normalizeQuaternions();
        simTime_ += dt_;
    }

    rig_.update(baseTransform(), jointAngles());

    if (simTime_ >= nextLogTime_) {
        logState();
        nextLogTime_ += logInterval_;
    }
}

void DroneSimulationApp::logIntegratorSettings() const {
    const auto& s = dynamics_.params().integratorSettings;
    std::cout << "Integrator: " << integratorLabel(integratorType_)
              << " | dt: " << dt_
              << " | substeps: " << substeps_;
    if (integratorType_ == IntegratorType::ImplicitEuler ||
        integratorType_ == IntegratorType::ImplicitMidpoint) {
        std::cout << " | implicit iters: " << s.implicitMaxIterations
                  << " | tol: " << s.implicitTolerance
                  << " | fd_eps: " << s.implicitFdEps;
    }
    std::cout << std::endl;
}

void DroneSimulationApp::initRotorData() {
    const auto& params = dynamics_.params();
    Eigen::Vector3d ez(0.0, 0.0, 1.0);
    for (int i = 0; i < 4; ++i) {
        rotorPositions_[i] = params.T_BP[i].block<3,1>(0,3);
        Eigen::Matrix3d R_BP = params.T_BP[i].block<3,3>(0,0);
        rotorDirsB_[i] = R_BP * ez;
    }
}

void DroneSimulationApp::updateController() {
    Eigen::Vector3d p = state_.segment<3>(0);
    Eigen::Vector3d v = state_.segment<3>(7);

    Eigen::Vector3d errorPos = positionRef_ - p;
    integralError_ += errorPos * (dt_ * substeps_);
    Eigen::Vector3d errorVel = -v;

    Eigen::Vector3d accelCmd = gains_.kp.cwiseProduct(errorPos)
                               + gains_.ki.cwiseProduct(integralError_)
                               + gains_.kd.cwiseProduct(errorVel)
                               - gravity_;

    Eigen::Vector3d F_des = dynamics_.params().massTotal * accelCmd;

    Eigen::Vector3d b3d = accelCmd;
    if (b3d.norm() < 1e-6) b3d = Eigen::Vector3d(0,0,1);
    b3d.normalize();
    Eigen::Vector3d a_yaw(std::cos(yawRef_), std::sin(yawRef_), 0.0);
    Eigen::Vector3d b2d = b3d.cross(a_yaw);
    if (b2d.norm() < 1e-6) b2d = Eigen::Vector3d(0,1,0);
    b2d.normalize();
    Eigen::Vector3d b1d = b2d.cross(b3d);
    Eigen::Matrix3d R_d;
    R_d.col(0) = b1d;
    R_d.col(1) = b2d;
    R_d.col(2) = b3d;

    Eigen::Quaterniond q(state_(3), state_(4), state_(5), state_(6));
    if (q.norm() > 1e-9) q.normalize();
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Matrix3d R_err = R_d.transpose() * R;
    Eigen::Vector3d e_R = 0.5 * Eigen::Vector3d(R_err(2,1) - R_err(1,2),
                                                R_err(0,2) - R_err(2,0),
                                                R_err(1,0) - R_err(0,1));
    Eigen::Vector3d w = state_.segment<3>(10);
    Eigen::Vector3d e_w = w;
    Eigen::Vector3d tau_cmd = - attGains_.kR.cwiseProduct(e_R) - attGains_.kOm.cwiseProduct(e_w);

    Eigen::Vector3d F_body = R.transpose() * F_des;
    Eigen::Matrix<double,6,4> H;
    double torqueRatio = dynamics_.params().kappaTorque / (dynamics_.params().kappaThrust + 1e-9);
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d e_i = rotorDirsB_[i];
        Eigen::Vector3d r_i = rotorPositions_[i];
        H.block<3,1>(0,i) = e_i;
        H.block<3,1>(3,i) = r_i.cross(e_i) + dynamics_.params().motorDirection[i] * torqueRatio * e_i;
    }
    Eigen::Matrix<double,6,1> wrench;
    wrench << F_body, tau_cmd;

    Eigen::Vector4d u;
    Eigen::JacobiSVD<Eigen::Matrix<double,6,4>> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    u = svd.solve(wrench);
    for (int i = 0; i < 4; ++i) {
        u[i] = std::clamp(u[i], 0.0, motorThrustMax());
    }
    thrust_ = u;
}

double DroneSimulationApp::motorThrustMax() const {
    const auto& params = dynamics_.params();
    return params.propellerMaxThrust.value_or(10.0);
}

void DroneSimulationApp::logState() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "\n=== t = " << simTime_ << " s ===\n";
    Eigen::Vector3d p = state_.segment<3>(0);
    Eigen::Vector3d v = state_.segment<3>(7);
    Eigen::Vector3d w = state_.segment<3>(10);
    Eigen::Vector4d q = state_.segment<4>(3);
    oss << "p [m]: " << p.transpose() << " | v [m/s]: " << v.transpose() << "\n";
    oss << "w [rad/s]: " << w.transpose() << " | q_base: " << q.transpose() << "\n";
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector4d qi = state_.segment<4>(13 + 4 * i);
        Eigen::Vector3d wi = state_.segment<3>(29 + 3 * i);
        oss << "arm_" << i << " q: " << qi.transpose()
            << " | rel_w [rad/s]: " << wi.transpose() << "\n";
    }
    oss << "thrust [N]: " << thrust_.transpose() << "\n";
    std::cout << oss.str() << std::endl;
}

double DroneSimulationApp::hoverThrust() const {
    return dynamics_.params().massTotal * 9.8066 / 4.0;
}

void DroneSimulationApp::normalizeQuaternions() {
    auto normalizeSegment = [this](int offset) {
        Eigen::Vector4d q = state_.segment<4>(offset);
        double n = q.norm();
        if (n > 1e-9) {
            state_.segment<4>(offset) = q / n;
        } else {
            state_.segment<4>(offset) << 1.0, 0.0, 0.0, 0.0;
        }
    };
    normalizeSegment(3);
    for (int i = 0; i < 4; ++i) {
        normalizeSegment(13 + 4 * i);
    }
}

Eigen::Isometry3d DroneSimulationApp::baseTransform() const {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = state_.segment<3>(0);
    Eigen::Quaterniond q(state_(3), state_(4), state_(5), state_(6));
    if (q.norm() > 1e-9) q.normalize();
    T.linear() = q.toRotationMatrix();
    return T;
}

std::unordered_map<std::string, double> DroneSimulationApp::jointAngles() const {
    std::unordered_map<std::string, double> joints;
    const auto eulers = dynamics_.armEulerZYX(state_);
    for (int i = 0; i < 4; ++i) {
        joints["base_link_to_connecting_link_" + std::to_string(i) + "_z"] = eulers[i].x();
        joints["base_link_to_connecting_link_" + std::to_string(i) + "_y"] = eulers[i].y();
        joints["base_link_to_connecting_link_" + std::to_string(i) + "_x"] = eulers[i].z();
    }
    return joints;
}
