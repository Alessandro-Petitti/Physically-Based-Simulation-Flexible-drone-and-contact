#include "DroneDynamics.h"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <iostream>

namespace {
constexpr double kQuatEps = 1e-12;
const Eigen::Vector3d kEz(0.0, 0.0, 1.0);

template <typename Derived>
void assertFinite(const Eigen::MatrixBase<Derived>& value, const std::string& label) {
    if (!value.allFinite()) {
        std::ostringstream oss;
        oss << "Non-finite entries detected in " << label << ":\n" << value;
        throw std::runtime_error(oss.str());
    }
}
} // namespace

DroneDynamics::DroneDynamics(const std::string& yamlPath)
    : params_(loadDroneParameters(yamlPath)) {
    jointDamping_ = params_.jointDamping * Eigen::Matrix3d::Identity();
    jointStiffness_ = params_.jointStiffness * Eigen::Matrix3d::Identity();
}

Eigen::Quaterniond DroneDynamics::makeQuaternion(const Eigen::Vector4d& wxyz) {
    Eigen::Quaterniond q(wxyz[0], wxyz[1], wxyz[2], wxyz[3]);
    if (std::abs(q.norm()) > kQuatEps) {
        q.normalize();
    } else {
        q = Eigen::Quaterniond::Identity();
    }
    return q;
}

Eigen::Matrix3d DroneDynamics::skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
}

Eigen::Vector3d DroneDynamics::vee(const Eigen::Matrix3d& S) {
    return Eigen::Vector3d(S(2,1) - S(1,2),
                           S(0,2) - S(2,0),
                           S(1,0) - S(0,1)) * 0.5;
}

Eigen::Vector3d DroneDynamics::logSO3(const Eigen::Matrix3d& R) {
    const double tr = R.trace();
    double cos_th = 0.5 * (tr - 1.0);
    cos_th = std::max(-1.0, std::min(1.0, cos_th));
    const double th = std::acos(cos_th);
    const double s_th = std::sin(th);
    const Eigen::Matrix3d S = R - R.transpose();
    const Eigen::Vector3d veeS = vee(S);
    const Eigen::Vector3d v_standard = (th / (2.0 * (s_th + 1e-12))) * veeS;
    const Eigen::Vector3d v_small = 0.5 * veeS;

    double rx = std::sqrt(std::max(0.0, (R(0,0) + 1.0) * 0.5));
    double ry = std::sqrt(std::max(0.0, (R(1,1) + 1.0) * 0.5));
    double rz = std::sqrt(std::max(0.0, (R(2,2) + 1.0) * 0.5));
    rx = std::copysign(rx, R(2,1) - R(1,2));
    ry = std::copysign(ry, R(0,2) - R(2,0));
    rz = std::copysign(rz, R(1,0) - R(0,1));
    Eigen::Vector3d A_u_pi(rx, ry, rz);
    if (A_u_pi.norm() > 1e-12) {
        A_u_pi.normalize();
    }
    const Eigen::Vector3d v_near_pi = th * A_u_pi;
    const Eigen::Vector3d v1 = (th < 1e-4) ? v_small : v_standard;
    if (th > (M_PI - 1e-3)) {
        return v_near_pi;
    }
    return v1;
}

std::array<DroneDynamics::ArmKinematics, 4> DroneDynamics::computeArmFrames(
    const Eigen::Quaterniond& q_base,
    const std::array<Eigen::Quaterniond,4>& armQuat) const {

    std::array<ArmKinematics,4> arms;
    const Eigen::Matrix3d R_WB = q_base.toRotationMatrix();
    for (int i = 0; i < 4; ++i) {
        ArmKinematics arm;
        const auto& T_BH = params_.T_BH[i];
        const auto& T_HP = params_.T_HP[i];
        arm.R_BH0 = T_BH.block<3,3>(0,0);
        const Eigen::Vector3d B_r_BH = T_BH.block<3,1>(0,3);
        arm.R_H0H = armQuat[i].toRotationMatrix();
        arm.R_BH = arm.R_BH0 * arm.R_H0H;
        arm.R_HP = T_HP.block<3,3>(0,0);
        const Eigen::Vector3d H_r_HP = T_HP.block<3,1>(0,3);

        arm.R_WH = R_WB * arm.R_BH;
        arm.R_WP = arm.R_WH * arm.R_HP;
        arm.W_r_BH = R_WB * B_r_BH;
        arm.W_r_HP = arm.R_WH * H_r_HP;
        arm.W_r_BP = arm.W_r_BH + arm.W_r_HP;
        arms[i] = arm;
    }
    return arms;
}

void DroneDynamics::buildTranslationalBlock(
    const std::array<ArmKinematics,4>& arms,
    const std::array<Eigen::Vector3d, 4>& WF_thrust,
    const Eigen::Vector3d& W_omega_B,
    const std::array<Eigen::Vector3d,4>& W_omega_P,
    Eigen::Matrix<double,3,18>& A,
    Eigen::Vector3d& b) const {

    A.setZero();
    b.setZero();

    const double mB = params_.massBase;
    const double mP = params_.massArm;
    const double Mtot = mB + 4.0 * mP;

    A.block<3,3>(0,0) = Mtot * Eigen::Matrix3d::Identity();
    b += Mtot * gravity_;

    Eigen::Vector3d sumThrust = Eigen::Vector3d::Zero();
    Eigen::Vector3d sumCentripetal = Eigen::Vector3d::Zero();

    for (int i = 0; i < 4; ++i) {
        const auto& arm = arms[i];
        const Eigen::Vector3d& W_r_BH = arm.W_r_BH;
        const Eigen::Vector3d& W_r_HP = arm.W_r_HP;
        assertFinite(W_r_BH, "W_r_BH arm " + std::to_string(i));
        assertFinite(W_r_HP, "W_r_HP arm " + std::to_string(i));
        A.block<3,3>(0,3) += -mP * skew(W_r_BH);
        const int col = 6 + 3 * i;
        A.block<3,3>(0,col) = -mP * skew(W_r_HP);
        assertFinite(WF_thrust[i], "WF thrust arm " + std::to_string(i));
        sumThrust += WF_thrust[i];
        b += WF_thrust[i];
        Eigen::Vector3d termB = -mP * (W_omega_B.cross(W_omega_B.cross(W_r_BH)));
        Eigen::Vector3d termP = -mP * (W_omega_P[i].cross(W_omega_P[i].cross(W_r_HP)));
        double termBMag = termB.cwiseAbs().maxCoeff();
        double termPMag = termP.cwiseAbs().maxCoeff();
        if (!termB.allFinite() || termBMag > 1e3) {
            std::ostringstream oss;
            oss << "translational base term abnormal arm " << i
                << " | W_omega_B: " << W_omega_B.transpose()
                << " | W_r_BH: " << W_r_BH.transpose()
                << " | termB: " << termB.transpose();
            throw std::runtime_error(oss.str());
        }
        if (!termP.allFinite() || termPMag > 1e3) {
            std::ostringstream oss;
            oss << "translational arm term abnormal arm " << i
                << " | W_omega_P: " << W_omega_P[i].transpose()
                << " | W_r_HP: " << W_r_HP.transpose()
                << " | termP: " << termP.transpose();
            throw std::runtime_error(oss.str());
        }
        sumCentripetal += termB + termP;
        b += termB;
        b += termP;
    }

    Eigen::Vector3d total = Mtot * gravity_ + sumThrust + sumCentripetal;
    double maxVal = total.cwiseAbs().maxCoeff();
    if (maxVal > 1e3) {
        std::ostringstream oss;
        oss << "Translational block large contributions\n"
            << "sumThrust: " << sumThrust.transpose()
            << " | sumCentripetal: " << sumCentripetal.transpose()
            << " | gravity: " << (Mtot * gravity_).transpose();
        throw std::runtime_error(oss.str());
    }
}

void DroneDynamics::buildBodyRotationalBlock(
    const std::array<ArmKinematics,4>& arms,
    const std::array<Eigen::Vector3d, 4>& WF_thrust,
    const Eigen::Matrix3d& R_WB,
    const Eigen::Vector3d& B_omega_B,
    const Eigen::Matrix<double,3,4>& P_tau_SD,
    Eigen::Matrix<double,3,18>& A,
    Eigen::Vector3d& b) const {

    A.setZero();
    b = -B_omega_B.cross(params_.inertiaBase * B_omega_B);

    const double mP = params_.massArm;
    const Eigen::Matrix3d R_BW = R_WB.transpose();
    const Eigen::Vector3d W_omega_B = R_WB * B_omega_B;

    A.block<3,3>(0,3) = params_.inertiaBase * R_BW;

    for (int i = 0; i < 4; ++i) {
        const auto& arm = arms[i];
        const Eigen::Vector3d& W_r_BH = arm.W_r_BH;
        const Eigen::Vector3d& W_r_HP = arm.W_r_HP;
        const Eigen::Vector3d B_r_BH = R_BW * W_r_BH;

        const Eigen::Matrix3d skewTerm = skew(B_r_BH) * R_BW;
        A.block<3,3>(0,0) += skewTerm * (mP * Eigen::Matrix3d::Identity());
        A.block<3,3>(0,3) += -skewTerm * (skew(W_r_BH) * mP);
        A.block<3,3>(0,6 + 3 * i) = -skewTerm * (skew(W_r_HP) * mP);

        b += skewTerm * (mP * gravity_);
        b += skewTerm * WF_thrust[i];
        b += -skewTerm * (mP * (W_omega_B.cross(W_omega_B.cross(W_r_BH))));
        b += -skewTerm * (mP * (W_omega_B.cross(W_omega_B.cross(W_r_HP))));

        const Eigen::Vector3d b_tau_sd_i = R_BW * (arm.R_WP * P_tau_SD.col(i));
        b += -b_tau_sd_i;
    }
}

void DroneDynamics::buildArmBlock(
    int armIndex,
    const std::array<ArmKinematics,4>& arms,
    const std::array<Eigen::Vector3d, 4>& WF_thrust,
    const Eigen::Matrix3d& R_WB,
    const Eigen::Vector3d& W_omega_B,
    const Eigen::Vector3d& W_omega_P_i,
    const Eigen::Vector3d& P_omega_P_i,
    const Eigen::Vector3d& P_tau_SD_i,
    const Eigen::Vector3d& P_tau_drag_i,
    const Eigen::Vector3d& P_tau_gyro_i,
    Eigen::Matrix<double,3,18>& A,
    Eigen::Vector3d& b) const {

    A.setZero();
    b.setZero();

    const auto& arm = arms[armIndex];
    const double mP = params_.massArm;
    const Eigen::Matrix3d& R_WP = arm.R_WP;
    const Eigen::Matrix3d R_PW = R_WP.transpose();
    const Eigen::Matrix3d R_PB = R_PW * R_WB;

    const Eigen::Vector3d& W_r_BH = arm.W_r_BH;
    const Eigen::Vector3d& W_r_HP = arm.W_r_HP;
    const Eigen::Vector3d P_r_PHi = R_PW * (-W_r_HP);

    const int col = 6 + 3 * armIndex;
    A.block<3,3>(0,col) = (params_.inertiaArm * R_PW) + (skew(P_r_PHi) * R_PW) * (mP * skew(W_r_HP));
    A.block<3,3>(0,3) = (skew(P_r_PHi) * R_PW) * (mP * skew(W_r_BH));
    A.block<3,3>(0,0) = -(skew(P_r_PHi) * R_PW) * (mP * Eigen::Matrix3d::Identity());

    b += -P_omega_P_i.cross(params_.inertiaArm * P_omega_P_i);
    b += (skew(P_r_PHi) * R_PW) *
         (mP * (W_omega_B.cross(W_omega_B.cross(W_r_BH)) +
                W_omega_P_i.cross(W_omega_P_i.cross(W_r_HP)) - gravity_));
    b += -(skew(P_r_PHi) * R_PW) * WF_thrust[armIndex];
    b += P_tau_SD_i + P_tau_drag_i + P_tau_gyro_i;
}

Eigen::VectorXd DroneDynamics::derivative(const Eigen::VectorXd& state,
                                          const Eigen::Vector4d& thrust) const {
    if (state.size() != kStateSize) {
        throw std::runtime_error("Unexpected state vector size");
    }
    assertFinite(state, "state vector");
    assertFinite(thrust, "thrust input");

    Eigen::Quaterniond q_base = makeQuaternion(state.segment<4>(3));
    Eigen::Vector3d v = state.segment<3>(7);
    Eigen::Vector3d w = state.segment<3>(10);

    std::array<Eigen::Quaterniond,4> armQuat;
    for (int i = 0; i < 4; ++i) {
        armQuat[i] = makeQuaternion(state.segment<4>(13 + 4 * i));
    }

    Eigen::Matrix<double,3,4> P_omega_rel;
    for (int i = 0; i < 4; ++i) {
        P_omega_rel.col(i) = state.segment<3>(29 + 3 * i);
    }

    const auto arms = computeArmFrames(q_base, armQuat);
    const Eigen::Matrix3d R_WB = q_base.toRotationMatrix();
    const Eigen::Matrix3d R_BW = R_WB.transpose();

    std::array<Eigen::Vector3d,4> WF_thrust;
    for (int i = 0; i < 4; ++i) {
        WF_thrust[i] = arms[i].R_WP * kEz * thrust[i];
    }

    Eigen::Vector4d omegaSq;
    for (int i = 0; i < 4; ++i) {
        omegaSq[i] = thrust[i] / (params_.kappaThrust + 1e-9);
        omegaSq[i] = std::max(0.0, omegaSq[i]);
    }

    Eigen::Matrix<double,3,4> P_tau_SD;
    Eigen::Matrix<double,3,4> P_tau_drag;
    Eigen::Matrix<double,3,4> P_tau_gyro;
    std::array<Eigen::Vector3d,4> P_omega_P;
    std::array<Eigen::Vector3d,4> W_omega_P;
    std::array<Eigen::Matrix3d,4> R_PB_list;
    std::array<Eigen::Matrix3d,4> R_WP_list;

    const Eigen::Vector3d W_omega_B = R_WB * w;
    double omegaNorm = W_omega_B.cwiseAbs().maxCoeff();
    if (omegaNorm > 1e6) {
        std::ostringstream oss;
        oss << "W_omega_B magnitude overflow (" << omegaNorm << ") with state w = "
            << w.transpose();
        throw std::runtime_error(oss.str());
    }

    for (int i = 0; i < 4; ++i) {
        const auto& arm = arms[i];
        const Eigen::Matrix3d& R_WP = arm.R_WP;
        const Eigen::Matrix3d R_PW = R_WP.transpose();
        const Eigen::Matrix3d R_PB = R_PW * R_WB;
        R_PB_list[i] = R_PB;
        R_WP_list[i] = R_WP;

        Eigen::Matrix3d R_H0H = arm.R_H0H;
        const Eigen::Vector3d H0_phi = logSO3(R_H0H);
        const Eigen::Vector3d H_phi = R_H0H.transpose() * H0_phi;
        const Eigen::Matrix3d R_PH = arm.R_HP.transpose();
        const Eigen::Vector3d P_phi = R_PH * H_phi;

        P_tau_SD.col(i) = -(jointDamping_ * P_omega_rel.col(i)) - (jointStiffness_ * P_phi);
        P_tau_drag.col(i) = -params_.motorDirection[i] * params_.kappaTorque * omegaSq[i] * kEz;
        const double spin = params_.motorDirection[i] * std::sqrt(omegaSq[i]);
        const Eigen::Vector3d h_i_P = params_.rotorInertia * spin * kEz;
        P_tau_gyro.col(i) = P_omega_rel.col(i).cross(h_i_P);

        Eigen::Vector3d P_omega_P_i = P_omega_rel.col(i) + R_PB * w;
        P_omega_P[i] = P_omega_P_i;
        W_omega_P[i] = R_WP * P_omega_P_i;
        assertFinite(W_omega_P[i], "W_omega_P arm " + std::to_string(i));
    }

    Eigen::Matrix<double,18,18> A = Eigen::Matrix<double,18,18>::Zero();
    Eigen::Matrix<double,18,1> rhs = Eigen::Matrix<double,18,1>::Zero();
    int row = 0;

    Eigen::Matrix<double,3,18> block;
    Eigen::Vector3d block_rhs;

    buildTranslationalBlock(arms, WF_thrust, W_omega_B, W_omega_P, block, block_rhs);
    A.block<3,18>(row,0) = block;
    rhs.segment<3>(row) = block_rhs;
    assertFinite(block_rhs, "translational RHS");
    assertFinite(block, "translational block");
    row += 3;

    buildBodyRotationalBlock(arms, WF_thrust, R_WB, w, P_tau_SD, block, block_rhs);
    A.block<3,18>(row,0) = block;
    rhs.segment<3>(row) = block_rhs;
    assertFinite(block_rhs, "body rotational RHS");
    assertFinite(block, "body rotational block");
    row += 3;

    for (int i = 0; i < 4; ++i) {
        buildArmBlock(i, arms, WF_thrust, R_WB, W_omega_B, W_omega_P[i], P_omega_P[i],
                      P_tau_SD.col(i), P_tau_drag.col(i), P_tau_gyro.col(i), block, block_rhs);
        A.block<3,18>(row,0) = block;
        rhs.segment<3>(row) = block_rhs;
        assertFinite(block_rhs, "arm block RHS (arm " + std::to_string(i) + ")");
        assertFinite(block, "arm block matrix (arm " + std::to_string(i) + ")");
        row += 3;
    }

    assertFinite(A, "matrix A");
    assertFinite(rhs, "rhs b");

        // --- TIMER: linear solve (A z = rhs) ---
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    Eigen::FullPivLU<Eigen::Matrix<double,18,18>> solver(A);
    if (!solver.isInvertible()) {
        throw std::runtime_error("Dynamics matrix is singular");
    }
    Eigen::Matrix<double,18,1> z = solver.solve(rhs);

    auto t1 = clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    //std::cout << "[TIMER] linear solve (A z = rhs) took " << solve_ms << " ms\n";
    
    // --- END TIMER ---
    double maxZ = z.cwiseAbs().maxCoeff();
    if (!(std::isfinite(maxZ) && maxZ < 1e8)) {
        std::ostringstream oss;
        oss << "Dynamics solve produced large entries (max |z| = " << maxZ << ")\n"
            << "state w: " << w.transpose() << "\n"
            << "state v: " << v.transpose() << "\n"
            << "RHS:\n" << rhs;
        throw std::runtime_error(oss.str());
    }
    assertFinite(z, "solution z");

    const Eigen::Vector3d W_a_B = z.segment<3>(0);
    const Eigen::Vector3d W_omega_dot_B = z.segment<3>(3);
    const Eigen::Vector3d B_omega_dot_B = R_BW * W_omega_dot_B;

    Eigen::Matrix<double,3,4> P_omega_rel_dot;
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector3d W_omega_dot_P_i = z.segment<3>(6 + 3 * i);
        const Eigen::Matrix3d R_PW = R_WP_list[i].transpose();
        const Eigen::Matrix3d& R_PB = R_PB_list[i];
        const Eigen::Vector3d P_omega_dot_P_i = R_PW * W_omega_dot_P_i;
        const Eigen::Vector3d P_omega_P_i = P_omega_P[i];
        const Eigen::Vector3d P_omega_B_i = R_PB * w;
        const Eigen::Vector3d P_omega_rel_dot_i =
            P_omega_dot_P_i - P_omega_P_i.cross(P_omega_B_i) - (R_PB * B_omega_dot_B);
        P_omega_rel_dot.col(i) = P_omega_rel_dot_i;
    }

    Eigen::VectorXd deriv(kStateSize);
    deriv.setZero();
    deriv.segment<3>(0) = v;

    const Eigen::Quaterniond omega_q(0.0, w.x(), w.y(), w.z());
    Eigen::Quaterniond q_dot = q_base * omega_q;
    q_dot.coeffs() *= 0.5;
    deriv.segment<4>(3) << q_dot.w(), q_dot.x(), q_dot.y(), q_dot.z();

    deriv.segment<3>(7) = W_a_B;
    deriv.segment<3>(10) = B_omega_dot_B;

    for (int i = 0; i < 4; ++i) {
        const Eigen::Quaterniond omega_arm(0.0,
                                           P_omega_rel.col(i).x(),
                                           P_omega_rel.col(i).y(),
                                           P_omega_rel.col(i).z());
        Eigen::Quaterniond q_dot_arm = armQuat[i] * omega_arm;
        q_dot_arm.coeffs() *= 0.5;
        deriv.segment<4>(13 + 4 * i) << q_dot_arm.w(), q_dot_arm.x(), q_dot_arm.y(), q_dot_arm.z();
    }

    for (int i = 0; i < 4; ++i) {
        deriv.segment<3>(29 + 3 * i) = P_omega_rel_dot.col(i);
    }

    assertFinite(deriv, "state derivative");
    return deriv;
}

std::array<Eigen::Vector3d,4> DroneDynamics::armEulerZYX(const Eigen::VectorXd& state) const {
    std::array<Eigen::Vector3d,4> eulers;
    for (int i = 0; i < 4; ++i) {
        Eigen::Quaterniond q = makeQuaternion(state.segment<4>(13 + 4 * i));
        const Eigen::Vector3d angles = q.toRotationMatrix().eulerAngles(2,1,0);
        eulers[i] = angles;
    }
    return eulers;
}
