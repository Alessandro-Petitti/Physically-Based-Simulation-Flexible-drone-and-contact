#include "DroneParameters.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace {
Eigen::Matrix4d pose7ToMatrix(const std::vector<double>& pose) {
    if (pose.size() != 7) {
        throw std::runtime_error("Pose must contain 7 elements");
    }
    Eigen::Vector3d t(pose[0], pose[1], pose[2]);
    Eigen::Vector4d q(pose[3], pose[4], pose[5], pose[6]);
    const double norm = q.norm();
    if (norm > 1e-9) {
        q /= norm;
    }
    Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = quat.toRotationMatrix();
    T.block<3,1>(0,3) = t;
    return T;
}

Eigen::Matrix3d diagMatrix(double x, double y, double z) {
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    M(0,0) = x;
    M(1,1) = y;
    M(2,2) = z;
    return M;
}

IntegratorType parseIntegrator(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "euler" || lower == "explicit_euler") return IntegratorType::ExplicitEuler;
    if (lower == "rk4" || lower == "rungekutta4") return IntegratorType::Rk4;
    if (lower == "implicit_euler" || lower == "euler_implicit" || lower == "impliciteuler") {
        return IntegratorType::ImplicitEuler;
    }
    if (lower == "irk" || lower == "implicit_midpoint" || lower == "midpoint") {
        return IntegratorType::ImplicitMidpoint;
    }
    throw std::runtime_error("Unknown integrator: " + name);
}
} // namespace

DroneParameters loadDroneParameters(const std::string& path) {
    SimpleYaml yaml(path);
    DroneParameters params;

    params.massTotal = yaml.nodeAtPath("mass.total").asScalar();
    params.massBase = yaml.nodeAtPath("mass.base").asScalar();
    params.massArm = (params.massTotal - params.massBase) / 4.0;

    const double P_Ixx = yaml.nodeAtPath("inertia.P_Ixx_P").asScalar();
    const double P_Iyy = yaml.nodeAtPath("inertia.P_Iyy_P").asScalar();
    const double P_Izz = yaml.nodeAtPath("inertia.P_Izz_P").asScalar();
    params.inertiaArm = diagMatrix(P_Ixx, P_Iyy, P_Izz);

    const double B_Ixx = yaml.nodeAtPath("inertia.B_Ixx_B").asScalar();
    const double B_Iyy = yaml.nodeAtPath("inertia.B_Iyy_B").asScalar();
    const double B_Izz = yaml.nodeAtPath("inertia.B_Izz_B").asScalar();
    params.inertiaBase = diagMatrix(B_Ixx, B_Iyy, B_Izz);

    params.kappaThrust = yaml.nodeAtPath("propellers.kappa_thrust").asScalar();
    params.kappaTorque = yaml.nodeAtPath("propellers.kappa_torque").asScalar();
    params.rotorInertia = yaml.nodeAtPath("propellers.J_r").asScalar();
    try {
        params.propellerMaxThrust = yaml.nodeAtPath("propellers.thrust_max").asScalar();
    } catch (const std::exception&) {
        params.propellerMaxThrust.reset();
    }

    const auto& dir = yaml.nodeAtPath("propellers.direction").asSequence();
    if (dir.size() != 4) {
        throw std::runtime_error("propellers.direction must have 4 entries");
    }
    for (size_t i = 0; i < 4; ++i) {
        params.motorDirection[i] = dir[i];
    }

    params.jointDamping = yaml.nodeAtPath("morphing_joint.b_joint").asScalar();
    params.jointStiffness = yaml.nodeAtPath("morphing_joint.k_joint").asScalar();

    const std::array<std::string, 4> motorKeys{"motor_0", "motor_1", "motor_2", "motor_3"};
    for (size_t i = 0; i < 4; ++i) {
        params.T_BP[i] = pose7ToMatrix(yaml.nodeAtPath("transforms.T_BP." + motorKeys[i]).asSequence());
    }

    const std::array<std::string, 4> hingeKeys{"H0", "H1", "H2", "H3"};
    for (size_t i = 0; i < 4; ++i) {
        params.T_BH[i] = pose7ToMatrix(yaml.nodeAtPath("transforms.T_BH." + hingeKeys[i]).asSequence());
    }

    const std::array<std::string, 4> HPKeys{
        "H0_to_motor_0",
        "H1_to_motor_1",
        "H2_to_motor_2",
        "H3_to_motor_3"
    };
    Eigen::Matrix3d R_fix_HP = Eigen::AngleAxisd(M_PI * 0.5, Eigen::Vector3d::UnitY()).toRotationMatrix();
    for (size_t i = 0; i < 4; ++i) {
        params.T_HP[i] = pose7ToMatrix(yaml.nodeAtPath("transforms.T_HP." + HPKeys[i]).asSequence());
        // Inject the missing rotation from URDF joint (0, pi/2, 0)
        params.T_HP[i].block<3,3>(0,0) = params.T_HP[i].block<3,3>(0,0) * R_fix_HP;
    }

    try {
        params.integrator = parseIntegrator(yaml.nodeAtPath("integrator").asString());
    } catch (const std::exception&) {
        params.integrator = IntegratorType::Rk4;
    }

    try {
        params.integratorSettings.dt = yaml.nodeAtPath("integrator_settings.dt").asScalar();
    } catch (const std::exception&) {}
    try {
        params.integratorSettings.substeps =
            static_cast<int>(yaml.nodeAtPath("integrator_settings.substeps").asScalar());
    } catch (const std::exception&) {}
    try {
        params.integratorSettings.implicitMaxIterations =
            static_cast<int>(yaml.nodeAtPath("integrator_settings.implicit_max_iterations").asScalar());
    } catch (const std::exception&) {}
    try {
        params.integratorSettings.implicitTolerance =
            yaml.nodeAtPath("integrator_settings.implicit_tolerance").asScalar();
    } catch (const std::exception&) {}
    try {
        params.integratorSettings.implicitFdEps =
            yaml.nodeAtPath("integrator_settings.implicit_fd_epsilon").asScalar();
    } catch (const std::exception&) {}

    try {
        const auto pos = yaml.nodeAtPath("x0_pos").asSequence();
        if (pos.size() == 3) {
            params.x0_pos = Eigen::Vector3d(pos[0], pos[1], pos[2]);
        }
    } catch (const std::exception&) {}
    try {
        const auto quat = yaml.nodeAtPath("x0_rotation").asSequence();
        if (quat.size() == 4) {
            params.x0_rotation = Eigen::Vector4d(quat[0], quat[1], quat[2], quat[3]);
            const double n = params.x0_rotation.norm();
            if (n > 1e-9) params.x0_rotation /= n;
        }
    } catch (const std::exception&) {}

    return params;
}
