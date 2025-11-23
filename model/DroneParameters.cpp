#include "DroneParameters.h"

#include <stdexcept>

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
    for (size_t i = 0; i < 4; ++i) {
        params.T_HP[i] = pose7ToMatrix(yaml.nodeAtPath("transforms.T_HP." + HPKeys[i]).asSequence());
    }

    return params;
}
