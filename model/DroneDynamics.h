#pragma once

#include "DroneParameters.h"
#include "ContactTypes.h"

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>


class DroneDynamics {
public:
    explicit DroneDynamics(const std::string& yamlPath);

    static constexpr int kStateSize = 41;
    static constexpr int kInputSize = 4;

    struct ArmKinematics {
        Eigen::Matrix3d R_BH0;
        Eigen::Matrix3d R_H0H;
        Eigen::Matrix3d R_BH;
        Eigen::Matrix3d R_WH;
        Eigen::Matrix3d R_WP;
        Eigen::Matrix3d R_HP;
        Eigen::Vector3d W_r_BH;
        Eigen::Vector3d W_r_HP;
        Eigen::Vector3d W_r_BP;
    };

    Eigen::VectorXd derivative(const Eigen::VectorXd& state,
                               const Eigen::Vector4d& thrust) const;

    const DroneParameters& params() const { return params_; }
    const std::vector<Plane>& contactPlanes() const { return contactPlanes_; }
    std::array<Eigen::Vector3d, 4> armEulerZYX(const Eigen::VectorXd& state) const;
    std::array<ArmKinematics,4> computeArmFramesFromState(
        const Eigen::Quaterniond& q_base,
        const std::array<Eigen::Quaterniond,4>& armQuat) const;

private:
    DroneParameters params_;
    Eigen::Matrix3d jointDamping_;
    Eigen::Matrix3d jointStiffness_;
    Eigen::Vector3d gravity_{0.0, 0.0, -9.8066};
    std::vector<Plane> contactPlanes_;

    static Eigen::Quaterniond makeQuaternion(const Eigen::Vector4d& wxyz);
    static Eigen::Matrix3d skew(const Eigen::Vector3d& v);
    static Eigen::Vector3d vee(const Eigen::Matrix3d& S);
    static Eigen::Vector3d logSO3(const Eigen::Matrix3d& R);

    std::array<ArmKinematics, 4> computeArmFrames(
        const Eigen::Quaterniond& q_base,
        const std::array<Eigen::Quaterniond,4>& armQuat) const;

    void buildTranslationalBlock(
        const std::array<ArmKinematics,4>& arms,
        const std::array<Eigen::Vector3d, 4>& WF_thrust,
        const Eigen::Vector3d& W_omega_B,
        const std::array<Eigen::Vector3d,4>& W_omega_P,
        Eigen::Matrix<double,3,18>& A,
        Eigen::Vector3d& b) const;

    void buildBodyRotationalBlock(
        const std::array<ArmKinematics,4>& arms,
        const std::array<Eigen::Vector3d, 4>& WF_thrust,
        const Eigen::Matrix3d& R_WB,
        const Eigen::Vector3d& B_omega_B,
        const Eigen::Matrix<double,3,4>& P_tau_SD,
        Eigen::Matrix<double,3,18>& A,
        Eigen::Vector3d& b) const;

    void buildArmBlock(
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
        const Eigen::Vector3d& P_tau_contact_i,
        Eigen::Matrix<double,3,18>& A,
        Eigen::Vector3d& b) const;
};
