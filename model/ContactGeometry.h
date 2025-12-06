#pragma once

#include "DroneDynamics.h"
#include "HullLoader.h"

#include <Eigen/Dense>
#include <vector>

struct Plane {
    Eigen::Vector3d n; // unit normal pointing into the workspace
    double d;          // plane equation: n.dot(x) + d = 0
};

struct ContactPoint {
    Eigen::Vector3d x_W;      // contact point in world coordinates
    Eigen::Vector3d n_W;      // contact normal in world (unit)
    double penetration{0.0};  // depth > 0 when inside activation band
    Eigen::Vector3d force_W;  // contact force in world
    int bodyId{0};            // 0 = base, 1..4 = arm index
};

struct ContactParams {
    double contactStiffness{0.0};
    double contactDamping{0.0};
    double activationDistance{0.0};
};

std::vector<ContactPoint> computeContacts(
    const DroneDynamics::ArmKinematics arms[4],
    const Eigen::Vector3d& W_r_B,
    const Eigen::Matrix3d& R_WB,
    const Eigen::Vector3d& v_WB,
    const Eigen::Vector3d& W_omega_B,
    const Eigen::Matrix<double,3,4>& W_omega_P,
    const ConvexHullShapes& hulls,
    const std::vector<Plane>& planes,
    const ContactParams& params);
