#pragma once

#include "ContactTypes.h"
#include "DroneDynamics.h"
#include "HullLoader.h"

#include <Eigen/Dense>
#include <vector>

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
    bool enableFriction{true};  // Enable/disable Coulomb friction
    double frictionCoefficient{0.5};  // Coulomb friction coefficient μ
    bool enableCCD{false};  // Continuous Collision Detection
};

/**
 * Previous frame state for CCD (Continuous Collision Detection).
 * Stores the world positions of hull vertices from the previous timestep.
 */
struct CCDPrevState {
    std::vector<Eigen::Vector3d> baseVertices_W;  // Previous base hull vertices in world
    std::array<std::vector<Eigen::Vector3d>, 4> armVertices_W;  // Previous arm hull vertices in world
    bool valid{false};  // True if this state has been initialized
};

/**
 * Compute contact forces using spring-damper model.
 * 
 * For each vertex of the convex hulls (base + 4 arms), checks if it's within
 * the activation distance of any plane, and computes a penalty force:
 *   F = k * penetration * n + b * v_approach * n
 */
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

/**
 * Compute contact forces with optional Continuous Collision Detection (CCD).
 * 
 * CCD prevents tunneling by detecting when a vertex crosses a plane between
 * the previous and current timestep. When CCD detects a crossing, it places
 * the contact at the intersection point and uses the full velocity for damping.
 * 
 * @param prevState Previous frame vertex positions (updated after each call)
 * @param dt Timestep for velocity estimation in CCD mode
 */
std::vector<ContactPoint> computeContactsWithCCD(
    const DroneDynamics::ArmKinematics arms[4],
    const Eigen::Vector3d& W_r_B,
    const Eigen::Matrix3d& R_WB,
    const Eigen::Vector3d& v_WB,
    const Eigen::Vector3d& W_omega_B,
    const Eigen::Matrix<double,3,4>& W_omega_P,
    const ConvexHullShapes& hulls,
    const std::vector<Plane>& planes,
    const ContactParams& params,
    CCDPrevState& prevState,
    double dt);
