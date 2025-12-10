#include "ContactGeometry.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr double kForceEps = 1e-12;

// BASE HULL FRAME: The URDF applies rpy="0 0 1.57079" (+π/2 around Z) to core_battery_transformed.stl
// The hull file core_battery_transformed_hull.obj is in the same frame, so we need the same rotation.
const Eigen::Matrix3d kBaseAlign = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();

// ARM HULL FRAME: We now use arm_transformed_hull.obj which matches arm_transformed.stl.
// The R_WP from computeArmFrames already includes the joint rotation (π/2 around Y via T_HP).
// No additional alignment needed.
const Eigen::Matrix3d kArmAlign = Eigen::Matrix3d::Identity();

/**
 * Compute contact force using spring-damper model with Coulomb friction.
 * 
 * Force model:
 *   F_n = k * penetration * n + b * max(-v_n, 0) * n  (normal force)
 *   F_t = -μ * |F_n| * v_t / |v_t|  (tangential friction, opposes sliding)
 *   F = F_n + F_t
 * 
 * where:
 *   - penetration = max(d_activation - phi, 0) with phi = n·x + d (signed distance)
 *   - v_n = velocity component along normal (positive = moving away from surface)
 *   - v_t = velocity component tangent to surface (sliding velocity)
 *   - μ = friction coefficient (Coulomb)
 *   - The damping only acts when approaching the surface (v_n < 0)
 *   - Force is always repulsive in normal direction
 * 
 * NOTE: For numerical stability, damping should satisfy: b << m / dt
 * With dt=0.0002s and m=0.26kg, this means b << 1300 Ns/m
 */
inline Eigen::Vector3d contactForce(const Plane& plane,
                                    double penetration,
                                    double v_n,
                                    const Eigen::Vector3d& v_point,
                                    const ContactParams& params) {
    // Normal force: spring + damping
    const double Fn_magnitude = params.contactStiffness * penetration;
    const double v_approach = std::max(-v_n, 0.0);  // positive when approaching
    const double Fd_magnitude = params.contactDamping * v_approach;
    const double total_normal = Fn_magnitude + Fd_magnitude;
    const Eigen::Vector3d F_normal = total_normal * plane.n;
    
    // Coulomb friction (if enabled)
    Eigen::Vector3d F_friction = Eigen::Vector3d::Zero();
    if (params.enableFriction && params.frictionCoefficient > 0.0) {
        // Tangential velocity (sliding velocity on the surface)
        const Eigen::Vector3d v_tangent = v_point - v_n * plane.n;
        const double v_t_mag = v_tangent.norm();
        
        if (v_t_mag > 1e-8) {
            // Maximum friction force (Coulomb limit)
            const double F_friction_max = params.frictionCoefficient * total_normal;
            // Friction opposes sliding: F_t = -μ * |F_n| * v_t / |v_t|
            F_friction = -F_friction_max * (v_tangent / v_t_mag);
        }
    }
    
    return F_normal + F_friction;
}
} // namespace

std::vector<ContactPoint> computeContacts(
    const DroneDynamics::ArmKinematics arms[4],
    const Eigen::Vector3d& W_r_B,
    const Eigen::Matrix3d& R_WB,
    const Eigen::Vector3d& v_WB,
    const Eigen::Vector3d& W_omega_B,
    const Eigen::Matrix<double,3,4>& W_omega_P,
    const ConvexHullShapes& hulls,
    const std::vector<Plane>& planes,
    const ContactParams& params) {

    const std::size_t estimatedContacts =
        (hulls.baseHull_B.size() + 4 * hulls.armHull_P[0].size()) * planes.size();
    std::vector<ContactPoint> contacts;
    contacts.reserve(estimatedContacts);

    auto processVertex = [&](const Eigen::Vector3d& x_W,
                             const Eigen::Vector3d& bodyOrigin_W,
                             const Eigen::Vector3d& omega_W,
                             const Eigen::Vector3d& v_bodyOrigin_W,
                             int bodyId) {
        for (const auto& plane : planes) {
            const double phi = plane.n.dot(x_W) + plane.d;
            if (phi >= params.activationDistance) {
                continue;
            }
            const double penetration = params.activationDistance - phi;
            const Eigen::Vector3d r_W = x_W - bodyOrigin_W;
            const Eigen::Vector3d v_point = v_bodyOrigin_W + omega_W.cross(r_W);
            const double v_n = v_point.dot(plane.n);
            const Eigen::Vector3d F_contact = contactForce(plane, penetration, v_n, v_point, params);
            if (F_contact.squaredNorm() < kForceEps) {
                continue;
            }
            ContactPoint cp;
            cp.x_W = x_W;
            cp.n_W = plane.n;
            cp.penetration = penetration;
            cp.force_W = F_contact;
            cp.bodyId = bodyId;
            contacts.push_back(cp);
        }
    };

    // Base hull
    for (const auto& p_B : hulls.baseHull_B) {
        const Eigen::Vector3d x_W = W_r_B + R_WB * (kBaseAlign * p_B);
        const Eigen::Vector3d bodyOrigin_W = W_r_B;
        const Eigen::Vector3d v_bodyOrigin_W = v_WB;
        processVertex(x_W, bodyOrigin_W, W_omega_B, v_bodyOrigin_W, 0);
    }

    // Arms hulls
    // Position: at hinge (H) since URDF joint has xyz="0 0 0"
    // Rotation: use R_WP because arm mesh is in frame P (after joint rotation rpy="0 π/2 0")
    for (int i = 0; i < 4; ++i) {
        const auto& armHull = hulls.armHull_P[i];
        const Eigen::Vector3d armOrigin_W = W_r_B + arms[i].W_r_BH;
        const Eigen::Matrix3d& R_WP = arms[i].R_WP;

        Eigen::Vector3d v_bodyOrigin_W = v_WB + W_omega_B.cross(arms[i].W_r_BH);
        const Eigen::Vector3d omega_W = W_omega_P.col(i);

        for (const auto& p_P : armHull) {
            const Eigen::Vector3d x_W = armOrigin_W + R_WP * (kArmAlign * p_P);
            processVertex(x_W, armOrigin_W, omega_W, v_bodyOrigin_W, i + 1);
        }
    }

    return contacts;
}

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
    double dt) {

    // Compute current vertex positions
    std::vector<Eigen::Vector3d> currBaseVertices_W;
    std::array<std::vector<Eigen::Vector3d>, 4> currArmVertices_W;
    
    currBaseVertices_W.reserve(hulls.baseHull_B.size());
    for (const auto& p_B : hulls.baseHull_B) {
        currBaseVertices_W.push_back(W_r_B + R_WB * (kBaseAlign * p_B));
    }
    
    for (int i = 0; i < 4; ++i) {
        const auto& armHull = hulls.armHull_P[i];
        const Eigen::Vector3d armOrigin_W = W_r_B + arms[i].W_r_BH;
        const Eigen::Matrix3d& R_WP = arms[i].R_WP;
        currArmVertices_W[i].reserve(armHull.size());
        for (const auto& p_P : armHull) {
            currArmVertices_W[i].push_back(armOrigin_W + R_WP * (kArmAlign * p_P));
        }
    }

    std::vector<ContactPoint> contacts;
    const std::size_t estimatedContacts =
        (hulls.baseHull_B.size() + 4 * hulls.armHull_P[0].size()) * planes.size();
    contacts.reserve(estimatedContacts);

    // Lambda to process a single vertex with CCD
    auto processVertexCCD = [&](const Eigen::Vector3d& x_curr,
                                 const Eigen::Vector3d& x_prev,
                                 const Eigen::Vector3d& bodyOrigin_W,
                                 const Eigen::Vector3d& omega_W,
                                 const Eigen::Vector3d& v_bodyOrigin_W,
                                 int bodyId,
                                 bool hasPrev) {
        for (const auto& plane : planes) {
            const double phi_curr = plane.n.dot(x_curr) + plane.d;
            
            // Standard detection: within activation distance
            bool inContact = (phi_curr < params.activationDistance);
            
            // CCD: check if we crossed the plane since last frame
            double penetration = 0.0;
            Eigen::Vector3d contactPoint = x_curr;
            
            if (hasPrev && params.enableCCD) {
                const double phi_prev = plane.n.dot(x_prev) + plane.d;
                
                // Tunneling detection: was above plane, now below (or in activation zone)
                if (phi_prev >= params.activationDistance && phi_curr < params.activationDistance) {
                    inContact = true;
                    // Compute intersection point along the trajectory
                    // x_intersection = x_prev + t * (x_curr - x_prev) where phi(x_int) = activation_distance
                    // phi_prev + t * (phi_curr - phi_prev) = activation_distance
                    double denom = phi_prev - phi_curr;
                    if (std::abs(denom) > 1e-12) {
                        double t = (phi_prev - params.activationDistance) / denom;
                        t = std::clamp(t, 0.0, 1.0);
                        contactPoint = x_prev + t * (x_curr - x_prev);
                    }
                }
            }
            
            if (!inContact) {
                continue;
            }
            
            // Compute penetration from current position
            penetration = params.activationDistance - phi_curr;
            if (penetration < 0.0) penetration = 0.0;
            
            // Compute velocity at contact point
            const Eigen::Vector3d r_W = contactPoint - bodyOrigin_W;
            const Eigen::Vector3d v_point = v_bodyOrigin_W + omega_W.cross(r_W);
            const double v_n = v_point.dot(plane.n);
            
            // Compute contact force
            const Eigen::Vector3d F_contact = contactForce(plane, penetration, v_n, v_point, params);
            if (F_contact.squaredNorm() < kForceEps) {
                continue;
            }
            
            ContactPoint cp;
            cp.x_W = contactPoint;
            cp.n_W = plane.n;
            cp.penetration = penetration;
            cp.force_W = F_contact;
            cp.bodyId = bodyId;
            contacts.push_back(cp);
        }
    };

    // Process base hull vertices
    const bool hasPrevBase = prevState.valid && 
                             prevState.baseVertices_W.size() == currBaseVertices_W.size();
    for (std::size_t j = 0; j < currBaseVertices_W.size(); ++j) {
        const Eigen::Vector3d x_prev = hasPrevBase ? prevState.baseVertices_W[j] : currBaseVertices_W[j];
        processVertexCCD(currBaseVertices_W[j], x_prev, W_r_B, W_omega_B, v_WB, 0, hasPrevBase);
    }

    // Process arm hull vertices
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector3d armOrigin_W = W_r_B + arms[i].W_r_BH;
        const Eigen::Vector3d v_bodyOrigin_W = v_WB + W_omega_B.cross(arms[i].W_r_BH);
        const Eigen::Vector3d omega_W = W_omega_P.col(i);
        
        const bool hasPrevArm = prevState.valid &&
                                prevState.armVertices_W[i].size() == currArmVertices_W[i].size();
        
        for (std::size_t j = 0; j < currArmVertices_W[i].size(); ++j) {
            const Eigen::Vector3d x_prev = hasPrevArm ? prevState.armVertices_W[i][j] : currArmVertices_W[i][j];
            processVertexCCD(currArmVertices_W[i][j], x_prev, armOrigin_W, omega_W, v_bodyOrigin_W, i + 1, hasPrevArm);
        }
    }

    // Update previous state for next frame
    prevState.baseVertices_W = std::move(currBaseVertices_W);
    prevState.armVertices_W = std::move(currArmVertices_W);
    prevState.valid = true;

    return contacts;
}
