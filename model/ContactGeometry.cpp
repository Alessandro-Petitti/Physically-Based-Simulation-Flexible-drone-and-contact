#include "ContactGeometry.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kForceEps = 1e-12;

// BASE HULL FRAME: The URDF applies rpy="0 0 1.57079" (+π/2 around Z) to core_battery_transformed.stl
// The hull file core_battery_transformed_hull.obj is in the same frame, so we need the same rotation.
const Eigen::Matrix3d kBaseAlign = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();

// ARM HULL FRAME: We now use arm_transformed_hull.obj which matches arm_transformed.stl.
// The R_WP from computeArmFrames already includes the joint rotation (π/2 around Y via T_HP).
// No additional alignment needed.
const Eigen::Matrix3d kArmAlign = Eigen::Matrix3d::Identity();

inline Eigen::Vector3d contactForce(const Plane& plane,
                                    double penetration,
                                    double v_n,
                                    const ContactParams& params) {
    const Eigen::Vector3d Fn = params.contactStiffness * penetration * plane.n;
    const Eigen::Vector3d Fd = -params.contactDamping * v_n * plane.n;
    return Fn + Fd;
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
            const Eigen::Vector3d F_contact = contactForce(plane, penetration, v_n, params);
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
    for (int i = 0; i < 4; ++i) {
        const auto& armHull = hulls.armHull_P[i];
        // W_r_BP is the propeller position RELATIVE to base, so add W_r_B for absolute position
        const Eigen::Vector3d armOrigin_W = W_r_B + arms[i].W_r_BP;
        const Eigen::Matrix3d& R_WP = arms[i].R_WP;

        Eigen::Vector3d v_bodyOrigin_W = v_WB + W_omega_B.cross(arms[i].W_r_BP);
        const Eigen::Vector3d omega_W = W_omega_P.col(i);

        for (const auto& p_P : armHull) {
            const Eigen::Vector3d x_W = armOrigin_W + R_WP * (kArmAlign * p_P);
            processVertex(x_W, armOrigin_W, omega_W, v_bodyOrigin_W, i + 1);
        }
    }

    return contacts;
}
