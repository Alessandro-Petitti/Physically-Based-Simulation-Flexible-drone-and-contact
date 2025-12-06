#include "ContactGeometry.h"
#include "DroneDynamics.h"
#include "HullLoader.h"
#include "SceneUtils.h"

#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::array<DroneDynamics::ArmKinematics,4> computeIdentityArms(const DroneDynamics& dyn) {
    std::array<DroneDynamics::ArmKinematics,4> arms;
    const auto& params = dyn.params();
    const Eigen::Matrix3d R_WB = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 4; ++i) {
        DroneDynamics::ArmKinematics arm;
        const auto& T_BH = params.T_BH[i];
        const auto& T_HP = params.T_HP[i];
        arm.R_BH0 = T_BH.block<3,3>(0,0);
        const Eigen::Vector3d B_r_BH = T_BH.block<3,1>(0,3);
        arm.R_H0H = Eigen::Matrix3d::Identity();
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

void printContacts(const std::vector<ContactPoint>& contacts) {
    double maxPen = 0.0;
    double maxForce = 0.0;
    for (const auto& c : contacts) {
        maxPen = std::max(maxPen, c.penetration);
        maxForce = std::max(maxForce, c.force_W.norm());
    }
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "contacts: " << contacts.size()
              << " | max penetration: " << maxPen
              << " m | max |F|: " << maxForce << " N\n";
    for (std::size_t i = 0; i < contacts.size(); ++i) {
        const auto& c = contacts[i];
        std::cout << "  [" << i << "] body " << c.bodyId
                  << " | pen: " << c.penetration
                  << " | x: " << c.x_W.transpose()
                  << " | F: " << c.force_W.transpose() << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string paramsPath = scene::resolveResource("model/drone_parameters.yaml").string();
        DroneDynamics dyn(paramsPath);

        double hullScale = 0.001;
        if (const char* env = std::getenv("MORPHY_HULL_SCALE")) {
            try { hullScale = std::stod(env); } catch (...) {}
        }
        const auto hulls = loadConvexHullShapes(scene::resolveResource("graphics/hulls").string(), hullScale);
        auto bboxZ = [](const std::vector<Eigen::Vector3d>& verts) {
            double zmin = std::numeric_limits<double>::infinity();
            double zmax = -std::numeric_limits<double>::infinity();
            for (const auto& v : verts) {
                zmin = std::min(zmin, v.z());
                zmax = std::max(zmax, v.z());
            }
            return std::make_pair(zmin, zmax);
        };
        const auto [baseZmin, baseZmax] = bboxZ(hulls.baseHull_B);

        double baseZ = -(baseZmin) + 0.001; // place lowest point 1 mm above plane
        if (argc == 2) {
            baseZ = std::stod(argv[1]);
        } else if (argc > 2) {
            std::cout << "Usage: " << argv[0] << " [base_z]\n";
            return 1;
        }

        std::vector<Plane> planes = {
            Plane{Eigen::Vector3d(0.0, 0.0, 1.0), 0.0} // floor z = 0
        };
        ContactParams contactParams;
        contactParams.contactStiffness = 20.0;
        contactParams.contactDamping = 0.5;
        contactParams.activationDistance = 0.0005;

        auto arms = computeIdentityArms(dyn);
        const Eigen::Vector3d W_r_B(0.0, 0.0, baseZ);
        const Eigen::Matrix3d R_WB = Eigen::Matrix3d::Identity();
        const Eigen::Vector3d v_WB = Eigen::Vector3d::Zero();
        const Eigen::Vector3d W_omega_B = Eigen::Vector3d::Zero();
        Eigen::Matrix<double,3,4> W_omega_P = Eigen::Matrix<double,3,4>::Zero();

        const auto contacts = computeContacts(
            arms.data(),
            W_r_B,
            R_WB,
            v_WB,
            W_omega_B,
            W_omega_P,
            hulls,
            planes,
            contactParams);

        std::cout << "Base z = " << baseZ << " m (base hull zmin=" << baseZmin
                  << ", zmax=" << baseZmax << ")\n";
        printContacts(contacts);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "contact_smoketest failed: " << e.what() << "\n";
        return 1;
    }
}
