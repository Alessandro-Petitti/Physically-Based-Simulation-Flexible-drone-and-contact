#include "model/ContactGeometry.h"
#include "model/HullLoader.h"
#include "src/SceneUtils.h"
#include <iostream>
#include <iomanip>
#include <Eigen/Geometry>

int main() {
    try {
        const double scale = 0.001;
        auto hulls = loadConvexHullShapes(
            scene::resolveResource("graphics/hulls").string(), scale);
        
        std::cout << "\n=== ARM HULL ANALYSIS ===\n";
        std::cout << "Number of arm hull points: " << hulls.armHull_P[0].size() << "\n\n";
        
        // Try different rotations
        Eigen::Matrix3d rot_identity = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d rot_pi2_pos = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d rot_pi2_neg = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
        
        // Check a few points
        for (int idx : {0, 500, 1000, 2000}) {
            if (idx >= (int)hulls.armHull_P[0].size()) break;
            const auto& p_orig = hulls.armHull_P[0][idx];
            auto p_identity = rot_identity * p_orig;
            auto p_pi2_pos = rot_pi2_pos * p_orig;
            auto p_pi2_neg = rot_pi2_neg * p_orig;
            
            std::cout << "Point " << idx << ":\n";
            std::cout << "  Original:       " << std::fixed << std::setprecision(4) 
                      << p_orig.transpose() << "\n";
            std::cout << "  After identity: " << p_identity.transpose() << "\n";
            std::cout << "  After +pi/2:    " << p_pi2_pos.transpose() << "\n";
            std::cout << "  After -pi/2:    " << p_pi2_neg.transpose() << "\n\n";
        }
        
        // Check Z ranges
        std::cout << "\n=== Z RANGE ANALYSIS ===\n";
        double minZ_orig = hulls.armHull_P[0][0].z();
        double maxZ_orig = hulls.armHull_P[0][0].z();
        double minZ_pi2 = minZ_orig;
        double maxZ_pi2 = minZ_orig;
        
        for (const auto& p : hulls.armHull_P[0]) {
            auto p_pi2 = rot_pi2_pos * p;
            minZ_orig = std::min(minZ_orig, p.z());
            maxZ_orig = std::max(maxZ_orig, p.z());
            minZ_pi2 = std::min(minZ_pi2, p_pi2.z());
            maxZ_pi2 = std::max(maxZ_pi2, p_pi2.z());
        }
        
        std::cout << "Original Z range:  [" << minZ_orig << ", " << maxZ_orig << "]\n";
        std::cout << "After +pi/2 Z:     [" << minZ_pi2 << ", " << maxZ_pi2 << "]\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
