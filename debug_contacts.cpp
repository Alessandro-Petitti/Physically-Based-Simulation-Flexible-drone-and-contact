#include "ContactGeometry.h"
#include "DroneDynamics.h"
#include "HullLoader.h"
#include "SceneUtils.h"

#include <Eigen/Geometry>
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    // Load hulls
    const double hullScale = 0.001;
    auto hulls = loadConvexHullShapes(scene::resolveResource("graphics/hulls").string(), hullScale);
    
    std::cout << std::fixed << std::setprecision(4);
    
    // Simulated drone state: at z=1.5m, no rotation
    Eigen::Vector3d W_r_B(0.0, 0.0, 1.5);  // Base position in world
    Eigen::Matrix3d R_WB = Eigen::Matrix3d::Identity();  // No rotation
    Eigen::Vector3d v_WB = Eigen::Vector3d::Zero();
    Eigen::Vector3d W_omega_B = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 4> W_omega_P = Eigen::Matrix<double, 3, 4>::Zero();
    
    // Create fake arm kinematics (arms straight out, at rest)
    DroneDynamics::ArmKinematics arms[4];
    
    // Arm positions from URDF (approximate)
    std::array<Eigen::Vector3d, 4> armOffsets = {
        Eigen::Vector3d(0.089, -0.081, 0.0),
        Eigen::Vector3d(-0.089, 0.081, 0.0),
        Eigen::Vector3d(0.089, 0.081, 0.0),
        Eigen::Vector3d(-0.089, -0.081, 0.0)
    };
    
    for (int i = 0; i < 4; ++i) {
        arms[i].W_r_BP = W_r_B + R_WB * armOffsets[i];
        arms[i].R_WP = R_WB;  // No rotation for simplicity
    }
    
    // Ground plane at z=0
    std::vector<Plane> planes = { Plane{Eigen::Vector3d(0.0, 0.0, 1.0), 0.0} };
    
    // Contact params
    ContactParams params;
    params.contactStiffness = 20.0;
    params.contactDamping = 0.5;
    params.activationDistance = 0.0005;
    
    std::cout << "\n=== DRONE STATE ===\n";
    std::cout << "Base position W_r_B: " << W_r_B.transpose() << "\n";
    std::cout << "Ground plane at z = 0.0, activation distance = " << params.activationDistance << " m\n";
    
    std::cout << "\n=== ARM POSITIONS ===\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "Arm " << i << " origin (W_r_BP): " << arms[i].W_r_BP.transpose() << "\n";
    }
    
    // Compute contacts
    auto contacts = computeContacts(arms, W_r_B, R_WB, v_WB, W_omega_B, W_omega_P, hulls, planes, params);
    
    std::cout << "\n=== CONTACT ANALYSIS ===\n";
    std::cout << "Total contacts detected: " << contacts.size() << "\n";
    
    if (!contacts.empty()) {
        // Find min/max z of contact points
        double minZ = contacts[0].x_W.z();
        double maxZ = contacts[0].x_W.z();
        for (const auto& c : contacts) {
            minZ = std::min(minZ, c.x_W.z());
            maxZ = std::max(maxZ, c.x_W.z());
        }
        std::cout << "Contact point Z range: [" << minZ << ", " << maxZ << "]\n";
        
        // Show first 5 contacts
        std::cout << "\nFirst 5 contact points:\n";
        for (size_t i = 0; i < std::min(size_t(5), contacts.size()); ++i) {
            const auto& c = contacts[i];
            std::cout << "  Contact " << i << ": x_W = " << c.x_W.transpose() 
                      << " | bodyId = " << c.bodyId 
                      << " | penetration = " << c.penetration << " m\n";
        }
        
        // Count by body
        int baseCount = 0, armCount = 0;
        for (const auto& c : contacts) {
            if (c.bodyId == 0) baseCount++;
            else armCount++;
        }
        std::cout << "\nContacts by body: base=" << baseCount << ", arms=" << armCount << "\n";
    }
    
    // Now check raw hull points
    std::cout << "\n=== RAW HULL POINT CHECK ===\n";
    
    // kArmAlign rotation
    Eigen::Matrix3d kArmAlign = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
    
    std::cout << "kArmAlign rotation matrix:\n" << kArmAlign << "\n";
    
    // Check some arm hull points after transformation
    std::cout << "\nArm hull point transformations (first 3 points):\n";
    for (int i = 0; i < std::min(3, (int)hulls.armHull_P[0].size()); ++i) {
        const auto& p_P = hulls.armHull_P[0][i];
        Eigen::Vector3d rotated = kArmAlign * p_P;
        Eigen::Vector3d x_W = arms[0].W_r_BP + arms[0].R_WP * rotated;
        
        std::cout << "  p_P = " << p_P.transpose() 
                  << " -> rotated = " << rotated.transpose()
                  << " -> x_W = " << x_W.transpose() << "\n";
    }
    
    // Check base hull points
    std::cout << "\nBase hull point transformations (first 3 points):\n";
    for (int i = 0; i < std::min(3, (int)hulls.baseHull_B.size()); ++i) {
        const auto& p_B = hulls.baseHull_B[i];
        Eigen::Vector3d x_W = W_r_B + R_WB * p_B;
        
        std::cout << "  p_B = " << p_B.transpose() 
                  << " -> x_W = " << x_W.transpose() << "\n";
    }
    
    return 0;
}
