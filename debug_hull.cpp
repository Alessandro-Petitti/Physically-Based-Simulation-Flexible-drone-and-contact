#include "model/HullLoader.h"
#include "src/SceneUtils.h"
#include <iostream>
#include <algorithm>
#include <limits>

int main() {
    try {
        const double scale = 0.001;
        auto hulls = loadConvexHullShapes(
            scene::resolveResource("graphics/hulls").string(), scale);
        
        std::cout << "\n=== BASE HULL ===\n";
        std::cout << "Number of points: " << hulls.baseHull_B.size() << "\n";
        if (!hulls.baseHull_B.empty()) {
            auto minZ = hulls.baseHull_B[0].z();
            auto maxZ = hulls.baseHull_B[0].z();
            for (const auto& p : hulls.baseHull_B) {
                minZ = std::min(minZ, p.z());
                maxZ = std::max(maxZ, p.z());
            }
            std::cout << "Z range: [" << minZ << ", " << maxZ << "]\n";
            std::cout << "First 3 points:\n";
            for (int i = 0; i < std::min(3, (int)hulls.baseHull_B.size()); ++i) {
                const auto& p = hulls.baseHull_B[i];
                std::cout << "  (" << p.x() << ", " << p.y() << ", " << p.z() << ")\n";
            }
        }
        
        std::cout << "\n=== ARM HULL (arm 0) ===\n";
        std::cout << "Number of points: " << hulls.armHull_P[0].size() << "\n";
        if (!hulls.armHull_P[0].empty()) {
            auto minZ = hulls.armHull_P[0][0].z();
            auto maxZ = hulls.armHull_P[0][0].z();
            for (const auto& p : hulls.armHull_P[0]) {
                minZ = std::min(minZ, p.z());
                maxZ = std::max(maxZ, p.z());
            }
            std::cout << "Z range: [" << minZ << ", " << maxZ << "]\n";
            std::cout << "First 5 points:\n";
            for (int i = 0; i < std::min(5, (int)hulls.armHull_P[0].size()); ++i) {
                const auto& p = hulls.armHull_P[0][i];
                std::cout << "  (" << p.x() << ", " << p.y() << ", " << p.z() << ")\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
