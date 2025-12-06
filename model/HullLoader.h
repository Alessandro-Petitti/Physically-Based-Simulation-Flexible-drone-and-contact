#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

std::vector<Eigen::Vector3d> loadHullVertices(const std::string& objPath, double scale = 1.0);

struct ConvexHullShapes {
    std::vector<Eigen::Vector3d> baseHull_B;
    std::array<std::vector<Eigen::Vector3d>,4> armHull_P;
};

ConvexHullShapes loadConvexHullShapes(const std::string& basePath, double scale = 1.0);
