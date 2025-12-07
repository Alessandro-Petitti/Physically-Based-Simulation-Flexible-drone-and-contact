#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

struct HullMesh {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::vector<size_t>> faces;
};

std::vector<Eigen::Vector3d> loadHullVertices(const std::string& objPath, double scale = 1.0);
HullMesh loadHullMesh(const std::string& objPath, double scale = 1.0);

struct ConvexHullShapes {
    std::vector<Eigen::Vector3d> baseHull_B;
    std::array<std::vector<Eigen::Vector3d>,4> armHull_P;
};

struct ConvexHullMeshes {
    HullMesh baseHull_B;
    std::array<HullMesh, 4> armHull_P;
};

ConvexHullShapes loadConvexHullShapes(const std::string& basePath, double scale = 1.0);
ConvexHullMeshes loadConvexHullMeshes(const std::string& basePath, double scale = 1.0);
