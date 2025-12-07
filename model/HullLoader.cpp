#include "HullLoader.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <Eigen/Dense>

namespace {
std::vector<Eigen::Vector3d> parseObjVertices(const std::string& objPath, double scale) {
    std::ifstream in(objPath);
    if (!in) {
        throw std::runtime_error("Failed to open OBJ: " + objPath);
    }
    std::vector<Eigen::Vector3d> vertices;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        if (line[0] != 'v' || (line.size() > 1 && !std::isspace(line[1]))) continue;
        std::istringstream iss(line.substr(1));
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        if (!(iss >> v.x() >> v.y() >> v.z())) {
            continue;
        }
        vertices.push_back(v * scale);
    }
    return vertices;
}

HullMesh parseObjMesh(const std::string& objPath, double scale) {
    std::ifstream in(objPath);
    if (!in) {
        throw std::runtime_error("Failed to open OBJ: " + objPath);
    }
    HullMesh mesh;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        
        // Parse vertices
        if (line[0] == 'v' && line.size() > 1 && std::isspace(line[1])) {
            std::istringstream iss(line.substr(1));
            Eigen::Vector3d v = Eigen::Vector3d::Zero();
            if (iss >> v.x() >> v.y() >> v.z()) {
                mesh.vertices.push_back(v * scale);
            }
            continue;
        }
        
        // Parse faces
        if (line[0] == 'f' && line.size() > 1 && std::isspace(line[1])) {
            std::istringstream iss(line.substr(1));
            std::vector<size_t> face;
            std::string token;
            while (iss >> token) {
                // Handle "v", "v/vt", "v/vt/vn", "v//vn" formats
                size_t slashPos = token.find('/');
                std::string indexStr = (slashPos != std::string::npos) ? token.substr(0, slashPos) : token;
                int idx = std::stoi(indexStr);
                // OBJ indices are 1-based
                face.push_back(static_cast<size_t>(idx - 1));
            }
            if (face.size() >= 3) {
                mesh.faces.push_back(face);
            }
            continue;
        }
    }
    return mesh;
}

std::string firstExisting(const std::vector<std::string>& candidates) {
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return {};
}
} // namespace

std::vector<Eigen::Vector3d> loadHullVertices(const std::string& objPath, double scale) {
    return parseObjVertices(objPath, scale);
}

HullMesh loadHullMesh(const std::string& objPath, double scale) {
    return parseObjMesh(objPath, scale);
}

ConvexHullShapes loadConvexHullShapes(const std::string& basePath, double scale) {
    ConvexHullShapes shapes;

    const std::vector<std::string> baseCandidates = {
        (std::filesystem::path(basePath) / "core_battery_transformed_hull.obj").string(),
        (std::filesystem::path(basePath) / "core_battery_hull.obj").string()
    };
    const std::string baseFile = firstExisting(baseCandidates);
    if (baseFile.empty()) {
        throw std::runtime_error("Base hull OBJ not found in " + basePath);
    }
    shapes.baseHull_B = loadHullVertices(baseFile, scale);

    // Use arm_transformed_hull.obj to match arm_transformed.stl used by URDF
    const std::vector<std::string> armCandidates = {
        (std::filesystem::path(basePath) / "arm_transformed_hull.obj").string(),
        (std::filesystem::path(basePath) / "arm_hull.obj").string()
    };
    const std::string armFile = firstExisting(armCandidates);
    if (armFile.empty()) {
        throw std::runtime_error("Arm hull OBJ not found in " + basePath);
    }
    for (int i = 0; i < 4; ++i) {
        shapes.armHull_P[i] = loadHullVertices(armFile, scale);
    }

    return shapes;
}

ConvexHullMeshes loadConvexHullMeshes(const std::string& basePath, double scale) {
    ConvexHullMeshes meshes;

    const std::vector<std::string> baseCandidates = {
        (std::filesystem::path(basePath) / "core_battery_transformed_hull.obj").string(),
        (std::filesystem::path(basePath) / "core_battery_hull.obj").string()
    };
    const std::string baseFile = firstExisting(baseCandidates);
    if (baseFile.empty()) {
        throw std::runtime_error("Base hull OBJ not found in " + basePath);
    }
    meshes.baseHull_B = loadHullMesh(baseFile, scale);

    // Use arm_transformed_hull.obj to match arm_transformed.stl used by URDF
    const std::vector<std::string> armCandidates = {
        (std::filesystem::path(basePath) / "arm_transformed_hull.obj").string(),
        (std::filesystem::path(basePath) / "arm_hull.obj").string()
    };
    const std::string armFile = firstExisting(armCandidates);
    if (armFile.empty()) {
        throw std::runtime_error("Arm hull OBJ not found in " + basePath);
    }
    for (int i = 0; i < 4; ++i) {
        meshes.armHull_P[i] = loadHullMesh(armFile, scale);
    }

    return meshes;
}
