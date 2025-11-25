#include "SceneUtils.h"

#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <vector>

namespace fs = std::filesystem;

namespace scene {

void registerGroundPlaneMesh() {
    std::vector<Eigen::Vector3f> vertices = {
        {-2.5f, -2.5f, 0.0f},
        { 2.5f, -2.5f, 0.0f},
        { 2.5f,  2.5f, 0.0f},
        {-2.5f,  2.5f, 0.0f}
    };
    std::vector<Eigen::Vector3i> faces = {
        {0, 1, 2},
        {0, 2, 3}
    };
    auto* ground = polyscope::registerSurfaceMesh("ground_plane", vertices, faces);
    ground->setSurfaceColor({0.75f, 0.75f, 0.75f});
    ground->setTransparency(0.1f);
    ground->setSmoothShade(false);
    ground->setBackFacePolicy(polyscope::BackFacePolicy::Different);
}

fs::path resolveResource(const fs::path& relative) {
    std::vector<fs::path> candidates = {
        fs::current_path(),
        fs::current_path().parent_path(),
        fs::current_path().parent_path().parent_path()
    };
    for (const auto& base : candidates) {
        if (base.empty()) continue;
        fs::path candidate = base / relative;
        if (fs::exists(candidate)) {
            return fs::canonical(candidate);
        }
    }
    return relative;
}

} // namespace scene
