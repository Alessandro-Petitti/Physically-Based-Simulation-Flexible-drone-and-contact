#include "DroneSimulationApp.h"
#include "SceneUtils.h"
#include "UrdfRig.h"

#include <Eigen/Geometry>
#include <glm/glm.hpp>
#include <polyscope/polyscope.h>
#include <polyscope/view.h>

#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::string backend;
    if (const char* forced = std::getenv("MORPHY_POLYSCOPE_BACKEND")) {
        backend = forced;
    } else if (std::getenv("DISPLAY") == nullptr) {
        backend = "openGL_mock";
        polyscope::options::usePrefsFile = false;
    }
    const bool viewOnly = (std::getenv("MORPHY_VIEW_ONLY") != nullptr);

    polyscope::init(backend);
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    const fs::path urdfPath = scene::resolveResource("graphics/urdf/morphy.urdf");
    if (viewOnly) {
        UrdfRig rig;
        if (!rig.initialize(urdfPath.string())) {
            return 1;
        }
        rig.update(Eigen::Isometry3d::Identity(), {});
        polyscope::view::lookAt(glm::vec3(0.3f, -0.35f, 0.2f),
                                glm::vec3(0.0f, 0.0f, 0.0f));
        polyscope::show();
        return 0;
    }

    DroneSimulationApp app;
    if (!app.initializeScene(urdfPath.string())) {
        return 1;
    }
    polyscope::view::lookAt(glm::vec3(0.3f, -0.35f, 0.2f),
                            glm::vec3(0.0f, 0.0f, 0.0f));

    polyscope::state::userCallback = [&]() { app.step(); };
    polyscope::show();
    return 0;
}
