#include "DroneDynamics.h"
#include "RungeKutta4.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/view.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <urdf_model/model.h>
#include <urdf_model/joint.h>
#include <urdf_model/pose.h>
#include <urdf_model/link.h>
#include <urdf_parser/urdf_parser.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

bool loadSTL(const std::string& filename,
             std::vector<Eigen::Vector3f>& vertices,
             std::vector<Eigen::Vector3i>& faces)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filename,
        aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    if (!scene || !scene->HasMeshes()) {
        std::cerr << "❌ Assimp failed to load: " << filename
                  << " (" << importer.GetErrorString() << ")" << std::endl;
        return false;
    }

    aiMesh* mesh = scene->mMeshes[0];
    vertices.reserve(mesh->mNumVertices);
    faces.reserve(mesh->mNumFaces);

    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        const aiVector3D& v = mesh->mVertices[i];
        vertices.emplace_back(v.x, v.y, v.z);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
        const aiFace& f = mesh->mFaces[i];
        if (f.mNumIndices == 3) {
            faces.emplace_back(f.mIndices[0], f.mIndices[1], f.mIndices[2]);
        }
    }
    return true;
}

Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw)
{
    Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
    return (Rz * Ry * Rx).matrix();
}

Eigen::Matrix4d poseToMatrix(const urdf::Pose& pose)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Vector3d t(pose.position.x, pose.position.y, pose.position.z);
    double roll, pitch, yaw;
    pose.rotation.getRPY(roll, pitch, yaw);
    T.block<3,3>(0,0) = rpyToMatrix(roll, pitch, yaw);
    T.block<3,1>(0,3) = t;
    return T;
}

struct LinkVisual {
    polyscope::SurfaceMesh* mesh{nullptr};
    Eigen::Matrix4f visualOffset{Eigen::Matrix4f::Identity()};
};

class UrdfRig {
public:
    UrdfRig() {
        Eigen::AngleAxisf Rx(-static_cast<float>(M_PI) / 2.0f, Eigen::Vector3f::UnitX());
        viewAdjust_.setIdentity();
        viewAdjust_.block<3,3>(0,0) = Rx.matrix();
    }

    bool initialize(const std::string& urdfPath) {
        std::ifstream urdf_file(urdfPath);
        if (!urdf_file.is_open()) {
            std::cerr << "❌ Cannot open URDF file: " << urdfPath << std::endl;
            return false;
        }

        std::string urdf_xml((std::istreambuf_iterator<char>(urdf_file)),
                             std::istreambuf_iterator<char>());

        model_ = urdf::parseURDF(urdf_xml);
        if (!model_) {
            std::cerr << "❌ Failed to parse URDF: " << urdfPath << std::endl;
            return false;
        }

        const fs::path urdfFs = fs::absolute(urdfPath);
        projectRoot_ = urdfFs.parent_path().parent_path().parent_path();
        if (!fs::exists(projectRoot_ / "graphics")) {
            projectRoot_ = fs::current_path();
        }
        assetRoot_ = projectRoot_ / "graphics/meshes";

        for (const auto& [name, link] : model_->links_) {
            if (!link || !link->visual || !link->visual->geometry) {
                continue;
            }
            auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(link->visual->geometry);
            if (!mesh) continue;

            std::string mesh_path = mesh->filename;
            if (mesh_path.rfind("package://", 0) == 0) {
                const fs::path relPath(mesh_path.substr(10));
                mesh_path = (assetRoot_ / relPath.filename()).string();
            } else if (!fs::path(mesh_path).is_absolute()) {
                mesh_path = (projectRoot_ / mesh_path).string();
            }
            if (!fs::exists(mesh_path)) {
                std::cerr << "⚠️  Missing mesh file: " << mesh_path << std::endl;
                continue;
            }

            std::vector<Eigen::Vector3f> vertices;
            std::vector<Eigen::Vector3i> faces;
            if (!loadSTL(mesh_path, vertices, faces)) continue;

            Eigen::Vector3f scale(1.f, 1.f, 1.f);
            scale = Eigen::Vector3f(
                static_cast<float>(mesh->scale.x),
                static_cast<float>(mesh->scale.y),
                static_cast<float>(mesh->scale.z));
            for (auto& v : vertices) {
                v = v.cwiseProduct(scale);
            }

            Eigen::Matrix4f T_visual = Eigen::Matrix4f::Identity();
            const urdf::Pose& vpose = link->visual->origin;
            Eigen::Vector3f vt(vpose.position.x, vpose.position.y, vpose.position.z);
            double vroll, vpitch, vyaw;
            vpose.rotation.getRPY(vroll, vpitch, vyaw);
            Eigen::AngleAxisf Rx(vroll, Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf Ry(vpitch, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf Rz(vyaw, Eigen::Vector3f::UnitZ());
            T_visual.block<3,3>(0,0) = (Rz * Ry * Rx).matrix();
            T_visual.block<3,1>(0,3) = vt;

            auto* psMesh = polyscope::registerSurfaceMesh(name, vertices, faces);
            psMesh->setSurfaceColor({0.3, 0.6, 0.8});
            psMesh->setSmoothShade(true);

            linkVisuals_[name] = {psMesh, T_visual};
        }

        if (linkVisuals_.empty()) {
            std::cerr << "❌ URDF loaded but no visual meshes were registered." << std::endl;
            return false;
        }

        std::cout << "Loaded " << linkVisuals_.size() << " link visuals from "
                  << urdfFs << std::endl;
        return true;
    }

    void update(const Eigen::Isometry3d& baseTransform,
                const std::unordered_map<std::string, double>& jointPositions) {
        if (!model_) return;

        auto root = model_->getRoot();
        if (!root) return;

        Eigen::Matrix4d base = Eigen::Matrix4d::Identity();
        base.block<3,3>(0,0) = baseTransform.rotation();
        base.block<3,1>(0,3) = baseTransform.translation();

        std::map<std::string, Eigen::Matrix4d> global;
        propagate(root, base, jointPositions, global);

        for (const auto& [name, vis] : linkVisuals_) {
            auto it = global.find(name);
            if (it == global.end()) continue;

            Eigen::Matrix4f T = (viewAdjust_ * it->second.cast<float>()) * vis.visualOffset;
            glm::mat4 transform(1.0f);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    transform[i][j] = T(j, i);
            vis.mesh->setTransform(transform);

        }
    }

private:
    urdf::ModelInterfaceSharedPtr model_;
    std::map<std::string, LinkVisual> linkVisuals_;
    Eigen::Matrix4f viewAdjust_{Eigen::Matrix4f::Identity()};
    fs::path projectRoot_;
    fs::path assetRoot_;

    void propagate(urdf::LinkConstSharedPtr link,
                   const Eigen::Matrix4d& parent,
                   const std::unordered_map<std::string, double>& jointPositions,
                   std::map<std::string, Eigen::Matrix4d>& global) {
        global[link->name] = parent;

        for (const auto& joint : link->child_joints) {
            if (!joint) continue;
            auto child = model_->getLink(joint->child_link_name);
            if (!child) continue;
            double value = 0.0;
            auto it = jointPositions.find(joint->name);
            if (it != jointPositions.end()) value = it->second;
            Eigen::Matrix4d T_child = parent * jointTransform(joint, value);
            propagate(child, T_child, jointPositions, global);
        }
    }

    Eigen::Matrix4d jointTransform(const urdf::JointConstSharedPtr& joint,
                                   double value) const {
        Eigen::Matrix4d T = poseToMatrix(joint->parent_to_joint_origin_transform);
        Eigen::Matrix4d motion = Eigen::Matrix4d::Identity();

        if (joint->type == urdf::Joint::REVOLUTE || joint->type == urdf::Joint::CONTINUOUS) {
            Eigen::Vector3d axis(joint->axis.x, joint->axis.y, joint->axis.z);
            if (axis.norm() < 1e-9) axis = Eigen::Vector3d::UnitZ();
            axis.normalize();
            motion.block<3,3>(0,0) = Eigen::AngleAxisd(value, axis).toRotationMatrix();
        } else if (joint->type == urdf::Joint::PRISMATIC) {
            Eigen::Vector3d axis(joint->axis.x, joint->axis.y, joint->axis.z);
            if (axis.norm() < 1e-9) axis = Eigen::Vector3d::UnitZ();
            axis.normalize();
            motion.block<3,1>(0,3) = axis * value;
        }

        return T * motion;
    }
};

struct PIDGains {
    Eigen::Vector3d kp{Eigen::Vector3d::Constant(2.0)};
    Eigen::Vector3d ki{Eigen::Vector3d::Constant(0.0)};
    Eigen::Vector3d kd{Eigen::Vector3d::Constant(1.0)};
};

class DroneSimulationApp {
public:
    DroneSimulationApp()
        : dynamics_("model/drone_parameters.yaml"),
          state_(Eigen::VectorXd::Zero(DroneDynamics::kStateSize)) {
        state_(3) = 1.0;
        for (int i = 0; i < 4; ++i) {
            state_(13 + 4 * i) = 1.0;
        }
        thrust_.setZero();
        std::cout << "Hover thrust per rotor: " << hoverThrust() << " N" << std::endl;
    }

    bool initializeScene(const std::string& urdfPath) {
        if (!rig_.initialize(urdfPath)) {
            return false;
        }
        rig_.update(baseTransform(), jointAngles());
        return true;
    }

    void step() {
        updateController();

        auto dyn = [this](const Eigen::VectorXd& x) {
            return dynamics_.derivative(x, thrust_);
        };

        for (int i = 0; i < substeps_; ++i) {
            state_ = integrator_.step(state_, dt_, dyn);
            normalizeQuaternions();
            simTime_ += dt_;
        }

        rig_.update(baseTransform(), jointAngles());
        updateCamera();

        if (simTime_ >= nextLogTime_) {
            logState();
            nextLogTime_ += logInterval_;
        }
    }

private:
    DroneDynamics dynamics_;
    RungeKutta4 integrator_;
    Eigen::VectorXd state_;
    Eigen::Vector4d thrust_;
    UrdfRig rig_;
    double simTime_{0.0};
    const double dt_{0.002};
    const int substeps_{5};
    double nextLogTime_{0.0};
    const double logInterval_{0.5};
    bool followBase_{false};
    Eigen::Vector3d positionRef_{Eigen::Vector3d(0.0, 0.0, 0.3)};
    Eigen::Vector3d integralError_{Eigen::Vector3d::Zero()};
    PIDGains gains_;

    void updateCamera() {
        if (!followBase_) return;
        Eigen::Vector3d pos = state_.segment<3>(0);
        Eigen::Quaterniond q(state_(3), state_(4), state_(5), state_(6));
        if (q.norm() > 1e-9) q.normalize();
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Vector3d offsetBody(0.0, -0.4, 0.25);
        Eigen::Vector3d camPos = pos + R * offsetBody;
        Eigen::Vector3d up = R * Eigen::Vector3d::UnitZ();

        glm::vec3 cam(static_cast<float>(camPos.x()),
                      static_cast<float>(camPos.y()),
                      static_cast<float>(camPos.z()));
        glm::vec3 target(static_cast<float>(pos.x()),
                         static_cast<float>(pos.y()),
                         static_cast<float>(pos.z()));
        glm::vec3 upVec(static_cast<float>(up.x()),
                        static_cast<float>(up.y()),
                        static_cast<float>(up.z()));
        polyscope::view::lookAt(cam, target, upVec);
    }

    void updateController() {
        Eigen::Vector3d p = state_.segment<3>(0);
        Eigen::Vector3d v = state_.segment<3>(7);

        Eigen::Vector3d errorPos = positionRef_ - p;
        integralError_ += errorPos * (dt_ * substeps_);
        Eigen::Vector3d errorVel = -v;

        Eigen::Vector3d accelCmd = gains_.kp.cwiseProduct(errorPos)
                                   + gains_.ki.cwiseProduct(integralError_)
                                   + gains_.kd.cwiseProduct(errorVel)
                                   + Eigen::Vector3d(0.0, 0.0, 9.8066);

        double thrustTotal = dynamics_.params().massTotal * accelCmd.z();
        thrustTotal = std::clamp(thrustTotal, 0.0, 4.0 * motorThrustMax());
        double perRotor = thrustTotal / 4.0;
        thrust_.setConstant(perRotor);
    }

    double motorThrustMax() const {
        const auto& params = dynamics_.params();
        return params.propellerMaxThrust.value_or(10.0);
    }

    void logState() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\n=== t = " << simTime_ << " s ===\n";
        Eigen::Vector3d p = state_.segment<3>(0);
        Eigen::Vector3d v = state_.segment<3>(7);
        Eigen::Vector3d w = state_.segment<3>(10);
        Eigen::Vector4d q = state_.segment<4>(3);
        oss << "p [m]: " << p.transpose() << " | v [m/s]: " << v.transpose() << "\n";
        oss << "w [rad/s]: " << w.transpose() << " | q_base: " << q.transpose() << "\n";
        for (int i = 0; i < 4; ++i) {
            Eigen::Vector4d qi = state_.segment<4>(13 + 4 * i);
            Eigen::Vector3d wi = state_.segment<3>(29 + 3 * i);
            oss << "arm_" << i << " q: " << qi.transpose()
                << " | rel_w [rad/s]: " << wi.transpose() << "\n";
        }
        oss << "thrust [N]: " << thrust_.transpose() << "\n";
        std::cout << oss.str() << std::endl;
    }

    double hoverThrust() const {
        return dynamics_.params().massTotal * 9.8066 / 4.0;
    }

    void normalizeQuaternions() {
        auto normalizeSegment = [this](int offset) {
            Eigen::Vector4d q = state_.segment<4>(offset);
            double n = q.norm();
            if (n > 1e-9) {
                state_.segment<4>(offset) = q / n;
            } else {
                state_.segment<4>(offset) << 1.0, 0.0, 0.0, 0.0;
            }
        };
        normalizeSegment(3);
        for (int i = 0; i < 4; ++i) {
            normalizeSegment(13 + 4 * i);
        }
    }

    Eigen::Isometry3d baseTransform() const {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = state_.segment<3>(0);
        Eigen::Quaterniond q(state_(3), state_(4), state_(5), state_(6));
        if (q.norm() > 1e-9) q.normalize();
        T.linear() = q.toRotationMatrix();
        return T;
    }

    std::unordered_map<std::string, double> jointAngles() const {
        std::unordered_map<std::string, double> joints;
        const auto eulers = dynamics_.armEulerZYX(state_);
        for (int i = 0; i < 4; ++i) {
            joints["base_link_to_connecting_link_" + std::to_string(i) + "_z"] = eulers[i].x();
            joints["base_link_to_connecting_link_" + std::to_string(i) + "_y"] = eulers[i].y();
            joints["base_link_to_connecting_link_" + std::to_string(i) + "_x"] = eulers[i].z();
        }
        return joints;
    }
};

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

int main(int argc, char** argv) {
    std::string backend;
    if (const char* forced = std::getenv("MORPHY_POLYSCOPE_BACKEND")) {
        backend = forced;
    } else if (std::getenv("DISPLAY") == nullptr) {
        backend = "openGL_mock";
        polyscope::options::usePrefsFile = false;
    }
    const bool runSimulation = (std::getenv("MORPHY_RUN_SIM") != nullptr);

    polyscope::init(backend);

    const fs::path urdfPath = resolveResource("graphics/urdf/morphy.urdf");
    if (!runSimulation) {
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
