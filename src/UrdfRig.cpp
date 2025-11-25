#include "UrdfRig.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/glm.hpp>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <urdf_parser/urdf_parser.h>

#include <cmath>
#include <fstream>
#include <iterator>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

namespace {

Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw) {
    Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
    return (Rz * Ry * Rx).matrix();
}

Eigen::Matrix4d poseToMatrix(const urdf::Pose& pose) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Vector3d t(pose.position.x, pose.position.y, pose.position.z);
    double roll, pitch, yaw;
    pose.rotation.getRPY(roll, pitch, yaw);
    T.block<3,3>(0,0) = rpyToMatrix(roll, pitch, yaw);
    T.block<3,1>(0,3) = t;
    return T;
}

bool loadSTL(const std::string& filename,
             std::vector<Eigen::Vector3f>& vertices,
             std::vector<Eigen::Vector3i>& faces) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        filename,
        aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    if (!scene || !scene->HasMeshes()) {
        std::cerr << "Error: Assimp failed to load " << filename
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

} // namespace

UrdfRig::UrdfRig() {
    Eigen::AngleAxisf Rx(-static_cast<float>(M_PI) / 2.0f, Eigen::Vector3f::UnitX());
    viewAdjust_.setIdentity();
    viewAdjust_.block<3,3>(0,0) = Rx.matrix();
}

bool UrdfRig::initialize(const std::string& urdfPath) {
    std::ifstream urdf_file(urdfPath);
    if (!urdf_file.is_open()) {
        std::cerr << "Error: cannot open URDF file: " << urdfPath << std::endl;
        return false;
    }

    std::string urdf_xml((std::istreambuf_iterator<char>(urdf_file)),
                         std::istreambuf_iterator<char>());

    model_ = urdf::parseURDF(urdf_xml);
    if (!model_) {
        std::cerr << "Error: failed to parse URDF: " << urdfPath << std::endl;
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
            std::cerr << "Warning: missing mesh file: " << mesh_path << std::endl;
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
        psMesh->setSurfaceColor({0.3f, 0.6f, 0.8f});
        psMesh->setSmoothShade(true);

        linkVisuals_[name] = {psMesh, T_visual};
    }

    if (linkVisuals_.empty()) {
        std::cerr << "Error: URDF loaded but no visual meshes were registered." << std::endl;
        return false;
    }

    std::cout << "Loaded " << linkVisuals_.size() << " link visuals from "
              << urdfFs << std::endl;
    return true;
}

void UrdfRig::update(const Eigen::Isometry3d& baseTransform,
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

void UrdfRig::propagate(urdf::LinkConstSharedPtr link,
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

Eigen::Matrix4d UrdfRig::jointTransform(const urdf::JointConstSharedPtr& joint,
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
