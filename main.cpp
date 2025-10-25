#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <glm/glm.hpp>

#include <urdf_parser/urdf_parser.h>
#include <urdf_model/model.h>
#include <urdf_model/link.h>
#include <urdf_model/pose.h>

namespace fs = std::filesystem;

// === Helper to load STL using Assimp ===
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

// === Helper: build a 4x4 transform matrix from xyz + rpy ===
Eigen::Matrix4f makeTransform(const urdf::Pose& pose)
{
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::Vector3f t(pose.position.x, pose.position.y, pose.position.z);

    double roll, pitch, yaw;
    pose.rotation.getRPY(roll, pitch, yaw);

    Eigen::AngleAxisf Rx(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf Ry(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf Rz(yaw, Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f R = (Rz * Ry * Rx).matrix();

    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;
}

void loadURDFAndDisplay(const std::string& urdf_path)
{
    std::ifstream urdf_file(urdf_path);
    if (!urdf_file.is_open()) {
        std::cerr << "❌ Cannot open URDF file: " << urdf_path << std::endl;
        return;
    }

    std::string urdf_xml((std::istreambuf_iterator<char>(urdf_file)),
                         std::istreambuf_iterator<char>());

    urdf::ModelInterfaceSharedPtr model = urdf::parseURDF(urdf_xml);
    if (!model) {
        std::cerr << "❌ Failed to parse URDF: " << urdf_path << std::endl;
        return;
    }

    std::cout << "✅ Loaded URDF model: " << model->getName() << std::endl;

    // === Mappa nome link → trasformazione globale ===
    std::map<std::string, Eigen::Matrix4f> global_T;

    // === Ricorsione robusta per calcolare le pose globali ===
    std::function<void(urdf::LinkConstSharedPtr, const Eigen::Matrix4f&)> recurse;
    recurse = [&](urdf::LinkConstSharedPtr link, const Eigen::Matrix4f& parent_T) {
        global_T[link->name] = parent_T;

        for (const auto& joint : link->child_joints) {
            if (!joint) continue;

            auto child_link = model->getLink(joint->child_link_name);
            if (!child_link) continue;

            // Trasformazione relativa (pose del joint)
            const urdf::Pose& pose = joint->parent_to_joint_origin_transform;
            Eigen::Matrix4f T_rel = Eigen::Matrix4f::Identity();

            Eigen::Vector3f t(pose.position.x, pose.position.y, pose.position.z);
            double roll, pitch, yaw;
            pose.rotation.getRPY(roll, pitch, yaw);

            Eigen::AngleAxisf Rx(roll, Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf Ry(pitch, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf Rz(yaw, Eigen::Vector3f::UnitZ());
            T_rel.block<3,3>(0,0) = (Rz * Ry * Rx).matrix();
            T_rel.block<3,1>(0,3) = t;

            Eigen::Matrix4f T_child = parent_T * T_rel;
            recurse(child_link, T_child);
        }
    };

    auto root_link = model->getRoot();
    if (root_link)
        recurse(root_link, Eigen::Matrix4f::Identity());

    // === Rotazione globale per allineare l'URDF con Polyscope (Z-up) ===
    Eigen::Matrix4f R_global = Eigen::Matrix4f::Identity();
    float angle = -M_PI / 2.0f;  // -90° intorno all’asse X
    Eigen::AngleAxisf Rx(angle, Eigen::Vector3f::UnitX());
    R_global.block<3,3>(0,0) = Rx.matrix();

    // Applica la rotazione globale a tutte le trasformazioni calcolate
    for (auto& [name, T] : global_T) {
        T = R_global * T;
    }

    // === Visualizza le mesh ===
    for (auto& [name, link] : model->links_) {
        if (!link->visual || !link->visual->geometry) continue;

        auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(link->visual->geometry);
        if (!mesh) continue;

        std::string mesh_path = mesh->filename;
        if (mesh_path.rfind("package://", 0) == 0) {
            mesh_path = "graphics/meshes/" + fs::path(mesh_path.substr(10)).filename().string();
        }

        if (!fs::exists(mesh_path)) {
            std::cerr << "⚠️  Missing mesh file: " << mesh_path << std::endl;
            continue;
        }

        // === Carica la mesh STL ===
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> faces;
        if (!loadSTL(mesh_path, vertices, faces)) continue;

        // === Applica la scala del <mesh scale="..."> ===
        Eigen::Vector3f scale(1.f, 1.f, 1.f);
        if (mesh) {
            scale = Eigen::Vector3f(
                static_cast<float>(mesh->scale.x),
                static_cast<float>(mesh->scale.y),
                static_cast<float>(mesh->scale.z));
        }

        for (auto& v : vertices)
            v = v.cwiseProduct(scale);

        // === Trasformazione globale del link ===
        Eigen::Matrix4f T_link = global_T[name];

        // === Offset locale del visual (origin del tag <visual>) ===
        Eigen::Matrix4f T_visual = Eigen::Matrix4f::Identity();
        if (link->visual) {
            const urdf::Pose& vpose = link->visual->origin;
            Eigen::Vector3f vt(vpose.position.x, vpose.position.y, vpose.position.z);
            double vroll, vpitch, vyaw;
            vpose.rotation.getRPY(vroll, vpitch, vyaw);
            Eigen::AngleAxisf Rx(vroll, Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf Ry(vpitch, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf Rz(vyaw, Eigen::Vector3f::UnitZ());
            T_visual.block<3,3>(0,0) = (Rz * Ry * Rx).matrix();
            T_visual.block<3,1>(0,3) = vt;
        }

        // === Trasformazione totale ===
        Eigen::Matrix4f T_total = T_link * T_visual;

        // === Converti in glm::mat4 per Polyscope ===
        glm::mat4 transform;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                transform[i][j] = T_total(j, i);

        // === Registra in Polyscope ===
        auto* psMesh = polyscope::registerSurfaceMesh(name, vertices, faces);
        psMesh->setTransform(transform);
        psMesh->setSurfaceColor({0.3, 0.6, 0.8});
        psMesh->setSmoothShade(true);

        // Debug log
        Eigen::Vector3f t_global = T_link.block<3,1>(0,3);
        std::cout << "Link: " << name
                  << " rendered at global pos = "
                  << t_global.transpose()
                  << " scale = " << scale.transpose() << std::endl;
    }
}



int main(int argc, char** argv) {
    polyscope::init();
    loadURDFAndDisplay("graphics/urdf/morphy.urdf");
    polyscope::show();
    return 0;
}
