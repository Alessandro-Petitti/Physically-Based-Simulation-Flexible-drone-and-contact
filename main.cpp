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

// === Load URDF and display meshes ===
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

    for (auto& [name, link] : model->links_) {
        if (!link->visual || !link->visual->geometry) continue;

        auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(link->visual->geometry);
        if (!mesh) continue;

        std::string mesh_path = mesh->filename;
        if (mesh_path.rfind("package://", 0) == 0) {
            mesh_path = "graphics/meshes/" +
                        fs::path(mesh_path.substr(10)).filename().string();
        }

        if (!fs::exists(mesh_path)) {
            std::cerr << "⚠️  Missing mesh file: " << mesh_path << std::endl;
            continue;
        }

        // Load STL
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> faces;
        if (!loadSTL(mesh_path, vertices, faces)) continue;

        // Compute transform
        Eigen::Matrix4f T = makeTransform(link->visual->origin);

        // Register in Polyscope
        polyscope::SurfaceMesh* psMesh =
            polyscope::registerSurfaceMesh(name, vertices, faces);

        // Convert Eigen::Matrix4f → glm::mat4
        glm::mat4 transform;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                transform[i][j] = T(j, i);  // note the transpose!

        psMesh->setTransform(transform);
        psMesh->setSurfaceColor({0.3, 0.6, 0.8});
        psMesh->setSmoothShade(true);

        std::cout << "Link: " << name << "  (" << mesh_path << ")\n";
    }
}

int main()
{
    polyscope::init();

    // Load URDF and display all links
    loadURDFAndDisplay("graphics/urdf/morphy.urdf");

    polyscope::show();
    return 0;
}
