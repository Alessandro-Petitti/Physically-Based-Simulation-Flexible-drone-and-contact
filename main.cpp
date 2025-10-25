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

    std::cout << "✅ Loaded " << filename << " with "
              << vertices.size() << " vertices and "
              << faces.size() << " faces.\n";
    return true;
}

int main(int argc, char** argv) {

    polyscope::init();

    // Default mesh folder: graphics/meshes
    std::string mesh_folder = "graphics/meshes";
    if (argc > 1)
        mesh_folder = argv[1];

    // Check if folder exists
    if (!fs::exists(mesh_folder)) {
        std::cerr << "❌ Folder not found: " << mesh_folder << std::endl;
        return 1;
    }

    bool anyLoaded = false;

    // Loop over all STL files in the folder
    for (const auto& entry : fs::directory_iterator(mesh_folder)) {
        if (!entry.is_regular_file()) continue;

        std::string path = entry.path().string();
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".stl") {
            std::vector<Eigen::Vector3f> vertices;
            std::vector<Eigen::Vector3i> faces;

            if (loadSTL(path, vertices, faces)) {
                std::string name = entry.path().stem().string();
                auto* mesh = polyscope::registerSurfaceMesh(name, vertices, faces);
                mesh->setSurfaceColor({0.3, 0.6, 0.8});
                mesh->setSmoothShade(true);
                anyLoaded = true;
            }
        }
    }

    if (!anyLoaded) {
        std::cerr << "⚠️  No STL files found in " << mesh_folder << std::endl;
    }

    polyscope::show();
    return 0;
}
