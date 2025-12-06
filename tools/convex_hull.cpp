#include <tiny_obj_loader.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Polyhedron = CGAL::Polyhedron_3<Kernel>;

struct HullResult {
    Polyhedron poly;
};

bool readObjPoints(const fs::path& meshPath, std::vector<Point_3>& pts) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = false;
    config.vertex_color = false;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(meshPath.string(), config)) {
        std::cerr << "Failed to read OBJ: " << meshPath << "\n";
        if (!reader.Error().empty()) {
            std::cerr << reader.Error() << "\n";
        }
        return false;
    }
    const auto& attr = reader.GetAttrib();
    if (attr.vertices.empty()) {
        std::cerr << "OBJ has no vertices: " << meshPath << "\n";
        return false;
    }
    pts.reserve(attr.vertices.size() / 3);
    for (size_t i = 0; i + 2 < attr.vertices.size(); i += 3) {
        pts.emplace_back(attr.vertices[i + 0], attr.vertices[i + 1], attr.vertices[i + 2]);
    }
    return true;
}

bool computeHull(const fs::path& meshPath, HullResult& out) {
    std::vector<Point_3> pts;
    if (!readObjPoints(meshPath, pts)) {
        return false;
    }
    if (pts.size() < 4) {
        std::cerr << "Not enough points for 3D hull: " << meshPath << "\n";
        return false;
    }
    CGAL::convex_hull_3(pts.begin(), pts.end(), out.poly);
    if (out.poly.empty()) {
        std::cerr << "CGAL convex hull failed for: " << meshPath << "\n";
        return false;
    }
    return true;
}

bool writeHullOBJ(const fs::path& outPath, const HullResult& hull) {
    fs::create_directories(outPath.parent_path());
    std::ofstream ofs(outPath);
    if (!ofs) return false;

    // Map vertex handles to indices
    std::unordered_map<Polyhedron::Vertex_const_handle, int> vidx;
    int idx = 1;
    for (auto v = hull.poly.vertices_begin(); v != hull.poly.vertices_end(); ++v) {
        const auto& p = v->point();
        ofs << "v " << CGAL::to_double(p.x()) << " "
                    << CGAL::to_double(p.y()) << " "
                    << CGAL::to_double(p.z()) << "\n";
        vidx[v] = idx++;
    }

    for (auto f = hull.poly.facets_begin(); f != hull.poly.facets_end(); ++f) {
        ofs << "f";
        auto h = f->facet_begin();
        // Facet is assumed triangular
        do {
            ofs << " " << vidx[h->vertex()];
            ++h;
        } while (h != f->facet_begin());
        ofs << "\n";
    }
    return true;
}

int main(int /*argc*/, char** /*argv*/) {
    fs::path meshesDir = "graphics/meshes";
    fs::path outputDir = "graphics/hulls";
    if (const char* inDir = std::getenv("HULL_IN_DIR")) meshesDir = inDir;
    if (const char* outDir = std::getenv("HULL_OUT_DIR")) outputDir = outDir;

    std::cout << "[Hull] Input dir: " << meshesDir << "\n";
    std::cout << "[Hull] Output dir: " << outputDir << "\n";

    if (!fs::exists(meshesDir)) {
        std::cerr << "Input directory does not exist: " << meshesDir << "\n";
        return 1;
    }

    std::vector<fs::path> meshFiles;
    for (const auto& entry : fs::directory_iterator(meshesDir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(c));
        if (ext == ".obj") {
            meshFiles.push_back(entry.path());
        }
    }

    if (meshFiles.empty()) {
        std::cerr << "No OBJ mesh files found in " << meshesDir << "\n";
        return 1;
    }

    int ok = 0, fail = 0;
    for (const auto& meshPath : meshFiles) {
        std::cout << "[Hull] Processing " << meshPath.filename() << " ... ";
        HullResult hull;
        if (!computeHull(meshPath, hull)) {
            std::cout << "FAIL\n";
            ++fail;
            continue;
        }
        fs::path outPath = outputDir / (meshPath.stem().string() + "_hull.obj");
        if (!writeHullOBJ(outPath, hull)) {
            std::cout << "WRITE FAIL\n";
            ++fail;
            continue;
        }
        std::cout << "OK (" << hull.poly.size_of_vertices()
                  << " verts, " << hull.poly.size_of_facets() << " faces) -> "
                  << outPath << "\n";
        ++ok;
    }

    std::cout << "[Hull] Done. Success: " << ok << " | Fail: " << fail << "\n";
    return (fail == 0) ? 0 : 2;
}
