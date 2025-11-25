#include "SceneUtils.h"

#include <vector>

namespace fs = std::filesystem;

namespace scene {

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
