#pragma once

#include <filesystem>

namespace scene {

void registerGroundPlaneMesh();
std::filesystem::path resolveResource(const std::filesystem::path& relative);

} // namespace scene
