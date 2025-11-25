#pragma once

#include <filesystem>

namespace scene {

std::filesystem::path resolveResource(const std::filesystem::path& relative);

} // namespace scene
