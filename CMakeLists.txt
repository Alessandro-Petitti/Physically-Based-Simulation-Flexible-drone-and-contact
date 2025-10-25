cmake_minimum_required(VERSION 3.16)
project(MorphyViewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# Polyscope (updated tag)
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v2.2.1   # ✅ valid as of 2025
)
FetchContent_MakeAvailable(polyscope)

# tinyobjloader (per leggere STL/OBJ)
FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG v2.0.0rc10
)
FetchContent_MakeAvailable(tinyobjloader)

add_executable(morphy_viewer main.cpp)
target_link_libraries(morphy_viewer PRIVATE polyscope Eigen3::Eigen tinyobjloader)


find_package(assimp REQUIRED)
target_link_libraries(morphy_viewer PRIVATE assimp)

find_package(urdfdom REQUIRED)
target_link_libraries(morphy_viewer PRIVATE urdfdom_model urdfdom_world)