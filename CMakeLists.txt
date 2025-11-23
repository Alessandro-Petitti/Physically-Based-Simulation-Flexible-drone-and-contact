cmake_minimum_required(VERSION 3.16)
project(MorphyViewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ============================================================
# 🔹 Eigen
# ============================================================
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# ============================================================
# 🔹 Polyscope (viewer)
# ============================================================
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v2.2.1
)
FetchContent_MakeAvailable(polyscope)

# ============================================================
# 🔹 tinyobjloader (eventuale supporto OBJ)
# ============================================================
FetchContent_Declare(
  tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG v2.0.0rc10
)
FetchContent_MakeAvailable(tinyobjloader)

# ============================================================
# 🔹 Assimp (robusto import di STL binari/ASCII)
# ============================================================
find_package(assimp REQUIRED)

# ============================================================
# 🔹 urdfdom (lettura URDF)
#  - Su Ubuntu 24.04 ARM64 i file non esportano i cmake targets,
#    quindi aggiungiamo manualmente gli include path e le librerie
# ============================================================
include_directories(/usr/include)
include_directories(/usr/include/urdfdom_headers)
include_directories(/usr/include/urdfdom)
include_directories(/usr/include/urdf_parser)

# Verifica librerie
find_library(URDFDOM_MODEL_LIB urdfdom_model REQUIRED)
find_library(URDFDOM_WORLD_LIB urdfdom_world REQUIRED)

# ============================================================
# 🔹 Executable
# ============================================================
add_executable(morphy_viewer
    main.cpp
    model/DroneDynamics.cpp
    model/DroneParameters.cpp
    model/SimpleYaml.cpp
)

target_include_directories(morphy_viewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/model)

target_link_libraries(morphy_viewer PRIVATE
    polyscope
    Eigen3::Eigen
    tinyobjloader
    assimp
    ${URDFDOM_MODEL_LIB}
    ${URDFDOM_WORLD_LIB}
)

# ============================================================
# 🔹 Messaggio di configurazione
# ============================================================
message(STATUS "URDFDOM model lib: ${URDFDOM_MODEL_LIB}")
message(STATUS "URDFDOM world lib: ${URDFDOM_WORLD_LIB}")
