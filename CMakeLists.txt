cmake_minimum_required(VERSION 3.16)
project(MorphyViewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

option(ENABLE_MESH_HULL "Build convex hull utility (libigl)" OFF)

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
    src/SceneUtils.cpp
    src/UrdfRig.cpp
    src/DroneSimulationApp.cpp
    model/DroneDynamics.cpp
    model/DroneParameters.cpp
    model/SimpleYaml.cpp
    model/ContactGeometry.cpp
    model/HullLoader.cpp
)

target_include_directories(morphy_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(morphy_viewer PRIVATE
    polyscope
    Eigen3::Eigen
    tinyobjloader
    assimp
    ${URDFDOM_MODEL_LIB}
    ${URDFDOM_WORLD_LIB}
)

# ============================================================
# 🔹 Collision check (headless free-fall → ground contact)
# ============================================================
add_executable(collision_check
    collision_check.cpp
    src/SceneUtils.cpp
    model/DroneDynamics.cpp
    model/DroneParameters.cpp
    model/SimpleYaml.cpp
    model/ContactGeometry.cpp
    model/HullLoader.cpp
)
target_include_directories(collision_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(collision_check PRIVATE
    Eigen3::Eigen
)

# ============================================================
# 🔹 Contact smoke test (headless; prints contact summary)
# ============================================================
add_executable(contact_smoketest
    tools/contact_smoketest.cpp
    src/SceneUtils.cpp
    model/DroneDynamics.cpp
    model/DroneParameters.cpp
    model/SimpleYaml.cpp
    model/ContactGeometry.cpp
    model/HullLoader.cpp
)
target_include_directories(contact_smoketest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(contact_smoketest PRIVATE
    Eigen3::Eigen
)

# ============================================================
# 🔹 Contact drop test (free fall with contact forces)
# ============================================================
add_executable(contact_drop
    tools/contact_drop.cpp
    src/SceneUtils.cpp
    model/DroneDynamics.cpp
    model/DroneParameters.cpp
    model/SimpleYaml.cpp
    model/ContactGeometry.cpp
    model/HullLoader.cpp
)
target_include_directories(contact_drop PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(contact_drop PRIVATE
    Eigen3::Eigen
)

# ============================================================
# 🔹 Debug hull (inspect hull point coordinates)
# ============================================================
add_executable(debug_hull
    debug_hull.cpp
    src/SceneUtils.cpp
    model/HullLoader.cpp
)
target_include_directories(debug_hull PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(debug_hull PRIVATE
    Eigen3::Eigen
)

# ============================================================
# 🔹 Debug contacts (analyze contact point positions)
# ============================================================
add_executable(debug_contacts
    debug_contacts.cpp
    src/SceneUtils.cpp
    model/HullLoader.cpp
    model/ContactGeometry.cpp
    model/DroneDynamics.cpp
    model/DroneParameters.cpp
    model/SimpleYaml.cpp
)
target_include_directories(debug_contacts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(debug_contacts PRIVATE
    Eigen3::Eigen
)

# ============================================================
# 🔹 Debug rotation (test arm hull rotations)
# ============================================================
add_executable(debug_rotation
    debug_rotation.cpp
    src/SceneUtils.cpp
    model/HullLoader.cpp
)
target_include_directories(debug_rotation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(debug_rotation PRIVATE
    Eigen3::Eigen
)

# ============================================================
# 🔹 Convex hull utility (graphics/meshes -> graphics/hulls)
#    Uses system CGAL + tinyobjloader (no libigl)
# ============================================================
if (ENABLE_MESH_HULL)
  find_package(CGAL REQUIRED COMPONENTS Core)

  add_executable(mesh_convex_hull
      tools/convex_hull.cpp
  )
  target_include_directories(mesh_convex_hull PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/model
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(mesh_convex_hull PRIVATE
      CGAL::CGAL
      tinyobjloader
      Eigen3::Eigen
  )
endif()

# ============================================================
# 🔹 Messaggio di configurazione
# ============================================================
message(STATUS "URDFDOM model lib: ${URDFDOM_MODEL_LIB}")
message(STATUS "URDFDOM world lib: ${URDFDOM_WORLD_LIB}")
