# Physically Based Simulation – Flexible Drone & Contact

This project visualizes and simulates a morphing quadrotor in real time. It couples a C++ port of the original CasADi dynamics with a Polyscope viewer that renders the URDF geometry while the state is numerically integrated. You can switch between a pure viewer mode (to inspect the mesh hierarchy) or a full simulation mode with a simple PID position controller.

## Features

- **Parametric dynamics**: The full 41‑state, 18‑constraint system from the Python/CasADi model is reimplemented in C++ using Eigen for linear algebra. The 18×18 block system is assembled every frame to recover linear and angular accelerations for the base and each arm.
- **Configurable parameters**: All masses, inertias, propeller constants, joint stiffness/damping and rigid transforms are read from `model/drone_parameters.yaml`. No hard-coded values live in the source code.
- **Polyscope viewer**: Meshes from the URDF (`graphics/urdf/morphy.urdf`) are registered via Assimp. The Polyscope ground plane is off by default; enable it from the UI if you need a visual reference at `z = 0`.
- **Configurable integrators**: Choose between explicit Euler, RK4, implicit Euler, or implicit midpoint IRK. Time step, substeps, and implicit solver settings are read from the YAML.
- **PID hover controller**: A simple PID tracks a desired world-frame position (default `(1, 1, 1)` m) and distributes the required thrust evenly across motors, saturating at the per-rotor maximum defined in the YAML.

## Dependencies

The project uses CMake to pull in the required libraries via `FetchContent`:

- [Eigen 3.4](https://gitlab.com/libeigen/eigen)
- [Polyscope 2.2](https://github.com/nmwsharp/polyscope) with GLFW backend
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)
- [Assimp](https://www.assimp.org/) (found from the system)
- URDFDOM headers/libraries (also from the system)

Ensure the following packages/tools are installed on your system:

- CMake ≥ 3.16
- A C++17 compiler (tested with GCC 13)
- Git (for the FetchContent dependencies)
- `libassimp-dev`, `liburdfdom-dev` (or the equivalents on your platform)

On Ubuntu you can install the runtime prerequisites via:

```bash
sudo apt update
sudo apt install build-essential cmake git libassimp-dev liburdfdom-dev
```

## Building

Clone the repository (submodules are not used; all third-party code is fetched at configure time) and run:

```bash
cmake -S . -B build
cmake --build build
```

This produces the `build/morphy_viewer` executable. If you see `Exec format error` or similar, remove the `build/` directory and reconfigure.

## Running

- **Simulation mode (default)**: integrates the dynamics with PID control toward `(1, 1, 1)` m.

  ```bash
  ./build/morphy_viewer
  ```

- **Viewer-only mode**: load the URDF meshes without running dynamics (useful for inspecting transforms).

  ```bash
  MORPHY_VIEW_ONLY=1 ./build/morphy_viewer
  ```

The viewer uses Polyscope with a GLFW window. If you are running headless, set `MORPHY_POLYSCOPE_BACKEND=openGL_mock` to bypass the GUI.

## Dynamics Overview

The C++ dynamics mirror the CasADi script in `model_morphy/cleaned_code`. The state vector contains:

- Base position, quaternion, velocity, angular velocity (13 states)
- Four arm quaternions (16 states)
- Relative angular velocities in each arm frame (12 states)

A block-linear system (18 unknowns) is assembled each step from:

1. Translational block for base linear acceleration.
2. Rotational block for base angular acceleration (body frame).
3. Four arm rotational blocks for the hinge dynamics.

Solving this system returns world accelerations (`W_a_B`), angular accelerations (`W_omega_dot`) and the derivatives of the relative joint rates. Quaternions are propagated via the usual `0.5 * q ⊗ ω` relation, and all quaternions are renormalized after every RK4 substep.

## Controller

`DroneSimulationApp` maintains a PID controller in world coordinates:

```
a_cmd = Kp ∘ (p_ref - p) + Ki ∘ ∫(p_ref - p) dt + Kd ∘ (-v) + [0, 0, g]
thrust_total = m_total * a_cmd.z
```

The total thrust is equally split between the four rotors and clamped to the maximum specified in the YAML (`propellers.thrust_max`). You can tweak the gains and target in `main.cpp`.

## YAML Parameters

All physical constants live in `model/drone_parameters.yaml`. Important sections:

- `mass`, `inertia`: base and arm masses/inertias.
- `propellers`: thrust and torque coefficients, spin direction, rotor inertia, max thrust.
- `morphing_joint`: stiffness/damping of the morphing joints.
- `transforms`: rigid transforms from base to hinge (`T_BH`), hinge to prop (`T_HP`), base to prop (`T_BP`).
- `integrator`: one of `explicit_euler`, `rk4`, `implicit_euler`, `irk` (implicit midpoint).
- `integrator_settings`: `dt`, `substeps`, `implicit_max_iterations`, `implicit_tolerance`, `implicit_fd_epsilon`.
- `initial state`: `x0_pos`, `x0_rotation` and `v0`/`x0_vel` for base linear velocity.
- `contact` environment: keep the classic ground plane via `contact.ground_height`, or enable an
  axis-aligned box with `contact.box_enabled`, `contact.box_center` and `contact.box_size`
  (defaults to a 0.3 m cube centered at the origin so the drone bounces on every wall). Set
  `MORPHY_FREE_FALL=1` to let it simply drop and rebound without the PID.

Editing this file does not require recompiling; the values are loaded at runtime.

## Convex Hull Utility (optional)

If you want to precompute convex hulls of the meshes:

1) Install extra deps: `sudo apt install libcgal-dev libgmp-dev libmpfr-dev libboost-all-dev assimp-utils`.
2) Convert STL → OBJ:
   ```bash
   mkdir -p graphics/mesh_obj
   for f in graphics/meshes/*.[sS][tT][lL]; do
     base=$(basename "${f%.*}")
     assimp export "$f" "graphics/mesh_obj/${base}.obj"
   done
   ```
3) Build the hull tool:
   ```bash
   cmake -S . -B build -DENABLE_MESH_HULL=ON
   cmake --build build --target mesh_convex_hull
   ```
4) Run it (defaults to `graphics/mesh_obj` → `graphics/hulls`):
   ```bash
   HULL_IN_DIR=graphics/mesh_obj HULL_OUT_DIR=graphics/hulls ./build/mesh_convex_hull
   ```
The tool writes `*_hull.obj` you can load at runtime.

## Troubleshooting

- **Empty window / missing meshes**: ensure the URDF and meshes under `graphics/` are intact. The program prints a message if any mesh is missing.
- **GLFW “failed to open display”**: set `MORPHY_POLYSCOPE_BACKEND=openGL_mock` for headless runs.
- **Dynamics blow up**: the solver currently assumes the initial state is close to hover. Large impulses (e.g., setting high gains) can make the 18×18 system ill-conditioned; check console logs for diagnostics.

## License

Refer to the repository for licensing terms (if not specified, assume all rights reserved).
