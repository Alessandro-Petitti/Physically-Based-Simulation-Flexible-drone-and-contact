# Physically Based Simulation – Flexible Drone & Contact

## Quick build (commands first)
```bash
cmake -S . -B build
cmake --build build
```

- Optional hull utility: `cmake -S . -B build -DENABLE_MESH_HULL=ON && cmake --build build --target mesh_convex_hull`
- If you hit toolchain/config errors, remove `build/` and rerun the two commands above.

## Dependencies
- CMake ≥ 3.16, C++17 compiler, Git
- System packages: `libassimp-dev`, `liburdfdom-dev` (or platform equivalents)
- Ubuntu example:
  ```bash
  sudo apt update
  sudo apt install build-essential cmake git libassimp-dev liburdfdom-dev
  ```

## What this repo does
Real-time C++ simulation of a morphing quadrotor, rendering the URDF meshes with Polyscope while numerically integrating a 41-state model (base + four arms). Physics, controller, integrator choice, and contact shapes come from `model/drone_parameters.yaml` and runtime environment toggles.

## Run modes
- Default simulation (PID + contacts + viewer):
  ```bash
  ./build/morphy_viewer
  ```
- Viewer only (no dynamics, just URDF inspection):
  ```bash
  MORPHY_VIEW_ONLY=1 ./build/morphy_viewer
  ```
- Headless / CI render-safe:
  ```bash
  MORPHY_POLYSCOPE_BACKEND=openGL_mock ./build/morphy_viewer
  ```
- Hide contact point debug overlay:
  ```bash
  MORPHY_CONTACT_VIZ=0 ./build/morphy_viewer
  ```
- Every run writes JSON frames to `animation_data/frames/` (cleared on startup) for offline playback/export.

## Integrators (all configured in `model/drone_parameters.yaml`)
- Options: `integrator: explicit_euler`, `rk4`, `implicit_euler`, or `irk` (implicit midpoint Runge–Kutta).
- Timestep and solver knobs live under `integrator_settings` (`dt`, `substeps`, `implicit_max_iterations`, `implicit_tolerance`, `implicit_fd_epsilon`).
- Example: run with RK4 and 5 substeps after setting the YAML accordingly:
  ```bash
  ./build/morphy_viewer
  ```
- The chosen integrator and settings are printed at startup for verification.

## PID controller (on/off)
- Default: world-frame position PID toward `(0, 0, 1)` m with per-rotor thrust saturation from the YAML (`propellers.thrust_max`).
- Disable PID but keep viewer and contacts:
  ```bash
  MORPHY_DISABLE_PID=1 ./build/morphy_viewer
  ```
- Pure free fall (forces zeroed regardless of PID gains):
  ```bash
  MORPHY_FREE_FALL=1 ./build/morphy_viewer
  ```
- Re-enable after disabling:
  ```bash
  MORPHY_DISABLE_PID=0 ./build/morphy_viewer
  ```

## Contact model (plane or box)
- Uses convex hulls in `graphics/hulls` (millimeters scaled to meters). Friction and CCD are supported.
- Box workspace (default in the YAML):
  ```bash
  ./build/morphy_viewer
  ```
  Configure center/size in `contact.box_center` and `contact.box_size`; set `contact.box_enabled: true`.
- Ground plane only:
  ```bash
  MORPHY_GROUND_Z=0.0 ./build/morphy_viewer   # overrides YAML ground height
  ```
  Or set `contact.box_enabled: false` and `contact.ground_height` in the YAML.
- Tune stiffness/damping/activation distance without editing the file:
  ```bash
  MORPHY_CONTACT_K=200   MORPHY_CONTACT_D=40   MORPHY_CONTACT_D0=0.0015 ./build/morphy_viewer
  ```
- Toggle friction or CCD in the YAML (`contact.enable_friction`, `contact.enable_ccd`). Start height padding can be adjusted with:
  ```bash
  MORPHY_START_CLEARANCE=0.02 ./build/morphy_viewer
  ```

## Parameters at a glance (`model/drone_parameters.yaml`)
- Mass/inertia (`mass`, `inertia`), propeller constants and spin direction (`propellers`), joint stiffness/damping (`morphing_joint`).
- Rigid transforms for hinges and motors (`transforms.T_BH`, `T_HP`, `T_BP`).
- Integrator selection + solver settings (`integrator`, `integrator_settings`).
- Initial pose/velocity (`x0_pos`, `x0_rotation`, `v0`/`x0_vel`).
- Contact setup (`contact.*`), including friction, CCD, box vs plane, and optional env overrides listed above.
- Values are loaded at runtime; rebuilding is not required after edits.

## Command-line tools (headless utilities)
- Free-fall collision predictor (stops at plane crossing):
  ```bash
  ./build/collision_check [x y z [max_time]]
  ```
- Free-fall with penalty contacts and logging:
  ```bash
  ./build/contact_drop [drop_height [max_time]]
  ```
- Contact smoke test at a fixed height (prints penetration/forces):
  ```bash
  ./build/contact_smoketest [base_z]
  ```
- Hull / contact inspectors (print hull stats and sample contacts):
  ```bash
  ./build/debug_hull
  ./build/debug_contacts
  ./build/debug_rotation
  ```
- Optional convex hull generator (requires `ENABLE_MESH_HULL=ON` at configure time):
  ```bash
  HULL_IN_DIR=graphics/mesh_obj HULL_OUT_DIR=graphics/hulls ./build/mesh_convex_hull
  ```

## Notes and tips
- Polyscope ground plane is off by default; enable it from the UI if needed.
- `MORPHY_HULL_SCALE` can rescale hulls for the contact utilities; default is `0.001` (mm → m).
- If dynamics blow up, check console logs for the 18×18 solve and verify your gains, timestep, and contact stiffness.*** End Patch
