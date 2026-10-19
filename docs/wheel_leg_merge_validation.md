# Wheel-leg deployment comparison

The deployment target is RMCS `merge/wheel-leg-rl`, with `rmcs_rl` retained as a Git submodule.
Wheel-leg control follows `origin/dev/wheel_leg_rl`; deformable control and declarative policy I/O
follow `origin/dev/rmcs_rl`. Configuration and model changes require restarting RMCS.

## Reference identity

Compared on 2026-10-19 after fetching the parent repository:

| Reference | Commit |
| --- | --- |
| Wheel-leg | `eab2ec74bd28ba743af1e4bbbaf0d25382e4749a` |
| Deformable / YAML design | `2120d5e5b89230b3adbb1a30646ef8112bb567bc` |
| Deformable reference submodule | `f98953f7b4daa5da14d82b25cb92f7721cda81cb` |

The vehicle entry point is `rmcs_bringup/config/wheel-leg-infantry-recovery.yaml`.
It selects `models/wheel_leg/recovery_candidate.onnx`, whose SHA256 is
`a00d13df1d4113fe949c2a6a15b508e5c491fd1a5e6881570bf025a8d3c29c29`.
The YAML filename is a deployment profile; the ONNX file contains the actor weights.

The older merge selected a different actor (`11b764bee0ea2e6f51c7dad25cff464efaab24e20a90235e96299c51858e765e`).
It also omitted grounded push-off logic, overwrote the extended support reference with the low
actor-capture pose, and applied actor results asynchronously. Those differences have been corrected.

## Preserved behavior and configuration

- Observation order, scales, constants, joint defaults, history and six action channels come from YAML.
- Wheel-leg runs one synchronous 50 Hz actor frame before the recovery/PD update; recovery and host
  PD run at 1 kHz. Paired-107 winding, calibration, hinge guards, sensing guards and torque limits remain.
- Grounded upright recovery pushes off toward an intermediate extension while retaining the extended
  support pose. Actor transfer requires shell clearance and the reference capture conditions.
- The recovery profile selects host torque PD, matching the reference. The optional DM position PD
  backend, gain checks, fresh-command gating and telemetry are also ported from that reference.
- Reset clears torque, position references, command timestamp, history, action feedback and recovery.
- Deformable retains its reference hardware/control sources, actor bytes, 8 x 32 history, four action
  terms, 200 Hz policy rate, 100 Hz history rate and 10 ms maximum action age. It continues to use the
  in-process asynchronous worker introduced by the merge.

## Reproduce the offline checks

Use the ROS Jazzy build environment, with this submodule next to `rmcs_bringup` and the parent
repository's reference branches fetched. From `rmcs_ws`:

```sh
colcon build --merge-install --symlink-install \
  --packages-select rmcs_rl rmcs_core rmcs_bringup --cmake-args -DBUILD_TESTING=ON
colcon test --merge-install --packages-select rmcs_rl
colcon test-result --test-result-base build/rmcs_rl --verbose
python3 src/rmcs_rl/tools/verify_wheel_leg_reference.py --reference origin/dev/wheel_leg_rl
```

The verifier compares the recovery YAML's numeric parameters, the ONNX and recovery-geometry bytes,
the DM/DR16/hardware/broadcaster sources and the deformable contract. It compiles the original
wheel-leg recovery FSM and ONNX wrapper from the selected Git ref, then replays identical inputs
through the reference and merge implementations. Eight deterministic recovery input traces cover
grounded/upright starts, sagittal and side falls, winding, support loss and initial release. It
compares phases, routes, failures, reference positions, efforts and effective action history on
6,400 samples, plus the six outputs for 256 ONNX inputs.

These are input-replay and build checks. They do not simulate vehicle dynamics or establish recovery
success, tracking quality, real-time behavior, CAN timing or physical equivalence. No robot is
started by these commands. Vehicle commissioning still needs the operator to start the recovery
profile and check actual sensing, zero-command standing, motion, reset and stop behavior.

The 2026-10-19 check passed the three-package build and 21 deployment regressions (22 entries in
the colcon summary, including the CTest wrapper). All 6,400 recovery samples and 256 actor samples
matched the reference with a maximum absolute difference of zero.
