# rmcs_rl

RMCS RL deployment package, referenced by RMCS as `rmcs_ws/src/rmcs_rl` submodule.

## Components

| Deployment | Components | Responsibility |
| --- | --- | --- |
| deformable | `rmcs_rl::RlBridge` | YAML observation/history assembly, in-process worker inference, action freshness and reset handling |
| wheel-leg | `rmcs::rl::RlController` | Fixed wheel-leg observations, direct inference, preparation, feedback PD and recovery |

Vehicle hardware, remote commands, deformable action consumption and suspension arbitration
remain in `rmcs_core`. Vehicle configurations belong to `rmcs_bringup/config`.

## Replace a model

Change `rl_bridge.rl_model_path` for deformable or `rl_controller.model_path` for wheel-leg
in the vehicle YAML, then restart. Relative paths resolve against the installed `rmcs_rl`
share directory; absolute paths are also supported. No stamping, model ID or file hash is required.

The loader reads ONNX tensor names, types and shapes. Single-input/single-output tensor names
are automatic. Multiple inputs require an explicit `input_name` and configured `extra_inputs`.
The deformable bridge derives observation and action counts from its YAML terms. For rank-3 models,
set `rl_history_length` and `rl_feature_size` to match the ONNX input shape.

Startup checks tensor types, batch/rank, observation/history dimensions, action count and a trial
inference. Model initialization failures include the resolved path and incompatibility reason.
The worker runs inside RMCS. The 1 kHz executor never waits for inference; it accepts only completed,
fresh results from the current reset generation.

Compatible internal network changes need only a new path. A changed observation/action contract
also needs a matching vehicle configuration. Tensor shape cannot establish feature order, axes,
units or action meaning. Observation normalization metadata is optional numerical preprocessing,
not a model identity gate; existing normalization is preserved.

## Layout and reset

`src/policy` contains shared model validation and the bounded inference worker. `src/rl_bridge`
contains bridge configuration, observations/history and in-process snapshots. `src/wheel_leg` contains
the migrated real-vehicle deployment and its original GPL-3.0-only license; other original sources
retain their MIT license.

Internal controller data uses RMCS component interfaces. Bridge sequence IDs remain monotonic
across reset so delayed actions cannot match a new observation. Reset and control-authority changes
clear history and action feedback. Wheel-leg reset clears outputs, recovery and policy state.

## Build

```sh
colcon build --merge-install --symlink-install --packages-up-to rmcs_rl rmcs_core rmcs_bringup
```

The standalone policy server, launcher, ROS inference topics, model stamping utilities, generated
test models and commissioning pose-viewer regressions are removed. The in-process worker performs
compatibility checks at startup.
