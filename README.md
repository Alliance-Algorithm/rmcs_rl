# rmcs_rl

RMCS RL deployment package, referenced by RMCS as `rmcs_ws/src/rmcs_rl` submodule.

## Components

| Deployment | Components | Responsibility |
| --- | --- | --- |
| deformable | `rmcs_rl::RlBridge` | YAML observation/history assembly, in-process worker inference, action freshness and reset handling |
| wheel-leg | `rmcs::rl::RlController` | Shared YAML observation/history and action mapping, in-process worker inference, preparation, feedback PD and recovery |

Vehicle hardware, remote commands, deformable action consumption and suspension arbitration
remain in `rmcs_core`. Vehicle configurations belong to `rmcs_bringup/config`.

## Replace a model

Change `rl_bridge.rl_model_path` for deformable or `rl_controller.rl_model_path` for wheel-leg
in the vehicle YAML, then restart. Relative paths resolve against the installed `rmcs_rl`
share directory; absolute paths are also supported. Generic models do not require stamping or
model IDs. The wheel-leg paired-107 profile retains its model-spec and winding checks because its
mechanical recovery geometry depends on that contract.

The loader reads ONNX tensor names, types and shapes. Single-input/single-output tensor names
are automatic. Multiple inputs require an explicit `input_name` and configured `extra_inputs`.
Both deployments derive observation and action counts from `observation_terms` and `action_terms`. Set
`history_length` and, for rank-3 models, `rl_feature_size` to match the ONNX input shape.

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
contains shared policy interface parsing, observation binding/assembly, action scaling, history and
in-process snapshots. Wheel-leg uses these same implementations inside its control component, so
its recovery feedback does not introduce a component dependency cycle. `src/wheel_leg` contains
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

## Wheel-leg YAML interface

The bringup profiles `wheel-leg-infantry-rl.yaml` and `wheel-leg-infantry-recovery.yaml`
keep the real-vehicle hardware, calibration, paired leg winding and native recovery settings.
Their `rl_controller` block now declares the same policy terms as `RlBridge`:

- `rl_model_path`, `policy_rate`, `inference_timeout`, `history_length`, optional `history_rate`.
- `joint_names`, `joint_base_path`, `default_joint_pos`, `observation_terms`, `action_terms`.
- Optional `rl_obs_size` / `rl_action_size` assertions and normal model input/output options.

The controller publishes unscaled model inputs at `/wheel_leg/rl/input/{command,height,angular_velocity,projected_gravity}`
and calibrated joint angle/velocity at `/wheel_leg/rl/model/<motor>/{angle,velocity}`.
These local values are bound directly to the shared observation assembler. New upstream component
interfaces can also be used in terms. Control outputs cannot feed their own observation inputs.
The current YAML declares 35 features, one history frame and six actions; their order and numeric
scaling are no longer a C++ observation layout.

`action_terms` must map all six `/wheel_leg/rl/target/<motor>` channels exactly once. The list is
ordered by ONNX action index; changing `output=` changes which motor receives that channel.
As with `RlBridge`, scaling precedes clipping. Leg targets are offsets in model radians from
`default_joint_pos`; wheel targets are wheel-side rad/s. For example:

```yaml
observation_terms:
  - index=0 path=/wheel_leg/rl/input/command take=x name=command/forward
  # Continue with the complete model feature contract.
action_terms:
  - index=0 output=/wheel_leg/rl/target/left_hip_joint scale=0.25 clip=-0.75:0.75
  # Continue with all six physical actuators.
```

Leg winding, hinge guards, model-to-motor effort mapping and hardware torque limits run after
YAML action decoding. Recovery converts its executed references back through the configured
per-motor scales and actor indices for `last_action`. Reset clears history, action feedback and
effort, and rejects asynchronous results from the previous generation. PREPARE keeps its bounded
PD/script until the first valid inference; accepted actions expire at `inference_timeout`.
All configuration is startup-only. Changing a network's observation or actuator conventions still
requires its matching YAML and the existing mechanical recovery contract.

With RMCS bringup alongside this submodule, run the offline deployment regressions with:

```sh
colcon build --merge-install --symlink-install --packages-select rmcs_rl --cmake-args -DBUILD_TESTING=ON
colcon test --merge-install --packages-select rmcs_rl
colcon test-result --test-result-base build/rmcs_rl --verbose
```

These tests load the real actor and vehicle YAML without constructing hardware or enabling drives.
