# rmcs_rl

Generic YAML observation/action interfaces and in-process ONNX policy workers for
RMCS. Vehicle calibration, PD and actuator limits live in `rmcs_core`.
Wheel-leg self-righting belongs to the policy; there is no host recovery script.

## Wheel-leg live mirror

`rmcs_rl::WheelLegMirrorComponent` publishes measured joints, body IMU and
estimator references for the Isaac Sim kinematic mirror. Its implementation
is in `src/visualization/wheel_leg_mirror.cpp`; the Foxglove receiver and
JSONL recording/replay utilities are in `tool/visualization/`.

Enable it in the vehicle YAML, alongside the existing wheel-leg hardware,
chassis and state-estimator components:

```yaml
rmcs_executor:
  ros__parameters:
    components:
      # Append to the existing component list.
      - rmcs_rl::WheelLegMirrorComponent -> wheel_leg_mirror

wheel_leg_mirror:
  ros__parameters:
    enabled: true
    update_rate: 100.0
```

Loading the component enables telemetry at 100 Hz by default; the parameter
block can be omitted if those defaults are suitable. `enabled: false` disables
all of its inputs and publishers. It works with traditional and RL wheel-leg
profiles and does not require an `RlBridge` instance or a policy model.

`enabled` controls this telemetry component, not actuator enable. In the
`wheel-leg-infantry-wheelbipe-recovery` profile it stays on while chassis
control and DM drives are disabled. `control_enabled` and `dm_control_ready`
are diagnostic labels only. The viewer uses fresh measured encoders and IMU
even when estimator references are invalid. It applies no gravity, contact,
tilt or angular-speed gate; nominal joint limits produce diagnostics without
clamping measured poses. Stale/invalid raw data and unsolved CAD loop closure
remain explicit display errors. Motor safety and enable logic are unchanged.

Calibration stays in `rmcs_core::controller::chassis::WheelLegStateEstimator`,
which supplies `model_offsets`, `calibration_sha256` and
`calibration_model_spec_sha256` under `/wheel_leg/state/`. The mirror reads
these interfaces rather than including private vehicle headers or copying
calibration tables. It publishes the same `/wheel_leg/telemetry/*` messages
and recording schema as the earlier core broadcaster.

Build with `build-rmcs --packages-select rmcs_rl rmcs_core rmcs_bringup`. Run
Foxglove bridge in the robot's ROS environment; the Isaac GUI remains on the
viewing computer and connects over WebSocket. The parent repository's
`docs/zh-cn/wheel_leg_live_mirror.md` describes the complete configuration and
the training repository's `docs/WHEEL_LEG_LIVE_MIRROR.md` describes GUI startup,
joint-name mapping and measured-pose checks.

## Policy deployment

### Structured observation and action configuration

New profiles use ordered names and parameter blocks rather than `key=value`
strings. Both the deformable suspension profile and wheel-leg profiles use this
format. Nested mappings are ROS parameter namespaces; `order` is a string array.

```yaml
rl_bridge:
  ros__parameters:
    rl_model_path: models/actor.onnx
    policy_rate: 50.0
    observations:
      order: [gyro, command]
      gyro:
        source: /robot/gyro
        type: vector3
        scale: 0.5
        clip: [-10.0, 10.0]
      command:
        source: /robot/command
        type: vector3
        select: [0, 2]
    actions:
      order: [left, right]
      left:
        output: /robot/policy/left
        scale: 0.25
      right:
        output: /robot/policy/right
        scale: 0.25
    expected_metadata:
      training_contract: robot_v1
```

Supported observation types are `scalar` (default), `vector3`, `array` with
`size`, `scalars` with an ordered `sources` string array, `projected_gravity`
from a quaternion, `constant` with a numeric `value` array, and `previous_action`.
`select` selects/reorders source indices and may repeat indices. Processing is
selection, `value * scale + offset`, then optional `clip: [lower, upper]`.
`scale` and `offset` are scalar values applied to the whole term. A scalar source
may specify a finite `default` for an absent interface. Input and action sizes
are inferred; existing size parameters remain optional assertions.

`source_clip` optionally clips before scaling. `precision: float32` rounds the
source and arithmetic to float32, preserving training-time preprocessing order;
the default is float64. Nonfinite sources are rejected before clipping.

`previous_action` means the bridge's mapped/clipped action converted back to
normalized units, not measured or guaranteed actuator execution. A vehicle that
needs applied-action history must explicitly publish and bind that source.
`expected_metadata` checks arbitrary string metadata keys at model startup;
missing keys also fail validation. ONNX graphs own architecture and weights;
tensor rank does not identify a model as an MLP or Transformer. `model_type:auto`
therefore uses generic tensor validation. Vehicle-specific model type names are
no longer accepted; use explicit metadata assertions instead.

Legacy `observation_terms`/`action_terms` remain available for migration, but
cannot be mixed with the structured format. Producer-sequence-triggered
structured observations may contain multiple named features. Their producer
must update all features before advancing the sequence. Legacy complete-frame
mode retains its stricter single-array contract.

Wheel-leg publishes unnormalized command, height, gyro, velocity and action
feedback. YAML owns their preprocessing, feature concatenation, wheelbipe channel
permutation, and physical action scaling/clipping. The controller consumes leg
offsets and wheel speeds, retaining only mechanical winding and actuator safety.
`action_history_source: raw|mapped` selects raw actor feedback or inverse-mapped
executed feedback. Every action publishes `/raw`, `/scale`, `/offset`, and clip
metadata alongside its decoded value. Observation telemetry at
`<rl_base>/input/frame` now comes from the actual shared observation assembler.
Standing odometry still uses a frozen training-reference calculation at 100 Hz;
it explicitly rejects other rates until that stateful reference is generalized.
Mechanical observation conventions and reference-profile selection remain
vehicle-side. This is not yet a completely model-independent vehicle adapter.

Set `rl_bridge.rl_model_path` in the vehicle profile and restart. Relative paths
resolve inside the installed `rmcs_rl` share directory. `observation_terms` owns
feature order, source binding, scaling and clipping; `action_terms` owns actor
channel routing, target scaling and clipping. `history_length`, optional
`history_rate` and size assertions describe the model input contract.

The loader checks tensor types, ranks, dimensions and a trial inference. The
paired-107 wheel-leg profile also validates model-spec and winding metadata.
When ONNX supplies `policy_hz`, the loader rejects a different YAML `policy_rate`.
`expected_observation_contract_sha256` additionally requires matching
`rmcs_observation_contract_sha256` and `policy_hz` ONNX metadata.
Observation normalization metadata is optional numerical preprocessing.

`RlBridge` creates a policy worker thread and an RMCS action publisher partner.
The publisher runs before action consumers; consumers publish calibrated inputs
and the bridge assembles the next observation after them. This supports feedback
without a cycle in the executor dependency graph. The
executor never waits for ONNX inference. Results carry a monotonic sequence,
reset generation, submission timestamp and optional opaque context. Results
from a previous generation or beyond `inference_timeout` / `max_action_age`
cannot become valid actions. The publisher binds the reset source directly, so
reset invalidates pending results before the control component runs.

The bridge publishes scaled actions at the configured `output=` interfaces,
alongside `/scale`, `/clip_min` and `/clip_max` metadata for each target. It
also publishes `/valid`, `/healthy`, `/action_seq`, `/action_context` and
`/action_steady_ns` under `rl_base`. `/healthy` reports worker health; `/valid`
additionally requires an enabled session and a fresh finite action. Consumers
must gate actuation on validity and their own control/safety state. Observation
telemetry is available under `<rl_base>/observation/<term-name>`.

For wheel-leg, the bringup profile `wheel-leg-infantry-rl.yaml` uses
`rmcs_core::controller::chassis::WheelLegRlController` and `rmcs_rl::RlBridge`.
The controller supplies calibrated joint/IMU coordinates, command conditioning
and executed action history. Unified policies use a complete controller-owned
frame; the bridge copies the frame and runs inference.
Leg actions are nominal-position offsets in model radians. Wheel actions are
wheel-side speeds in rad/s. The controller applies paired winding, measured
hinge guards, feedback freshness, feedback PD and hardware torque limits.
PREPARE waits with zero effort for drive readiness and the first valid policy
action. The policy then controls all six axes without a scripted pose, upright
capture or handover blend. The original GPL license remains in `rmcs_core`.

```sh
build-rmcs --packages-select rmcs_rl rmcs_core rmcs_bringup --cmake-args -DBUILD_TESTING=ON
colcon test --merge-install --packages-select rmcs_rl rmcs_core
```

The integration test constructs only in-memory snapshots and production software
components. It checks direct policy entry from fallen poses, reset/rearm and
feedback guards. It does not establish physical self-righting or vehicle timing.

The optional directional wheel-leg preview uses 200 Hz policy / 1000 Hz feedback and
15 ms action expiry. `tool/prepare_wheel_leg_model.py` validates its source preview
package, training contract, mechanical model and YAML layout before stamping
the missing paired-107/layout metadata. It preserves graph bytes and compares
256 batch-one inferences, recording the final SHA256 in `.deployment.json`.
Without `--stamp` it checks an already prepared deployment. The original
preview package remains unchanged and the candidate remains unqualified.

The legacy vehicle YAML preserves `recovery_candidate.onnx` at 50 Hz.
`wheel-leg-infantry-unified-rl.yaml` selects `self_wake_V1.0.onnx` at 200 Hz,
with 1000 Hz host torque PD, 35 single-frame observations and six actions.
It is an experimental actor with `behavior_qualified=false` and no hardware
acceptance. Its model SHA256 is
`48d4880f2b928512be54bea0e4b24f09fc3c47057c3c36dba87066ab34d7e443`.

For producer-owned frames, set `observation_frame_sequence_interface` and use
one unscaled `take=array type=double_array dim=35` observation term with
`history_length=1`. Each new positive sequence submits once; duplicate sequences
do not refresh action age. Sequence rollback blocks until a session reset.
The controller owns the standing-displacement reference and applied-action
history. Slots 14/15 contain filtered standing wheel-coordinate displacement
in metres times 20, computed from continuous wheel angles, legs and IMU gravity.
They are not raw wheel angles. Raw leg actions allow ±16 (offset ±4 rad);
wheel actions allow ±9 (speed ±90 rad/s).

The observation contract is
`db0fc6121c73121dcd39eef0524ecfeab4b3aa69aa7f5700d4205499af2402a9`.
The parent repository's `rmcs_core/tools/verify_wheel_leg_unified.py` checks
the export package, current profile, frozen helpers and inference vectors.
