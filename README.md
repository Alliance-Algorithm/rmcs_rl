# rmcs_rl

Generic YAML observation/action interfaces and in-process ONNX policy workers for
RMCS. Vehicle recovery, calibration, PD and actuator control live in `rmcs_core`.

Set `rl_bridge.rl_model_path` in the vehicle profile and restart. Relative paths
resolve inside the installed `rmcs_rl` share directory. `observation_terms` owns
feature order, source binding, scaling and clipping; `action_terms` owns actor
channel routing, target scaling and clipping. `history_length`, optional
`history_rate` and size assertions describe the model input contract.

The loader checks tensor types, ranks, dimensions and a trial inference. The
paired-107 wheel-leg profile also validates model-spec and winding metadata.
When ONNX supplies `policy_hz`, the loader rejects a different YAML `policy_rate`.
Observation normalization metadata is optional numerical preprocessing.

`RlBridge` creates a policy worker thread and an RMCS action publisher partner.
The publisher runs before action consumers; consumers publish calibrated inputs
and the bridge assembles the next observation after them. This supports feedback
through recovery state without a cycle in the executor dependency graph. The
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
`rmcs_core::controller::chassis::WheelLegController` and `rmcs_rl::RlBridge`.
The controller supplies calibrated joint/IMU coordinates, command conditioning
and executed action history; only the bridge serializes and infers the policy.
Leg actions are nominal-position offsets in model radians. Wheel actions are
wheel-side speeds in rad/s. The controller applies paired winding, measured
hinge guards, recovery blending, feedback PD and hardware torque limits.
Recovery JSON assets and the original GPL license are in `rmcs_core`.

```sh
colcon build --merge-install --symlink-install --packages-select rmcs_rl rmcs_core rmcs_bringup --cmake-args -DBUILD_TESTING=ON
colcon test --merge-install --packages-select rmcs_rl rmcs_core
python3 src/rmcs_core/tools/verify_wheel_leg_reference.py --reference origin/dev/wheel_leg_rl --policy-receipt src/rmcs_rl/models/wheel_leg/flat_directional200_1346_preview_20261020.deployment.json
```

The integration test constructs only in-memory snapshots and production software
components. Replay checks recovery and model parity against an explicit fetched
RMCS branch. Neither is vehicle or timing acceptance.

The directional wheel-leg preview uses 200 Hz policy / 1000 Hz feedback and
15 ms action expiry. `tool/prepare_wheel_leg_model.py` validates its source preview
package, training contract, mechanical model and YAML layout before stamping
the missing paired-107/layout metadata. It preserves graph bytes and compares
256 batch-one inferences, recording the final SHA256 in `.deployment.json`.
Without `--stamp` it checks an already prepared deployment. The original
preview package remains unchanged and the candidate remains unqualified.
