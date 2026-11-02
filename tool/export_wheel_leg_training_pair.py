#!/usr/bin/env python3
"""Export two saved training snapshots for the versioned RMCS profiles, offline.

These are diagnostic candidates, including when a learner is still running.
This tool never changes a training receipt or claims behavior qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = ROOT / "rmcs_ws/src/rmcs_bringup/config/contracts/wheel_leg_training_pair_20261103.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_mlp(state, prefix):
    weights = sorted((int(key.split(".")[1]), key, value) for key, value in state.items()
                     if key.startswith(prefix + ".") and key.endswith(".weight"))
    require([(value.shape[1], value.shape[0]) for _, _, value in weights]
            == [(35, 256), (256, 128), (128, 64), (64, 6)], "Unexpected actor topology")
    layers = []
    for index, (_, key, weight) in enumerate(weights):
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(state[key.removesuffix("weight") + "bias"])
        layers.append(layer)
        if index < len(weights) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


class FrozenProneActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.normalizer = actor.obs_normalizer
        self.mlp = actor.mlp

    def forward(self, observation):
        return self.mlp(self.normalizer(observation))


def prone_actor(training_root, checkpoint, contract):
    run = checkpoint.parent
    require(digest(run / "task_contract.json") == contract["task_contract_sha256"],
            "Checkpoint run differs from the pinned prone task")
    source = json.loads((run / "source_revision.json").read_text())
    require(source["commit"] == contract["source"]["commit"], "Prone source identity differs")
    # Compare to the actual owning model, including fixed-scale mask and eps.
    sys.path.insert(0, str(training_root / "src"))
    from tensordict import TensorDict
    from wheeled_algo.prone_observation import ProneMotionModel
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(data["infos"]["schema"] == "prone-motion-checkpoint-v1", "Wrong prone checkpoint")
    actor = ProneMotionModel(TensorDict({"policy": torch.zeros(1, 35)}, batch_size=[1]),
        {"actor": ["policy"]}, "actor", 6, hidden_dims=[256, 128, 64], activation="elu",
        obs_normalization=True, distribution_cfg=dict(class_name="GaussianDistribution",
            init_std=.01, std_type="log", std_range=[.003, .3]))
    actor.load_state_dict(data["actor_state_dict"], strict=True)
    actor.eval()
    mask = actor.obs_normalizer.fixed_scale_mask.nonzero().flatten().tolist()
    require(mask == [14, 15, 31], "Missing gated physical-scale normalization")
    require(actor.obs_normalizer.eps == .01, "Training normalizer epsilon differs")
    module = FrozenProneActor(actor).eval()
    return module, int(data["infos"]["completed_updates"]), lambda x: actor(
        TensorDict({"policy": x}, batch_size=[len(x)]))


def export(module, native, checkpoint, target, profile, *, updates, contract, source_uri):
    require(all(torch.isfinite(value).all() for value in module.state_dict().values()),
            "Actor contains nonfinite state")
    sample = np.random.default_rng(107).uniform(-5., 5., (1024, 35)).astype(np.float32)
    sample[:4, [14, 15, 31]] = np.array([[-1.], [0.], [1.], [5.]])
    with torch.inference_mode():
        expected = native(torch.from_numpy(sample)).numpy()
        wrapped = module(torch.from_numpy(sample)).numpy()
    require(np.allclose(expected, wrapped, atol=1e-6, rtol=1e-6), "Training actor wrapper differs")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pending.onnx")
    torch.onnx.export(module, torch.zeros(1, 35), str(temporary), input_names=["obs"],
        output_names=["actions"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17, dynamo=False)
    prone = profile == "prone"
    metadata = dict(rmcs_model_spec_sha256=contract["model_spec_sha256"],
        leg_target_model_spec_sha256=contract["model_spec_sha256"],
        leg_winding_version="paired_branch107_v1" if prone else "paired_full_vehicle110_v1",
        rmcs_observation_contract_sha256=contract["observation_contract_sha256"],
        rmcs_observation_version="wheel-leg-prone-motion-v4-odom-v2" if prone else "wheelbipe-v202-foundation-v2",
        rmcs_obs_normalization="embedded" if prone else "none", policy_hz=str(contract["policy_hz"]),
        feedback_pd_hz="200", gas_spring_compensation_enabled="true" if prone else "false",
        rmcs_checkpoint_sha256=digest(checkpoint), completed_updates=str(updates),
        behavior_qualified="false", hardware_tested="false")
    model = onnx.load(str(temporary))
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, str(temporary))
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"obs": sample})[0]
    error = float(np.max(np.abs(actual - expected)))
    require(np.allclose(actual, expected, atol=2e-5, rtol=2e-5), f"ONNX parity failed: {error}")
    temporary.replace(target)
    receipt = dict(schema="rmcs-wheel-leg-snapshot-export-v1", profile=profile,
        checkpoint=source_uri, checkpoint_sha256=digest(checkpoint), completed_updates=updates,
        training_budget=10000 if prone else 20000, training_complete=False,
        onnx_sha256=digest(target), pair_contract_sha256=digest(CONTRACT), metadata=metadata,
        parity_samples=len(sample), max_absolute_error=error,
        behavior_qualified=False, hardware_tested=False, live_training_modified=False)
    target.with_suffix(".deployment.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--prone-checkpoint", type=Path, required=True)
    parser.add_argument("--wheelbipe-checkpoint", type=Path, required=True)
    parser.add_argument("--wheelbipe-checkpoint-sha256", required=True,
                        help="SHA256 read independently on Kaiser before copying")
    parser.add_argument("--wheelbipe-source-path", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "rmcs_ws/src/rmcs_rl/models/wheel_leg")
    args = parser.parse_args()
    torch.set_num_threads(1)
    pair = json.loads(CONTRACT.read_text())
    require(digest(args.wheelbipe_checkpoint) == args.wheelbipe_checkpoint_sha256,
            "Kaiser checkpoint copy differs")
    actor, updates, native = prone_actor(args.training_root, args.prone_checkpoint, pair["prone"])
    results = [export(actor, native, args.prone_checkpoint, args.output / "prone_motion_v4_100hz.onnx",
        "prone", updates=updates, contract=pair["prone"], source_uri=str(args.prone_checkpoint.resolve()))]
    data = torch.load(args.wheelbipe_checkpoint, map_location="cpu", weights_only=False)
    state = data["model_state_dict"]
    require(not any("normalizer" in key for key in state), "Unexpected Wheelbipe normalizer")
    actor = make_mlp(state, "actor")
    results.append(export(actor, actor, args.wheelbipe_checkpoint, args.output / "wheelbipe_v202_50hz.onnx",
        "wheelbipe", updates=int(data["iter"]) + 1, contract=pair["wheelbipe"],
        source_uri=args.wheelbipe_source_path))
    print(json.dumps([{key: row[key] for key in ("profile", "completed_updates", "max_absolute_error",
                                                "onnx_sha256")} for row in results], indent=2))


if __name__ == "__main__":
    main()
