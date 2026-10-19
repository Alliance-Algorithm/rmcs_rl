#!/usr/bin/env python3
"""Compare deployment files and replay outputs with an explicit RMCS Git ref.

Run inside the ROS build environment after building with BUILD_TESTING=ON.
No hardware component is instantiated, and no ROS control command is sent.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
PREFIX = "rmcs_ws/src/rmcs_rl/"
CONFIG = "rmcs_ws/src/rmcs_bringup/config/"


def run(command, **kwargs):
    return subprocess.check_output(command, **kwargs)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_files(root, reference):
    def source(path):
        return run(["git", "show", f"{reference}:{path}"], cwd=root)

    original = yaml.safe_load(source(CONFIG + "wheel-leg-infantry-recovery.yaml"))
    current = yaml.safe_load((root / CONFIG / "wheel-leg-infantry-recovery.yaml").read_text())
    for name in ("wheel_leg", "wheel_leg_chassis_controller"):
        require(original[name] == current[name], f"{name} parameters differ from {reference}")
    expected = original["rl_controller"]["ros__parameters"]
    actual = current["rl_controller"]["ros__parameters"]
    renamed = {
        "model_path": "rl_model_path",
        "rl_inference_frequency": "policy_rate",
        "rl_history_length": "history_length",
        "nominal_model_pos": "default_joint_pos",
    }
    for name, value in expected.items():
        if name == "spin_yaw_rate":  # The chassis component owns this parameter.
            continue
        key = renamed.get(name, name)
        candidate = actual.get(key, "v6" if key == "control_profile" else None)
        if name == "nominal_model_pos":
            candidate = [candidate[motor] for motor in actual["joint_names"]]
        require(candidate == value,
                f"Controller parameter differs: {name}")
    for path in (expected.get("model_path", expected.get("rl_model_path")),
                 expected["recovery_profile_path"], expected["height_lookup_path"]):
        deployed = (PACKAGE / path).read_bytes()
        require(deployed == source(PREFIX + path), f"Deployment asset differs: {path}")
        print(f"Asset match: {path} SHA256={hashlib.sha256(deployed).hexdigest()}")
    for path in ("hardware/wheel-leg.cpp", "hardware/device/dm_motor.hpp",
                 "hardware/device/dr16.hpp", "broadcaster/value_broadcaster.cpp"):
        full_path = "rmcs_ws/src/rmcs_core/src/" + path
        require((root / full_path).read_bytes() == source(full_path),
                f"Hardware/telemetry source differs: {path}")
    deformable = yaml.safe_load((root / CONFIG / "deformable-infantry-omni-rl.yaml").read_text())
    deformable_ref = yaml.safe_load(run([
        "git", "show", "origin/dev/rmcs_rl:" + CONFIG + "deformable-infantry-omni-rl.yaml"
    ], cwd=root))
    bridge = deformable["rl_bridge"]["ros__parameters"]
    bridge_ref = deformable_ref["rl_bridge"]["ros__parameters"]
    for key in ("observation_terms", "action_terms", "history_length", "history_rate",
                "policy_rate", "max_action_age", "joint_names", "default_joint_pos"):
        require(bridge.get(key) == bridge_ref.get(key), f"Deformable contract differs: {key}")
    model = deformable_ref["policy_server"]["ros__parameters"]["rl_model_path"]
    require(bridge["rl_model_path"] == model, "Deformable model selection differs")
    submodule_ref = run(["git", "rev-parse", "origin/dev/rmcs_rl:" + PREFIX[:-1]],
                        cwd=root, text=True).strip()
    require((PACKAGE / model).read_bytes() == run(["git", "show", f"{submodule_ref}:{model}"],
                                                cwd=PACKAGE), "Deformable model bytes differ")
    for path in run(["git", "ls-tree", "-r", "--name-only", "origin/dev/rmcs_rl",
                     "rmcs_ws/src/rmcs_core/src"], cwd=root, text=True).splitlines():
        if "deformable" in Path(path).name and Path(path).suffix in (".cpp", ".hpp"):
            require((root / path).read_bytes() == run([
                "git", "show", "origin/dev/rmcs_rl:" + path
            ], cwd=root), f"Deformable control source differs: {path}")
    print("YAML hardware/recovery parameters and deformable policy contract match")
    settings = dict(actual)
    settings["default_joint_pos"] = [actual["default_joint_pos"][name]
                                     for name in actual["joint_names"]]
    return settings, source


def compare_replay(reference_output, current_output):
    expected = reference_output.decode().splitlines()
    actual = current_output.decode().splitlines()
    require(len(expected) == len(actual), "Replay output row count differs")
    maximum = {"recovery": 0.0, "model": 0.0}
    counts = {"recovery": 0, "model": 0}
    for row, (left, right) in enumerate(zip(expected, actual), 1):
        left, right = left.split(), right.split()
        require(len(left) == len(right) and left[0] == right[0], f"Replay row {row} differs")
        kind = left[0]
        require(kind in counts, f"Unknown replay output: {kind}")
        counts[kind] += 1
        for column, (x, y) in enumerate(zip(left[1:], right[1:]), 1):
            x, y = float(x), float(y)
            require(math.isfinite(x) and math.isfinite(y), f"Nonfinite replay row {row}")
            difference = abs(x - y)
            maximum[kind] = max(maximum[kind], difference)
            # IDs, phases and boolean flags must match exactly.
            tolerance = 0.0 if kind == "recovery" and column <= 8 else 1e-6
            require(difference <= tolerance,
                    f"{kind} row {row}, column {column}: reference={x}, merge={y}")
    require(counts == {"recovery": 6400, "model": 256}, f"Incomplete replay: {counts}")
    for kind in counts:
        print(f"{kind}: {counts[kind]} snapshots, max absolute difference {maximum[kind]:.9g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmcs-root", type=Path, default=PACKAGE.parents[2])
    parser.add_argument("--reference", default="origin/dev/wheel_leg_rl")
    parser.add_argument("--build-dir", type=Path,
                        default=PACKAGE.parents[1] / "build/rmcs_rl")
    args = parser.parse_args()
    root, build = args.rmcs_root.resolve(), args.build_dir.resolve()
    reference = run(["git", "rev-parse", args.reference], cwd=root, text=True).strip()
    print(f"Reference: {args.reference} @ {reference}")
    parameters, source = verify_files(root, reference)
    current_binary = build / "wheel_leg_reference_replay"
    require(current_binary.exists(), "Build rmcs_rl with -DBUILD_TESTING=ON first")
    ort_candidates = list(build.glob("onnxruntime-*/include/onnxruntime_cxx_api.h"))
    require(len(ort_candidates) == 1, "Cannot identify the build's ONNX Runtime")
    ort = ort_candidates[0].parents[1]
    with tempfile.TemporaryDirectory(prefix="rmcs-wheel-reference-") as directory:
        temporary = Path(directory)
        files = run(["git", "ls-tree", "-r", "--name-only", reference, PREFIX + "src"],
                    cwd=root, text=True).splitlines()
        for path in files:
            (temporary / Path(path).name).write_bytes(source(path))
        settings = temporary / "parameters.json"
        settings.write_text(json.dumps(parameters))
        binary = temporary / "reference_replay"
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.check_call(compiler + [
            "-std=c++23", "-O2", "-DRMCS_WHEEL_LEG_REFERENCE",
            "-I" + str(temporary), "-I" + str(ort / "include"),
            str(PACKAGE / "test/wheel_leg_reference_replay.cpp"),
            str(temporary / "joint_reference_recovery_controller.cpp"),
            str(temporary / "recovery_profile.cpp"), str(temporary / "onnx_policy.cpp"),
            "-L" + str(ort / "lib"), "-lonnxruntime", "-lcrypto",
            "-Wl,-rpath," + str(ort / "lib"), "-o", str(binary),
        ])
        inputs = [str(PACKAGE / parameters["recovery_profile_path"]),
                  str(PACKAGE / parameters["height_lookup_path"]),
                  str(PACKAGE / parameters["rl_model_path"]), str(settings)]
        compare_replay(run([str(binary)] + inputs), run([str(current_binary)] + inputs))
    print("PASS: offline deployment parity; vehicle behavior has not been tested")


if __name__ == "__main__":
    main()
