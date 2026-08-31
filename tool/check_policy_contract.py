#!/usr/bin/env python3
"""Check an ONNX policy against the rmcs_rl contract.

Verifies tensor names ("obs"/"actions"), dtype float32, shapes [1, obs]/[1, act],
and runs one dummy inference to confirm finite outputs.

Usage:
  python3 check_policy_contract.py policy.onnx --obs 22 --act 4
"""
import argparse
import sys

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to policy.onnx")
    parser.add_argument("--obs", type=int, required=True)
    parser.add_argument("--act", type=int, required=True)
    args = parser.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"input : {inp.name} {inp.shape} {inp.type}")
    print(f"output: {out.name} {out.shape} {out.type}")

    ok = True
    if inp.name != "obs" or out.name != "actions":
        print("FAIL: tensor names must be 'obs' / 'actions'")
        ok = False
    if list(inp.shape) != [1, args.obs]:
        print(f"FAIL: input shape {inp.shape} != [1,{args.obs}]")
        ok = False
    if list(out.shape) != [1, args.act]:
        print(f"FAIL: output shape {out.shape} != [1,{args.act}]")
        ok = False
    if inp.type != "tensor(float)" or out.type != "tensor(float)":
        print(f"FAIL: dtype must be float32 (got {inp.type}/{out.type})")
        ok = False

    if ok:
        x = np.zeros((1, args.obs), dtype=np.float32)
        y = sess.run(["actions"], {"obs": x})[0]
        finite = bool(np.isfinite(y).all())
        print(
            f"dummy inference OK, actions finite={finite}, "
            f"range=[{y.min():.4f},{y.max():.4f}]"
        )
        ok = finite

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
