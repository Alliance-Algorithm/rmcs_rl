#!/usr/bin/env python3
"""Generate a synthetic policy ONNX matching the rmcs_rl contract.

Contract: input "obs" float32 [1, N_obs] -> output "actions" float32 [1, N_act].
The synthetic policy outputs constant zeros: in RL state, position PD holds
default_dof_pos — safe for bench pipeline validation (no learned behavior).

Negative test: generate with a wrong obs size (e.g. --obs 42) and verify
RlController refuses to load (rl_obs_size mismatch).

Usage:
  python3 gen_synthetic_policy.py --obs 22 --act 4 -o policy.onnx
"""
import argparse

import torch
import torch.nn as nn


class ZeroPolicy(nn.Module):
    def __init__(self, obs_size: int, act_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(obs_size, act_size, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", type=int, default=22, help="observation size")
    parser.add_argument("--act", type=int, default=4, help="action size")
    parser.add_argument("-o", "--output", default="policy.onnx")
    args = parser.parse_args()

    model = ZeroPolicy(args.obs, args.act).eval()
    dummy = torch.zeros(1, args.obs, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=13,
    )
    print(
        f"wrote {args.output}: obs float32[1,{args.obs}] -> actions float32[1,{args.act}] "
        "(zero policy)"
    )


if __name__ == "__main__":
    main()
