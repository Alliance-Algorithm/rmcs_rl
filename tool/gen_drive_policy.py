#!/usr/bin/env python3
"""Generate a drive-style validation policy ONNX for rmcs_rl deployment testing.

Contract: input "obs" float32 [1, 22] -> output "actions" float32 [1, 4].
Layout (identical to the deployment obs builder):
  obs[0:3]  = commands (vx, 0, yaw_rate)  -- reserved, unused
  obs[3]    = height_cmd * 5.0
  obs[4:7]  = ang_vel * 0.5
  obs[7:10] = projected gravity
  obs[10:14]= leg pos deviation
  obs[14:18]= leg vel * 0.1
  obs[18:22]= last actions

Behavior (deterministic, bounded, no state):
  a_i = clip( A * tanh(K * (height_cmd - h0)) + bias_i, -clip, clip )
  -> 四腿随高度指令联动（每腿带固定偏置产生轻微不对称），
     用于验证 观测 -> 策略 -> 动作 -> PD -> 力矩 -> 关节运动 的完整链路。
  height_cmd = 0.05 (低) -> a 负饱和；0.17 (高) -> a 正饱和。

Usage:
  python3 gen_drive_policy.py -o policy_drive.onnx
"""
import argparse

import torch
import torch.nn as nn


class DrivePolicy(nn.Module):
    def __init__(self, amp: float = 2.0, gain: float = 8.0, h0: float = 0.132,
                 bias=(0.15, -0.08, 0.08, -0.15), clip: float = 2.0):
        super().__init__()
        self.amp = amp
        self.gain = gain
        self.h0 = h0
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))
        self.clip = clip

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        height_cmd = obs[:, 3:4] / 5.0  # obs[3] = height_cmd * 5
        a = self.amp * torch.tanh(self.gain * (height_cmd - self.h0))
        a = a + self.bias.unsqueeze(0)
        return torch.clamp(a, -self.clip, self.clip)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="policy_drive.onnx")
    args = parser.parse_args()

    model = DrivePolicy().eval()
    dummy = torch.zeros(1, 22, dtype=torch.float32)
    # 抽样验证：低/基准/高 高度指令
    with torch.no_grad():
        for h in (0.05, 0.132, 0.17):
            obs = dummy.clone()
            obs[0, 3] = h * 5.0
            a = model(obs)
            target = 0.25 * a + 0.3  # action_scale=0.25, default_dof_pos=0.3
            print(f"height={h:.3f} -> actions={a[0].tolist()} (pos target ~{target[0].tolist()})")

    torch.onnx.export(
        model, dummy, args.output,
        input_names=["obs"], output_names=["actions"], opset_version=13,
    )
    print(f"wrote {args.output}: obs float32[1,22] -> actions float32[1,4]")


if __name__ == "__main__":
    main()
