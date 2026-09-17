#!/usr/bin/env python3
"""Generate a synthetic policy ONNX matching the rmcs_rl contract.

Contract: input "obs" float32 [1, N_obs] -> output "actions" float32 [1, N_act].
The synthetic policy outputs constant zeros: in RL state, position PD holds
default_dof_pos — safe for bench pipeline validation (no learned behavior).

Sizes come from --obs/--act, or from the deployment YAML (--from-config) so the
model shape always matches the bridge contract; metadata is *not* stamped here
(run stamp_layout_metadata.py afterwards, then check_policy_contract.py).

Negative test: generate with a wrong obs size (e.g. --obs 42) and verify the
consumer refuses to load (obs size mismatch).

Usage:
  python3 gen_synthetic_policy.py --obs 22 --act 4 -o policy.onnx
  python3 gen_synthetic_policy.py --from-config deploy.yaml --node rl_bridge -o policy.onnx
"""
import argparse
import sys

import rl_layout as layout


def _export_with_torch(obs: int, act: int, output: str) -> bool:
    """torch 可用则用 torch.onnx.export（保留历史导出路径）；不可用返回 False。"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return False

    class _ZeroPolicy(nn.Module):
        def __init__(self, obs_size: int, act_size: int) -> None:
            super().__init__()
            self.linear = nn.Linear(obs_size, act_size, bias=False)
            with torch.no_grad():
                self.linear.weight.zero_()

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.linear(obs)

    model = _ZeroPolicy(obs, act).eval()
    dummy = torch.zeros(1, obs, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=13,
    )
    return True


def _export_with_onnx(obs: int, act: int, output: str) -> None:
    """无 torch 时的等价实现：actions = obs @ W，W 全零 → 恒零动作。"""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.zeros((obs, act), dtype=np.float32), name="W")
    node = helper.make_node("MatMul", ["obs", "W"], ["actions"], name="zero_action")
    graph = helper.make_graph(
        [node],
        "zero_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, act])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    model.producer_name = "rmcs_rl/gen_synthetic_policy"
    onnx.checker.check_model(model)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成零动作合成策略 ONNX（obs float32[1,N] -> actions float32[1,M]）"
    )
    parser.add_argument("--obs", type=int, default=None, help="observation size（缺省 22）")
    parser.add_argument("--act", type=int, default=None, help="action size（缺省 4）")
    parser.add_argument("--from-config", "--config", dest="config", default=None,
                        help="部署 YAML：由 observation_terms/action_terms 推导 obs/act 尺寸")
    parser.add_argument("--node", default=layout.DEFAULT_NODE,
                        help=f"YAML 节点名（缺省 {layout.DEFAULT_NODE}）")
    parser.add_argument("-o", "--output", default="policy.onnx")
    args = parser.parse_args()

    obs, act = args.obs, args.act
    if args.config:
        try:
            _, _, obs_size, act_size = layout.load_config(args.config, args.node)
        except layout.LayoutError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        for name, given, derived in (("--obs", args.obs, obs_size), ("--act", args.act, act_size)):
            if given is not None and given != derived:
                print(
                    f"ERROR: {name}={given} 与 {args.config} 推导出的 {derived} 冲突"
                    f"（要么去掉 {name}，要么改 YAML）",
                    file=sys.stderr,
                )
                sys.exit(1)
        obs, act = obs_size, act_size
    obs = 22 if obs is None else obs
    act = 4 if act is None else act

    if obs <= 0 or act <= 0:
        print(f"ERROR: obs/act 必须为正整数（obs={obs} act={act}）", file=sys.stderr)
        sys.exit(1)

    backend = "torch" if _export_with_torch(obs, act, args.output) else "onnx"
    if backend == "onnx":
        _export_with_onnx(obs, act, args.output)

    print(
        f"wrote {args.output}: obs float32[1,{obs}] -> actions float32[1,{act}] "
        "(zero policy)"
    )
    print(f"backend: {backend}")


if __name__ == "__main__":
    main()
