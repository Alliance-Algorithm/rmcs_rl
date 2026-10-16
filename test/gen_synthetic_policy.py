#!/usr/bin/env python3
"""Generate a zero-action model for offline contract regression fixtures."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))

import rl_layout as layout


def _export_with_onnx(obs: int, act: int, output: str, model_type: str,
                      sequence_length: int, feature_size: int,
                      extra_specs: list) -> None:
    """Export a zero-action MLP/sequence fixture without a training dependency.

    extra_specs: [(name, dims)] 额外输入；每个 ReduceSum 成标量后加到主输出，
    保证额外输入被图消费（onnx.checker 要求）且输出仍为 [1, act]。
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if sequence_length > 1:
        input_shape = [1, sequence_length, feature_size]
        if model_type == "transformer":
            node = helper.make_node(
                "ReduceMean", ["obs"], ["pooled"], name="sequence_pool", axes=[1], keepdims=0)
            weight_shape = (feature_size, act)
            matmul_input = "pooled"
        else:
            node = helper.make_node("Flatten", ["obs"], ["flat"], name="flatten_history", axis=1)
            weight_shape = (obs, act)
            matmul_input = "flat"
    else:
        input_shape = [1, obs]
        node = None
        weight_shape = (obs, act)
        matmul_input = "obs"
    weight = numpy_helper.from_array(np.zeros(weight_shape, dtype=np.float32), name="W")
    matmul_output = "actions" if not extra_specs else "logits"
    matmul = helper.make_node("MatMul", [matmul_input, "W"], [matmul_output], name="zero_action")
    nodes = [node, matmul] if node is not None else [matmul]

    input_infos = [helper.make_tensor_value_info("obs", TensorProto.FLOAT, input_shape)]
    current = matmul_output
    for index, (name, dims) in enumerate(extra_specs):
        input_infos.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, *dims]))
        nodes.append(helper.make_node(
            "ReduceSum", [name], [f"{name}_sum"], name=f"sum_{name}", keepdims=0))
        final = "actions" if index == len(extra_specs) - 1 else f"add_{index}"
        nodes.append(helper.make_node(
            "Add", [current, f"{name}_sum"], [final], name=f"add_{name}"))
        current = final

    graph = helper.make_graph(
        nodes,
        "zero_policy",
        input_infos,
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
        description="生成零动作合成策略 ONNX（支持 MLP/Transformer 序列输入）"
    )
    parser.add_argument("--obs", type=int, default=None, help="observation size（缺省 22）")
    parser.add_argument("--act", type=int, default=None, help="action size（缺省 4）")
    parser.add_argument("--model-type", choices=("mlp", "transformer", "generic"), default="mlp",
                        help="fixture 结构；MLP 可生成 rank-2 或 rank-3 输入")
    parser.add_argument("--sequence-length", type=int, default=None,
                        help="序列长度；>1 时生成 rank-3 输入")
    parser.add_argument("--feature-size", type=int, default=None,
                        help="单帧维度；缺省由 obs/sequence 推导")
    parser.add_argument("--extra-input", action="append", default=[], metavar="NAME:DIMS",
                        help="额外输入张量，如 mask:4 或 pos:2x3（可重复）")
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

    sequence_length = args.sequence_length or 1
    if sequence_length < 1:
        print("ERROR: sequence-length 必须 >= 1", file=sys.stderr)
        sys.exit(1)
    feature_size = args.feature_size or (obs // sequence_length if obs % sequence_length == 0 else 0)
    if feature_size <= 0 or sequence_length * feature_size != obs:
        print("ERROR: obs 必须能按 sequence-length 分解为正整数 feature-size", file=sys.stderr)
        sys.exit(1)

    extra_specs = []
    for spec in args.extra_input:
        if ":" not in spec:
            print(f"ERROR: --extra-input 需要 NAME:DIMS 形式（got {spec!r}）", file=sys.stderr)
            sys.exit(1)
        name, dims_text = spec.split(":", 1)
        try:
            dims = [int(part) for part in dims_text.split("x")]
        except ValueError:
            print(f"ERROR: --extra-input 维度非法（got {spec!r}）", file=sys.stderr)
            sys.exit(1)
        if not name or not dims or any(dim <= 0 for dim in dims):
            print(f"ERROR: --extra-input 需要非空名字与正整数维（got {spec!r}）", file=sys.stderr)
            sys.exit(1)
        if name == "obs" or any(existing == name for existing, _ in extra_specs):
            print(f"ERROR: --extra-input 名字重复或与 obs 冲突（got {name!r}）", file=sys.stderr)
            sys.exit(1)
        extra_specs.append((name, dims))

    _export_with_onnx(obs, act, args.output, args.model_type, sequence_length, feature_size,
                      extra_specs)

    input_desc = f"[1,{obs}]" if sequence_length == 1 else f"[1,{sequence_length},{feature_size}]"
    for name, dims in extra_specs:
        input_desc += f" + {name}[1,{'x'.join(str(dim) for dim in dims)}]"
    print(f"wrote {args.output}: obs float32{input_desc} -> actions float32[1,{act}] (zero policy)")


if __name__ == "__main__":
    main()
