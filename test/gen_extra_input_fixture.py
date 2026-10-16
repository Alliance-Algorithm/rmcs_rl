#!/usr/bin/env python3
"""生成多输入策略回归夹具（确定性，无训练依赖）。

产出（写入 test/data/）：
  extra_input_identity.onnx
    inputs : obs float32[batch,2], offset float32[batch,1]
    output : actions float32[batch,2] = obs + offset（广播）
    用于校验 policy_server extra_inputs.<name> 常量喂入与缺失/多余/数量错配的拒绝。

用法：python3 gen_extra_input_fixture.py
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    import onnx
    from onnx import TensorProto, helper

    DATA_DIR.mkdir(exist_ok=True)
    add = helper.make_node("Add", ["obs", "offset"], ["actions"], name="add_offset")
    graph = helper.make_graph(
        [add],
        "extra_input_policy",
        [
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("offset", TensorProto.FLOAT, ["batch", 1]),
        ],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, ["batch", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.producer_name = "rmcs_rl/gen_extra_input_fixture"
    for key, value in {
        "rmcs_obs_layout": "v3-history=1|constant:0+0@2",
        "rmcs_actions_layout": "v2|#0:a|#1:b",
    }.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    path = DATA_DIR / "extra_input_identity.onnx"
    onnx.save(model, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
