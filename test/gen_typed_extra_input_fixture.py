#!/usr/bin/env python3
"""生成带 int64/bool 常量输入的确定性策略夹具。"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    import onnx
    from onnx import TensorProto, helper

    cast_ids = helper.make_node("Cast", ["position_ids"], ["position_float"],
                                name="cast_position", to=TensorProto.FLOAT)
    cast_mask = helper.make_node("Cast", ["attention_mask"], ["mask_float"],
                                 name="cast_mask", to=TensorProto.FLOAT)
    add_position = helper.make_node("Add", ["obs", "position_float"], ["with_position"],
                                    name="add_position")
    add_mask = helper.make_node("Add", ["with_position", "mask_float"], ["actions"],
                                name="add_mask")
    graph = helper.make_graph(
        [cast_ids, cast_mask, add_position, add_mask],
        "typed_extra_input_policy",
        [
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", 2]),
            helper.make_tensor_value_info("attention_mask", TensorProto.BOOL, ["batch", 2]),
        ],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, ["batch", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.producer_name = "rmcs_rl/gen_typed_extra_input_fixture"
    for key, value in {
        "rmcs_obs_layout": "v3-history=1|constant:0+0@2",
        "rmcs_actions_layout": "v2|#0:a|#1:b",
    }.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    path = DATA_DIR / "typed_extra_input_identity.onnx"
    onnx.save(model, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
