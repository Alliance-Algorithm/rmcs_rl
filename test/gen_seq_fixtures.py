#!/usr/bin/env python3
"""生成 rank-3 序列策略回归夹具（确定性，无训练依赖）。

产出（写入 test/data/）：
  seq_identity.onnx         obs float32[batch,2,3] -> actions float32[batch,3]
                            Flatten 后 actions[j] = x[j] + x[j+3]；
                            metadata: rmcs_model_type=transformer,
                            rmcs_history_length=2, rmcs_obs_frame_size=3。
  seq_history_mismatch.onnx 同图，但 rmcs_history_length=3（与张量 T=2 不符），
                            用于校验 policy_server 启动期拒绝。
  seq_layout_mismatch.onnx 同图，但 rmcs_obs_layout 的 v3-history=3（与张量 T=2 不符），
                            用于校验不依赖可选 metadata 的布局签名拒绝。
  seq_layout_only.onnx    同图，仅保留必需的布局签名，不带序列 metadata。
  seq_layout_dynamic.onnx 输入 T/F 为动态维度，仅用布局签名解析形状。

用法：python3 gen_seq_fixtures.py
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def _build(history: int, frame: int, metadata: dict, dynamic: bool = False) -> "object":
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    obs = history * frame
    act = frame
    # W (obs, act) = [I; I; ...] 每个历史帧块都加到同一输出槽：
    # actions[j] = sum_t x[t, j]
    weight = np.zeros((obs, act), dtype=np.float32)
    for t in range(history):
        for j in range(act):
            weight[t * frame + j, j] = 1.0
    weight_init = numpy_helper.from_array(weight, name="W")
    flatten = helper.make_node("Flatten", ["obs"], ["flat"], name="flatten_seq", axis=1)
    matmul = helper.make_node("MatMul", ["flat", "W"], ["actions"], name="sum_history")
    graph = helper.make_graph(
        [flatten, matmul],
        "seq_policy",
        [helper.make_tensor_value_info(
            "obs", TensorProto.FLOAT,
            ["batch", "sequence", "feature"] if dynamic else ["batch", history, frame])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, ["batch", act])],
        [weight_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.producer_name = "rmcs_rl/gen_seq_fixtures"
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    return model


def main() -> None:
    import onnx

    DATA_DIR.mkdir(exist_ok=True)
    shared = {
        "rmcs_obs_layout": "v3-history=2|constant:0+0+0@3",
        "rmcs_actions_layout": "v2|#0:a|#1:b|#2:c",
        "rmcs_model_type": "transformer",
        "rmcs_obs_frame_size": "3",
    }
    ok_meta = dict(shared, rmcs_history_length="2")
    mismatch_meta = dict(shared, rmcs_history_length="3")

    ok_path = DATA_DIR / "seq_identity.onnx"
    mismatch_path = DATA_DIR / "seq_history_mismatch.onnx"
    layout_mismatch_path = DATA_DIR / "seq_layout_mismatch.onnx"
    layout_only_path = DATA_DIR / "seq_layout_only.onnx"
    layout_dynamic_path = DATA_DIR / "seq_layout_dynamic.onnx"
    onnx.save(_build(2, 3, ok_meta), ok_path)
    onnx.save(_build(2, 3, mismatch_meta), mismatch_path)
    layout_mismatch_meta = dict(ok_meta,
                                rmcs_obs_layout="v3-history=3|constant:0+0+0@3")
    onnx.save(_build(2, 3, layout_mismatch_meta), layout_mismatch_path)
    layout_only_meta = {
        "rmcs_obs_layout": shared["rmcs_obs_layout"],
        "rmcs_actions_layout": shared["rmcs_actions_layout"],
        "rmcs_model_type": shared["rmcs_model_type"],
    }
    onnx.save(_build(2, 3, layout_only_meta), layout_only_path)
    onnx.save(_build(2, 3, layout_only_meta, dynamic=True), layout_dynamic_path)
    print(f"wrote {ok_path}")
    print(f"wrote {mismatch_path}")
    print(f"wrote {layout_mismatch_path}")
    print(f"wrote {layout_only_path}")
    print(f"wrote {layout_dynamic_path}")


if __name__ == "__main__":
    sys.exit(main())
