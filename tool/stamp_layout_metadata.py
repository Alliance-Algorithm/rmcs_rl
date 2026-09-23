#!/usr/bin/env python3
"""把部署 YAML 的布局契约盖章进 ONNX metadata（单一真源 = tool/rl_layout.py）。

桥（rl_bridge）在启动时看不到模型文件，因此两侧靠 layout_hash 运行期握手：
本工具把 YAML 推导出的规范串（v2）与 policy_layout_hash 写进模型 metadata，
策略进程据此与桥的 Observation.layout_hash 对账（planning/docs/bridge-design.md §6.2）。
**不需要重训**：换布局/换版本重跑本工具即可。

写入的键：
  rmcs_obs_layout      v2 观测规范串（必需）
  rmcs_actions_layout  v2 动作规范串（必需）
  policy_layout_hash   layout_hash（16 位 hex，与本工具打印值一致）
  policy_version       可选（--policy-version）
  rmcs_obs_mean/std    可选（--obs-mean/--obs-std，长度必须 == rl_obs_size）
  rmcs_obs_clip        可选单浮点（--obs-clip）
  rmcs_action_clip     可选单浮点（--action-clip）

用法：
  python3 stamp_layout_metadata.py --model policy.onnx --from-config deploy.yaml
  python3 stamp_layout_metadata.py --model policy.onnx --from-config deploy.yaml \
      --node rl_bridge --policy-version infantry_v4-2026.02 --obs-clip 100 --action-clip 100
"""
import argparse
import math
import sys

import rl_layout as layout


def _float_list(text, key, obs_size):
    """逗号分隔浮点串 → 列表（长度必须等于 obs_size）。"""
    items = [item.strip() for item in text.replace(" ", "").split(",") if item.strip() != ""]
    if not items:
        raise layout.LayoutError(f"{key} 为空")
    values = []
    for item in items:
        try:
            values.append(float(item))
        except ValueError:
            raise layout.LayoutError(f"{key} 含非法浮点数 {item!r}")
        if not math.isfinite(values[-1]):
            raise layout.LayoutError(f"{key} 含非有限值 {item!r}（NaN/Inf 会让归一化产出 NaN）")
    if len(values) != obs_size:
        raise layout.LayoutError(
            f"{key} 长度 {len(values)} != rl_obs_size {obs_size}（归一化向量必须与观测一一对应）"
        )
    if key == "rmcs_obs_std" and any(value <= 0.0 for value in values):
        raise layout.LayoutError(f"{key} 必须全部 > 0（归一化是 (x-mean)/std）")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按部署 YAML 把 v2 布局规范串 + layout_hash 写进 ONNX metadata"
    )
    parser.add_argument("--model", required=True, help="policy.onnx 路径（原地写回）")
    parser.add_argument("--from-config", "--config", dest="config", required=True,
                        help="部署 YAML（含 <node>.ros__parameters.observation_terms/action_terms）")
    parser.add_argument("--node", default=layout.DEFAULT_NODE,
                        help=f"YAML 节点名（缺省 {layout.DEFAULT_NODE}）")
    parser.add_argument("--policy-version", default=None, help="训练侧版本号（可选）")
    parser.add_argument("--obs-mean", default=None, help='经验均值，逗号分隔，如 "0.1,-0.2,..."')
    parser.add_argument("--obs-std", default=None, help="经验标准差，逗号分隔，长度同 obs_size")
    parser.add_argument("--obs-clip", type=float, default=None, help="观测 clip（单浮点，可选）")
    parser.add_argument("--action-clip", type=float, default=None, help="动作 clip（单浮点，可选）")
    args = parser.parse_args()

    try:
        obs_terms, act_terms, obs_size, act_size = layout.load_config(args.config, args.node)
        obs_sig = layout.obs_signature(obs_terms, act_size)
        act_sig = layout.action_signature(act_terms)
        digest = layout.layout_hash(obs_sig, act_sig, obs_size, act_size)

        obs_mean = _float_list(args.obs_mean, "rmcs_obs_mean", obs_size) if args.obs_mean else None
        obs_std = _float_list(args.obs_std, "rmcs_obs_std", obs_size) if args.obs_std else None
        if (obs_mean is None) != (obs_std is None):
            raise layout.LayoutError("rmcs_obs_mean 与 rmcs_obs_std 必须成对给出（归一化是 (x-mean)/std）")

        for option, value in (("--obs-clip", args.obs_clip), ("--action-clip", args.action_clip)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise layout.LayoutError(f"{option}={value!r} 必须是有限正数")

        before = layout.read_metadata(args.model)
        updates = layout.stamp_metadata(
            args.model, obs_sig, act_sig, obs_size, act_size,
            policy_version=args.policy_version,
            obs_mean=obs_mean, obs_std=obs_std,
            obs_clip=args.obs_clip, action_clip=args.action_clip,
        )
    except layout.LayoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"ERROR: 读写失败：{exc}", file=sys.stderr)
        sys.exit(1)

    changed = sorted(key for key, value in updates.items() if before.get(key) != value)
    print(f"config           : {args.config} (node {args.node})")
    print(f"model            : {args.model}")
    print(f"obs_size         : {obs_size}")
    print(f"action_size      : {act_size}")
    print(f"obs_signature    : {obs_sig}")
    print(f"action_signature : {act_sig}")
    print(f"layout_hash      : {layout.hex16(digest)}")
    print(f"layout_payload   : {layout.layout_payload(obs_sig, act_sig, obs_size, act_size)}")
    print(f"model_id         : {layout.model_id_hex(args.model)}")
    if changed:
        print(f"stamped          : {', '.join(changed)}")
    else:
        print("stamped          : (metadata 已一致，文件未改动)")


if __name__ == "__main__":
    main()
