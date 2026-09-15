#!/usr/bin/env python3
"""P0 验收探针：采集桥侧的观测/事实位并输出一行 JSON（供 p0_acceptance.sh 断言）。

桥与 RMCS 侧只用 executor 接口通信，因此要"从外面看"这些值，需要一个转发器：
config/bridge_test.yaml 里的 ValueBroadcaster 把 RL 专属接口（double）转成同名 topic，
本探针再订阅这些 topic + 桥的 obs topic。

用法:
  python3 tool/p0_probe.py --rl-base /test/rl \
      --float-topic /test/rl/valid --float-topic /test/rl/healthy \
      --float-topic /test/rl/action_age --float-topic /test/rl/action/j0 \
      --float-topic /test/rl/action/j1 --timeout 3.0
"""

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rmcs_rl_msgs.msg import Observation
from std_msgs.msg import Float64


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-base", default="/test/rl")
    parser.add_argument("--float-topic", action="append", default=[])
    parser.add_argument("--obs-topic", default=None)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help='必须收到消息的 topic；未收到则退出码 2（便于断言"完全收不到"的场景）',
    )
    args = parser.parse_args()

    rclpy.init()
    node = Node("p0_probe")
    control_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )

    received = {}
    obs_box = {}

    def make_callback(topic):
        def callback(message):
            received[topic] = float(message.data)
        return callback

    for topic in args.float_topic:
        node.create_subscription(Float64, topic, make_callback(topic), control_qos)

    obs_topic = args.obs_topic or (args.rl_base + "/obs")
    node.create_subscription(
        Observation,
        obs_topic,
        lambda message: obs_box.update(
            {
                "obs": list(message.obs),
                "obs_len": len(message.obs),
                "obs_seq": int(message.obs_seq),
                "layout_hash": int(message.layout_hash),
            }
        ),
        control_qos,
    )

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if (
            obs_box
            and len(received) >= len(args.float_topic)
            and all(topic in received for topic in args.require)
        ):
            break

    payload = {"received": received, "obs_len": obs_box.get("obs_len", 0)}
    for key in ("obs", "obs_seq", "layout_hash"):
        if key in obs_box:
            payload[key] = obs_box[key]
    print(json.dumps(payload))

    node.destroy_node()
    rclpy.shutdown()

    missing = [topic for topic in args.require if topic not in received]
    if missing:
        print("missing: " + ", ".join(missing), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
