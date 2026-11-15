# RL 桥排障

## 1. 观察手段

RMCS 组件接口不是 ROS 话题。先确认 YAML 中的生产者、路径和类型，再选择转发手段。
StateMirror 可以把状态放入诊断键值；以下路径以 `/chassis/rl` 为例，使用其它 rl_base 时一并替换：

```yaml
values:
  rl:
    valid: /chassis/rl/valid
    healthy: /chassis/rl/healthy
    action_seq: /chassis/rl/action_seq
```

动作的 value/raw 是 double，可由 `rmcs_core::broadcaster::ValueBroadcaster` 转发；具体实例参数沿用
同车配置中的 ValueBroadcaster 写法。bool、整数用 StateMirror。正常启动日志为：
`policy <path>: obs=TxF action=N rate=… Hz`。先核对模型路径、历史长度、帧宽和动作数。

部署前运行通用离线检查器（不需要 ROS；依赖 pyyaml、onnx、onnxruntime、numpy）：

```sh
python3 tool/check_policy.py --config <bringup.yaml> --component rl_bridge
```

相对模型路径以 rmcs_rl 包目录为基准。`--model` 可临时指定候选模型；`--receipt` 保存检查回执。
回执路径使用 `RMCS:<相对路径>`，文件身份由 SHA-256 确认。

## 2. 启动期报错

| 报错原文 | 原因与对应键 |
| --- | --- |
| `model shape does not match configuration` | history_length、观测宽度或动作数与模型不一致；核对 observations/actions.order、size、select |
| `model metadata mismatch: <key>` | expected_metadata 字符串不一致，数字字符串也严格比较；例如 `50` 与 `50.0` 不同 |
| `model normalization does not match configuration` | rmcs_obs_mean/std 未成对出现、长度不等于 T×F、非有限或 std 非正 |
| `invalid model clipping metadata` | rmcs_obs_clip 或 rmcs_action_clip 不是正有限数 |
| `update_rate, policy_rate and history_rate must divide evenly` | 必须 update_rate 整除 policy_rate，policy_rate 整除 history_rate，且 0 < history_rate ≤ policy_rate ≤ update_rate |
| 执行器缺少输入或类型不匹配 | source、sources、enable_interface、reset_interface、flat_hold_interface 的路径/类型或生产者错误 |
| `Mirror interface has the wrong type: <path>` | 接口类型不适合指定的镜像槽位，例如关节角必须 double |
| `Mirror cannot format interface <path>` | values 来源为不支持格式化的类型 |
| `Loaded policy does not match the compiled jump policy` | 加载模型的部署合同 SHA 与编译进去的 jump_policy.hpp 不一致；模型、数据头与 YAML 必须一起更新 |

模型形状、元数据、归一化、裁剪和整除问题可先用 check_policy.py 离线复现。执行器接线、镜像
类型和编译后的跳跃合同还需要原生加载验证，检查器不能凭 YAML 确认运行中的接口生产者。

## 3. valid=false

有效条件为：enable、已有结果、healthy、generation 一致、动作年龄不超过 max_action_age、输出全部有限。

- 核对 enable_interface 的生产者和值；未使能不会提交观测。
- 首次提交等待全局策略相位，最多 policy_divisor−1 个 tick，再加推理时间。
- max_action_age 应覆盖实际推理与结果消费耗时；使用 latency bench 检查提交到取回的时间。
- reset 或使能翻转会清历史和动作，短暂无效是预期行为。
- flat hold 翻转也会出现一个策略周期加推理时间的窗口；deformable 通常约 6 ms，期间 arbiter 使用传统姿态。
- 输出 NaN/Inf 会被拒绝。检查观测来源是否出现非有限值，以及推理错误日志。

## 4. healthy=false

推理抛异常后闭锁，日志为 `inference failed: <error>`。只有 reset_interface 的计数变化才清除
worker 故障。使能翻转与 flat hold 翻转只重置会话，不清除此闭锁；先解决异常原因，再遥控复位。

## 5. 观测构建失败

每个策略 tick 都读取并检查当前来源。来源未就绪或有非有限值时，整帧丢弃并 reset()，历史和
动作反馈一并清空。常见来源是上游初始 NaN，例如 deformable 的 feedback_current_raw。按
observations.order 逐项检查来源路径、初值、发布顺序、array 长度和 select 下标。

## 6. 历史与 flat hold

policy_rate 控制推理频率，history_rate 控制模型输入历史的推进。每次策略 tick 都构建一帧；距
上个历史样本未满一个历史周期时丢弃本帧，重新提交原堆叠。首帧填满所有槽，排列从旧到新。

previous_action 取“产生该动作的观测提交 tick 不晚于当前 tick 减一个历史周期”的最新有效动作；
没有新候选时保持原值，reset 后为 0。deformable 为 200 Hz 推理、100 Hz 历史，轮腿缺省二者均 50 Hz。

flat_hold_interface 配置后为必选 bool。为真且结果有效时，value 与 raw 均为 0，反馈给下一帧的动作
也为 0。hold 变化会清会话与延迟缓冲，因此旧 generation 的结果不能重新发布。

## 7. reset 语义

reset_interface 计数变化重置会话及 worker；使能变化与 flat hold 翻转只重置会话。会话重置会增加
generation、清历史、清动作反馈和有效状态。不要用使能开关替代清除推理故障所需的复位。
