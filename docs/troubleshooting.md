# 故障排查

策略链路出问题时，先看两个接口，再按日志里的 `valid=0: <reason>` 定位。

```sh
ros2 topic echo /wheel_leg/rl/valid        # 下游是否允许动作
ros2 topic echo /wheel_leg/rl/healthy      # 推理 worker 是否健康
ros2 topic echo /wheel_leg/rl/input/frame  # 实际送入模型的观测帧
```

`valid` 在六个条件同时成立时才为 `1`（`src/rl_bridge.cpp` 的 `update_actions`）：

```
valid = enabled && !worker_failed && has_action && fresh && sequence_ok && finite
```

`valid` 每次变化打一行日志，失效原因按固定顺序报出（`invalid_reason`）：

| 日志 reason | 对应条件 |
|---|---|
| `inference worker failed` | `worker_failed_` |
| `disabled by enable interface` | `last_enabled_` |
| `no inference result available` | `has_action` |
| `action contains non-finite values` | `finite` |
| `action is from an invalid or previous reset generation` | `sequence_ok` |
| `inference result is stale` | `fresh` |

## 按原因处理

### inference worker failed

worker 失败后是**闭锁**，不会自动重试：只有 `reset_interface` 计数变化才清除（`reset_runtime()` 里同时置 `worker_failed_ = false` 并调用 `worker_.reset()`）。

启动期失败直接抛异常、进程起不来，看的是报错原文：

| 报错 | 原因 |
|---|---|
| `model path must not be empty` | `rl_model_path` 为空或解析后不存在 |
| `model must have exactly one input and one output` | 导出的图不是单入单出 |
| `obs and actions element types must be float32` | 模型 IO 不是 float32 |
| `input batch dimension must be 1 or dynamic` | batch 维写死成别的值 |
| `supported policy output ranks are 1, 2, or 3` | 输出 rank 不对（`model_type` 还会再限制一侧） |
| `startup inference produced non-finite actions` | 零观测推理出 NaN/Inf，模型本身有问题 |
| `model_spec_sha256 does not match the configured contract` | 模型换了但 `expected_model_spec_sha256` 没更新 |
| `model observation contract does not match its preprocessor` | 观测契约 sha 不符 |
| `model observation version does not match its preprocessor` | `expected_observation_version` 不符 |
| `model metadata mismatch: expected_metadata.<key>` | 自定义元数据不符 |
| `model policy_hz metadata does not match policy_rate` | 模型标称频率与 `policy_rate` 不一致 |
| `explicit observation contract requires policy_hz metadata` | 设了契约 sha，但模型缺 `policy_hz` 元数据 |
| `rmcs_obs_mean and rmcs_obs_std must appear together` | 归一化元数据缺一半 |

运行期失败写进结果、不抛异常：

| 结果里的 error | 原因 |
|---|---|
| `observation/action dimensions do not match model` | 观测或动作维数与模型不符 |
| `non-finite observation at index N` | 第 N 维观测是 NaN/Inf |
| `policy output contains non-finite values` | 模型输出非有限 |

### disabled by enable interface

`enable_interface` 读到 `false`。没有组件产出该接口时退回 `enable_default`（默认 `false`），所以“接口没接上”和“被显式关掉”表现一样。轮腿车型上该接口由 `WheelLegRlController` 产出，不是遥控输入直接给的。

### no inference result available

还没有任何结果：要么刚启动，要么每一帧都在提交阶段就被拒。后者会留下：

```
observation sources not ready or non-finite; no obs published (<n> frames rejected)
```

依次查：观测源是否有组件发布、`source` 路径是否拼错、块内 `size` 是否与实际消息长度一致。整帧被拒还会重启会话，`action_seq` 归零。

### action contains non-finite values

模型输出了 NaN/Inf。观测越界、对已归一化的特征又做了一次 `scale`、或漏了 `clip` 是最常见的三种；先用 `/wheel_leg/rl/input/frame` 看实际数值范围。

### action is from an invalid or previous reset generation

`sequence_ok` 要求 `action.generation == generation_` 且 `action.sequence <= observation_sequence_`。不满足说明手里这份结果是 reset 之前算完的，或者序号跑到了观测前面——通常是 `reset_interface` 接到了一个频繁变化的计数器。

### inference result is stale

`age > min(inference_timeout, max_action_age)`。推理耗时超预算（`policy_rate` 太高、模型太大、机器负载高），先用 `wheel_leg_latency_bench` 量一下再对表。

## reset 语义

- `reset_interface` 计数变化 → `reset_runtime()` + `worker_.reset()`：开始新会话，丢弃旧 generation 的结果。
- 失败闭锁**只**在 reset 时清除；不接 `reset_interface` 等于失败后必须重启进程。
- 遥测（`wheel_leg_mirror`）与策略是两条独立链路，手动搬动失能车辆时不受上述闭锁影响，见 `docs/wheel_leg_live_mirror.md`。
