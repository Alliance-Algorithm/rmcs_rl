# rmcs_rl

RMCS 组件：在小 PC 上以进程内方式运行 ONNX 策略。整车标定、PD 控制与执行器限幅留在
`rmcs_core`；本包只负责组装观测、执行推理并发布策略动作。

| 组件 | 源文件 | 用途 |
|---|---|---|
| `rmcs_rl::RlBridge` | `src/rl_bridge.cpp` | YAML 定义的观测与动作、ONNX Runtime worker |
| `rmcs_rl::WheelLegMirrorComponent` | `src/visualization/wheel_leg_mirror.cpp` | 供 Isaac Sim 镜像显示的轮腿遥测，见 [docs/wheel_leg_live_mirror.md](docs/wheel_leg_live_mirror.md) |

## 快速开始

一个策略配置需要四样东西：ONNX Runtime、`models/` 下的模型、车型 YAML 里的观测与控制组件，
以及把它们绑定起来的 bridge 参数。`rmcs_bringup/config` 中的轮腿配置就是参考接线。

1. **依赖。** `tool/install_rl_deps.sh local` 下载 ONNX Runtime 1.20.0 并校验 SHA-256；
   `remote <host>` 通过 SSH 在机器人上做同样的事。

2. **模型。** 把 `.onnx` 放到 `models/<robot>/`。相对路径的 `rl_model_path` 在安装后的
   `share/rmcs_rl` 内解析，所以源码树里能用的路径安装后同样能用。

3. **接线。** bridge 是被动的：每个观测都是它读取的输入，每个动作都是它发布的输出。轮腿平台：

   ```yaml
   rmcs_executor:
     ros__parameters:
       components:
         - rmcs_core::controller::chassis::WheelLegObservation -> wheel_leg_observation
         - rmcs_core::controller::chassis::WheelLegRlController -> wheel_leg_rl_controller
         - rmcs_rl::RlBridge -> rl_bridge
   ```

   | 接口 | 生产者 | bridge 参数 |
   |---|---|---|
   | `/wheel_leg/rl/features/*` | `WheelLegObservation` | `observations.<name>.source` |
   | `/wheel_leg/rl/input/enable` | `WheelLegRlController` | `enable_interface` |
   | `/chassis/reset_count` | `WheelLegChassisController` | `reset_interface` |
   | `/wheel_leg/rl/input/frame_sequence` | `WheelLegObservation` | `observation_frame_sequence_interface` |

   换到其它车型时，用它自己的组件发布同样的四类接口，再把 source 指过去；bridge 里没有任何
   轮腿专属逻辑。每个 source 都必须由执行器能排在 `rl_bridge` 之前的组件产出——可选输入
   同样计入依赖图。

4. **契约。** 保留 `expected_model_spec_sha256`、`expected_observation_contract_sha256` 与
   `expected_observation_version`（见[模型检查](#模型检查)）。它们是启动期唯一的防线：防止
   模型与错误的观测布局配对；删掉它们，加载失败就会变成静默输出错误动作。

5. **验证。** 下游开始动作之前，`valid` 与 `healthy` 必须都为 `1`：

   ```sh
   ros2 topic echo /wheel_leg/rl/valid
   ros2 topic echo /wheel_leg/rl/healthy
   ros2 topic echo /wheel_leg/rl/action_seq
   ```

   `valid` 在 bridge 使能、worker 健康、当前会话新鲜且动作有限之前一直是 `0`。失败处理与
   症状对照见 [docs/troubleshooting.md](docs/troubleshooting.md)。

## RlBridge 配置

```yaml
rl_bridge:
  ros__parameters:
    rl_base: /robot/rl
    rl_model_path: models/robot/actor.onnx
    model_type: mlp
    policy_rate: 50.0
    history_length: 1
    observations:
      order: [gyro, command]
      gyro: {source: /robot/gyro, type: vector3, scale: 0.5, clip: [-10.0, 10.0]}
      command: {source: /robot/command, type: array, size: 3, select: [0, 2]}
    actions:
      order: [left, right]
      left: {output: /robot/policy/left, scale: 0.25}
      right: {output: /robot/policy/right, scale: 0.25, clip: [-1.0, 1.0]}
    expected_metadata:
      training_contract: robot_v1
```

相对路径的 `rl_model_path` 在安装后的 `rmcs_rl` share 目录内解析。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `rl_base` | `/rl` | bridge 状态接口的前缀 |
| `policy_rate` | `50.0` | 没有帧序号时的提交频率；模型带 `policy_hz` 时必须与之一致 |
| `history_length` | `1` | 按从旧到新堆叠的帧数；reset 后的第一帧填满整个历史 |
| `max_action_age`、`inference_timeout` | `2 / policy_rate` | 动作失效前可容忍的秒数 |
| `invalid_value` | `"nan"` | 失效期间的动作输出：`"nan"`、`"zero"` 或 `"hold"` |
| `enable_interface` | `<rl_base>/enable` | 可选的 `bool`；缺失时退回 `enable_default`（`false`） |
| `reset_interface` | 无 | `std::size_t` 计数器；任何变化都开启新会话 |
| `observation_frame_sequence_interface` | 无 | `std::size_t`；每个新的正序号提交一次，而不是按频率 |
| `rl_obs_size`、`rl_feature_size`、`rl_action_size` | 无 | 可选的尺寸断言 |

### 观测

`observations.order` 决定观测帧的布局。每个命名块支持：

| 键 | 含义 |
|---|---|
| `type` | `scalar`（默认）、`vector3`、带 `size` 的 `array`、由四元数计算的 `projected_gravity`、带 `sources` 列表的 `scalars`、带数值 `value` 列表的 `constant`、`previous_action` |
| `source` | `scalar`、`vector3`、`array` 与 `projected_gravity` 的接口路径 |
| `select` | 按顺序保留的源下标；下标可重复 |
| `default` | `scalar` 源未被产出时使用的有限值 |
| `precision` | 对源和每一步运算做 `float64`（默认）或 `float32` 舍入 |
| `source_clip`、`scale`、`offset`、`clip` | 按此顺序施加：`clip(source_clip(x) * scale + offset)` |

`scalar` 绑定 `double`、`bool`、`int` 或 `std::size_t`；`vector3` 绑定 `Eigen::Vector3d` 或
`BaseLink::DirectionVector`；`array` 绑定 `std::vector<double>`。出现非有限值或某个源未就绪，
整帧被拒并重启会话。

`previous_action` 是 bridge 映射并限幅后的动作换算回归一化单位的结果，不是执行器实测动作。
需要"实际施加动作"历史的车型必须自己发布该接口并绑定为源。

### 动作与接口

每个 `actions.<name>` 块按顺序把一路模型输出映射到 `output`：`clip(raw * scale + offset)`。
动作发布 partner（`<component>_actions`）在动作消费者之前运行，发布：

| 接口 | 类型 | 含义 |
|---|---|---|
| `<output>` | `double` | 映射后的动作；失效期间为 `invalid_value` 指定的动作 |
| `<output>/raw`、`/scale`、`/offset`、`/clip_min`、`/clip_max` | `double` | 模型原始输出（失效时为 NaN）与映射关系 |
| `<rl_base>/valid` | `double` | 已使能、worker 健康、当前会话、结果新鲜且有限 |
| `<rl_base>/healthy` | `double` | 推理 worker 健康状态 |
| `<rl_base>/action_seq`、`/action_steady_ns`、`/action_age`、`/obs_seq` | | 结果序号、提交时间与年龄 |
| `<rl_base>/policy_rate`、`/model_spec_sha256`、`/observation_contract_sha256` | | 已加载模型的契约 |
| `<rl_base>/performance/inference_us`、`/latency/*` | | worker 耗时与队列计数 |

bridge 自身在 `<rl_base>/input/frame` 发布组装好的观测帧。消费者必须用 `valid` 与自身安全
状态共同作为动作门控。

执行器从不等待推理。reset、使能变化、序号回退、帧被拒或有效性丢失都会开启新会话；旧会话的
结果被丢弃。推理失败保持闭锁，直到 `reset_interface` 变化。

### 模型检查

启动期模型必须是单个 float32 输入（rank 1–3）、单个 float32 输出且尺寸与配置一致，并且零观测
必须产生有限动作。`model_type: mlp` 额外要求 rank 2 或 3，`transformer` 要求 rank 3。可选检查：

- `expected_model_spec_sha256` 对照 `leg_target_model_spec_sha256` 或 `rmcs_model_spec_sha256`；
- `expected_observation_contract_sha256`，同时要求 `policy_hz` 元数据；
- `expected_observation_version` 对照 `rmcs_observation_version`；
- `expected_obs_normalization: embedded|none` 对照 `rmcs_obs_normalization`；
- `expected_metadata.<key>` 对照任意字符串元数据。

`normalization_from_metadata`（默认 `true`）应用 `rmcs_obs_mean`/`rmcs_obs_std` 与
`rmcs_obs_clip`/`rmcs_action_clip` 元数据；`obs_clip` 与 `action_clip` 覆盖这些限幅。

## 轮腿配置

`wheel-leg-infantry-prone-v4`、`wheel-leg-infantry-wheelbipe`、
`wheel-leg-infantry-wheelbipe-recovery` 与 `wheel-leg-infantry-wheelbipe-jump-v4` 会同时加载
`RlBridge` 和 `rmcs_core::controller::chassis::WheelLegObservation`、`WheelLegRlController`。
观测组件在 `/wheel_leg/rl/features/` 下发布未归一化特征和帧序号；bridge 负责缩放与排序，
控制器用自己的反馈与安全守卫执行动作。

## 工具

- `tool/simulation/wheel_leg_policy_host.cpp`：本地仿真用的 stdin/stdout host；只加载轮腿
  状态估计、控制器、观测与 bridge 插件。
- `tool/simulation/wheel_leg_latency_bench.cpp`：仅主机端的 ONNX worker 时延基准。
- `tool/visualization/wheel_leg_telemetry.py`：实时镜像接收与 JSONL 记录。

```sh
build-rmcs --packages-select rmcs_rl rmcs_core rmcs_bringup
```
