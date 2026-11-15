# rmcs_rl

通用 RMCS 强化学习策略桥。`rmcs_rl::RlBridge` 从组件接口读取观测、组装历史帧，在进程内的单个 CPU worker 运行 ONNX 策略，再由动作 partner 发布结果。车型的状态估计、命令生成和执行器控制由对应组件负责。

在已配置依赖的 RMCS 工作区中构建：

```sh
bash .script/build-rmcs --packages-select rmcs_rl
```

ONNX Runtime 1.20.0 由 CMake 按固定 SHA-256 下载。模型输入支持 `[N]`、`[1, N]`、`[1, T, F]`，其中 `T = history_length`、`F` 由观测项计算，`N = T × F`。模型输入和输出使用 float32。

## 参数

| 参数 | 必填 | 含义 |
|---|---|---|
| `rl_base` | 是 | 状态输出前缀 |
| `rl_model_path` | 是 | 绝对路径或相对 `rmcs_rl` share 目录的路径 |
| `policy_rate` | 是 | 策略频率，Hz；必须整除执行器频率 |
| `history_length` | 是 | 历史帧数 |
| `max_action_age` | 是 | 从提交观测算起的动作有效时长，秒 |
| `enable_interface` | 是 | bool 使能接口 |
| `reset_interface` | 是 | `std::size_t` 复位计数接口 |
| `expected_metadata.<key>` | 否 | 与模型元数据逐键比较，字符串必须完全相等，例如 `policy_hz: "50"` |
| `observations.order` | 是 | 观测项顺序 |
| `observations.<name>.*` | 是 | 观测源、类型及预处理，见下表 |
| `actions.order` | 是 | 网络动作顺序 |
| `actions.<name>.output` | 是 | 动作输出接口路径 |
| `actions.<name>.scale` | 否 | 动作缩放，默认 `1.0` |
| `actions.<name>.clip` | 否 | 缩放后动作的 `[lower, upper]` 范围 |

YAML 浮点参数写成浮点字面量，如 `1.0`。元数据字符串保留引号，`"50"` 与 `"50.0"` 不相等。

## 观测

| `type` | 源接口 | 配置 |
|---|---|---|
| `scalar`，默认 | `double` | `source` |
| `vector3` | `Eigen::Vector3d` 或 `BaseLink::DirectionVector` | `source` |
| `array` | `std::vector<double>` | `source`、`size` |
| `scalars` | 多个 `double` | `sources`，按列表顺序读取 |
| `previous_action` | 桥内上一帧已发布动作 | 宽度等于动作数，无外部源 |

同一路径可由多个观测项复用。每项可设置 `select`、`precision`（`float64` 或 `float32`）、`source_clip`、`scale`、`clip`。处理顺序为精度舍入、源裁剪、缩放、输出裁剪；float32 项在运算阶段继续按 float32 舍入。未就绪、数组长度不匹配或选中值非有限时，该帧失效并清空会话历史。

历史从旧到新排列；复位后的首帧填满全部历史槽。采样统一使用执行器相位 `update_count % lround(update_rate / policy_rate) == 0`，观测生产者需要在同一相位先于 bridge 更新。

模型归一化和全局裁剪只读取元数据：`rmcs_obs_mean` / `rmcs_obs_std` 为逗号分隔的完整历史向量，`rmcs_obs_clip` 和 `rmcs_action_clip` 为可选正数。先归一化再裁剪观测，推理后再裁剪网络动作。车型导出的这些元数据应与训练时的预处理一致。

## 接口

| 方向 | 路径 | 类型与语义 |
|---|---|---|
| 输入 | `/predefined/update_count` | `std::size_t`，执行器 tick |
| 输入 | `/predefined/update_rate` | `double`，Hz |
| 输入 | 配置的 `enable_interface` | `bool`，必选 |
| 输入 | 配置的 `reset_interface` | `std::size_t`，主组件和 partner 均读取 |
| 输入 | 配置的观测源 | 见观测类型表 |
| 输出 | `<action.output>` | `double`，`clip(raw × scale)`；失效为 NaN |
| 输出 | `<action.output>/raw` | `double`，网络原始动作；失效为 NaN |
| 输出 | `<rl_base>/valid` | `bool`，当前动作有效 |
| 输出 | `<rl_base>/healthy` | `bool`，worker 健康；推理异常后闭锁至 reset 计数变化 |
| 输出 | `<rl_base>/action_seq` | `std::size_t`，动作序号；复位时输出 0，之后的序号继续单调递增，不会与复位前重复 |
| 输出 | `<rl_base>/policy_rate` | `double`，Hz |
| 输出 | `<rl_base>/model_metadata` | `std::map<std::string, std::string>`，已加载模型的全部自定义元数据，启动时写入一次；控制器可在 `before_updating()` 中核对编译期常量 |

动作 partner 只依赖 reset 计数，先于动作消费者执行。bridge 主组件依赖观测源，位于观测生产者之后。使能变化清空历史与动作；reset 计数变化还会清除推理异常闭锁。旧会话的推理结果不会跨 reset 发布。`previous_action` 使用缩放/裁剪后再除以 scale 的值，`/raw` 始终保留网络输出。

## YAML 示例

下例需要提供与观测宽度、历史长度和动作数相符的模型及上游接口。

```yaml
rmcs_executor:
  ros__parameters:
    update_rate: 1000.0
    components:
      - rmcs_rl::RlBridge -> rl_bridge

rl_bridge:
  ros__parameters:
    rl_base: /rl
    rl_model_path: models/policy.onnx
    policy_rate: 200.0
    history_length: 8
    max_action_age: 0.01
    enable_interface: /controller/enable
    reset_interface: /controller/reset_count
    expected_metadata:
      policy_hz: "200"
    observations:
      order: [command, gyro, previous_action]
      command: {source: /controller/command, type: vector3, clip: [-10.0, 10.0]}
      gyro: {source: /imu/gyro, type: vector3, scale: 0.5}
      previous_action: {type: previous_action, clip: [-1.0, 1.0]}
    actions:
      order: [left, right]
      left: {output: /rl/action/left, scale: 0.5}
      right: {output: /rl/action/right, scale: 0.5}
```

## 状态镜像

`rmcs_rl::StateMirror` 按 YAML 选择的接口发布关节、IMU、向量与诊断快照；同一执行器周期内的消息共用一个时间戳。可选来源没有生产者时省略该项，`mirror/topics` 列出本周期实际发布的话题。组件中不出现车型相关的接口名。已有生产者但类型不符合用途的接口在启动时报错，不会在运行中抛异常。参数、传输、录制与 v1 回放兼容见 [状态镜像](docs/state_mirror.md)。

## 保留的工具

- `tool/simulation/wheel_leg_policy_host.cpp`、`tool/simulation/wheel_leg_latency_bench.cpp`：删除前需逐项确认；时延基准记录提交与首次取用时刻。
- `tool/export_wheel_leg_training_pair.py`、`tool/prepare_wheel_leg_model.py`、`tool/install_rl_deps.sh`、`docs/troubleshooting.md`：同上，属于旧部署流程，没有迁移为新跳跃模型的正式导出；新模型与数据头由训练仓库按所选合同生成。
- `models/` 与 `model_receipts/` 全部保留，包括 16001 保底模型；保留资产不等于新跳跃控制器选用它。
- 旧 v1 接收端 `tool/visualization/wheel_leg_telemetry.py` 保留给旧客户端；新显示端使用 `tool/mirror/state_mirror_client.py` 与各自的车型适配层。

## 其他车型升级

其他车型分支升级子模块时，需要把 `<rl_base>/valid` 与 `/healthy` 的消费者从 `InputInterface<double>` 改为 `InputInterface<bool>`。动作只保留 value 与 `/raw` 两个输出，状态输出见上方接口表。
