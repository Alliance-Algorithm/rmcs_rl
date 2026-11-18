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
| `history_rate` | Optional; defaults to `policy_rate`. History samples advance at this rate. |
| `flat_hold_interface` | Optional required bool input when configured; valid actions and feedback become zero while held. |
| `policy_rate` | 是 | 策略频率，Hz；必须整除执行器频率 |
| `history_length` | 是 | 历史帧数 |
| `max_action_age` | 是 | 从提交观测算起的动作有效时长，秒 |
| `enable_interface` | 是 | bool 使能接口 |
| `reset_interface` | 是 | `std::size_t` 复位计数接口 |
| `observation_frame_sequence_interface` | 否 | `std::size_t` 观测帧序号接口；非零新序号触发推理，0 清空会话，重复序号不重复提交 |
| `observation_epoch_interface` | 否 | 观测会话计数；须同时配置帧序号，会话变化作废旧推理，下一有效帧重新填充历史 |
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

历史从旧到新排列；复位后的首帧填满全部历史槽。默认采样使用执行器相位 `update_count % lround(update_rate / policy_rate) == 0`，观测生产者需要在同一相位先于 bridge 更新。配置 `observation_frame_sequence_interface` 后，生产者按策略频率发布新序号，bridge 在新帧到达时提交推理；使能和控制权交接时可以立即发布首帧。序号 0 表示观测无效并清空会话。

模型归一化和全局裁剪只读取元数据：`rmcs_obs_mean` / `rmcs_obs_std` 为逗号分隔的完整历史向量，`rmcs_obs_clip` 和 `rmcs_action_clip` 为可选正数。先归一化再裁剪观测，推理后再裁剪网络动作。车型导出的这些元数据应与训练时的预处理一致。

模型可声明 `rmcs_observation_scales`，内容为逗号分隔的单帧尺度。bridge 启动时逐项核对 YAML 展开后的 scale，`tool/check_policy.py` 执行同一检查。它描述确定性特征缩放，独立于 `rmcs_obs_normalization`；遗漏或重复缩放会被拒绝。

## 接口

| 方向 | 路径 | 类型与语义 |
|---|---|---|
| 输入 | `/predefined/update_count` | `std::size_t`，执行器 tick |
| 输入 | `/predefined/update_rate` | `double`，Hz |
| 输入 | `/predefined/timestamp` | `std::chrono::steady_clock::time_point`，观测提交与动作年龄的执行器时间 |
| 输入 | 配置的 `enable_interface` | `bool`，必选 |
| 输入 | 配置的 `reset_interface` | `std::size_t`，主组件和 partner 均读取 |
| 输入 | 配置的 `observation_frame_sequence_interface` | `std::size_t`，配置后必需，观测生产者先于 bridge 更新 |
| 输入 | 配置的观测源 | 见观测类型表 |
| 输出 | `<action.output>` | `double`，`clip(raw × scale)`；失效为 NaN |
| 输出 | `<action.output>/raw` | `double`，网络原始动作；失效为 NaN |
| 输出 | `<rl_base>/valid` | `bool`，当前动作有效 |
| 输出 | `<rl_base>/healthy` | `bool`，worker 健康；推理异常后闭锁至 reset 计数变化 |
| 输出 | `<rl_base>/action_seq` | `std::size_t`，动作序号；复位时输出 0，之后的序号继续单调递增，不会与复位前重复 |
| 输出 | `<rl_base>/policy_rate` | `double`，Hz |
| 输出 | `<rl_base>/model_metadata` | `std::map<std::string, std::string>`，已加载模型的全部自定义元数据，启动时写入一次；控制器可在 `before_updating()` 中核对编译期常量 |
| 输出 | `<rl_base>/observation` | `std::vector<double>`，最近提交 worker 的逐项预处理及历史拼接结果；若存在模型级归一化，由 worker 后续处理 |
| 输出 | `<rl_base>/observation_sequence` | `std::size_t`，上述观测的提交序号；复位为 0 |

动作 partner 依赖 reset 计数与配置的 flat hold，先于动作消费者执行。bridge 主组件依赖观测源，位于观测生产者之后。使能变化清空历史与动作；reset 计数变化还会清除推理异常闭锁。旧会话的推理结果不会跨 reset 发布。`previous_action` 使用缩放/裁剪后再除以 scale 的值，`/raw` 是缩放前的动作；有效的 flat hold 期间执行值与 `/raw` 均为 0。

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

- `tool/simulation/wheel_leg_latency_bench.cpp` 保留，用于记录提交与首次取用时刻。
- `rmcs_rl::PipeFeedback` 由正常执行器加载，取代旧 policy host；通用协议见 [管道反馈](docs/pipe_feedback.md)，轮腿示例为 `config/simulation/wheel-leg-pipe.yaml`。
- `tool/check_policy.py` 检查部署 YAML 与 ONNX；模型与 `jump_policy.hpp` 由训练仓库的 `scripts/export_wheelbipe_port_onnx.py` 和 `scripts/generate_jump_policy_header.py` 按同一份合同生成。旧 export/prepare 工具已由这套流程取代。
- `tool/install_rl_deps.sh` 保留；当前桥的中文排障说明见 `docs/troubleshooting.md`。
- `models/` 与 `model_receipts/` 全部保留，包括 16001 保底模型；保留资产不等于新跳跃控制器选用它。
- 旧 v1 接收端 `tool/visualization/wheel_leg_telemetry.py` 保留给旧客户端；新显示端使用 `tool/mirror/state_mirror_client.py` 与各自的车型适配层。

## 其他车型升级

其他车型分支升级子模块时，需要把 `<rl_base>/valid` 与 `/healthy` 的消费者从 `InputInterface<double>` 改为 `InputInterface<bool>`。动作只保留 value 与 `/raw` 两个输出，状态输出见上方接口表。

The executor, policy and history rates must divide evenly: update_rate % policy_rate = 0,
policy_rate % history_rate = 0, with 0 < history_rate <= policy_rate <= update_rate.
Inference runs on every policy tick. History retains its prior stack until a full history period
has elapsed. The previous-action feature selects the newest accepted action whose source
observation was submitted at least one history period ago (zero after reset). Flat hold publishes
and feeds back zero; transitions reset the session without clearing a worker failure latch.

## Offline policy check

Run `python3 tool/check_policy.py --config <bringup.yaml> [--component rl_bridge]
[--model <candidate.onnx>] [--receipt <new.json>]` before deployment. It validates rate divisibility,
observation widths and selection, float32 I/O shapes, exact metadata pins, normalization/clipping,
and 256 CPU inference samples. It requires pyyaml, onnx, onnxruntime and numpy, with no ROS dependency.
`check_policy.py` and `ObservationBuilder` use the same frame-width rules; changes to either must
be reflected in the other. Relative model paths resolve against the package directory.
Receipts identify artifacts with repository-relative names and SHA-256.

动作提交和有效期使用 `/predefined/timestamp`；执行器须在调用组件 `before_updating()` 之前发布有效的 `/predefined/update_rate`。管道仿真使用 `clock: stepped`，普通车辆使用缺省的 `realtime`。

## 轮腿 V10 部署

实机和管道仿真入口均使用 `models/wheel_leg/V10_jump_apex_scaling.onnx`，
对应 receipt 位于 `model_receipts/wheel_leg/`。模型直接放在车型目录下，不另存来源
ONNX 或建立版本子目录。部署元数据绑定完整分量尺度与合同，`pilot=true` 不表示硬件已验收。

35D 帧末尾七项为 `[normal,stair,slope,recover,jump,apex,elapsed]`，尺度为
`[1,1,1,1,1,5,1]`。YAML 使用 select 将前三段重新按原顺序拼接，只缩放 apex。
外部 apex=0.46 m 对应网络输入 2.30；普通 height 指令的 ×5 是另一维。
V10 保持五帧 oldest-first，部署 ONNX 内部仅翻转一次。

轮腿观测在自起交接时递增 `/wheel_leg/rl/features/epoch` 并使帧无效，保留原有
20 ms 采样时钟；bridge 作废旧会话，下一有效帧填满全部历史。重复提交不额外追加历史。
动作失效或观测构造失败后等待新的帧序号，不将旧帧重新提交并刷新动作年龄。
跳跃期间继续读取前后速度，在 ±1 m/s 内限幅，按 `jump_vx_rate=2.0 m/s²` 限制变化率，
yaw 输入保持零。双下复位和禁用清除动作、历史与跳跃状态。

DM 启动清错只发送一次，等待新反馈再推进使能；`dm_arming_timeout_ms` 默认 100 ms。
超时或运行故障锁存失能，命令返回 NaN 后才能重新开始；发送力矩前须取得新的使能应答。
六轴反馈以 `motor_feedback_timeout_ms` 默认 20 ms 检查，IMU 保持 20 ms 阈值。
`/wheel_leg/control_ready_blockers` bit 0–5 为六轴反馈失鲜，bit 6 为 IMU 失鲜，
bit 7–10 按左髋、左膝、右髋、右膝表示未使能；未来时间戳同样不就绪。

管道反馈支持 `std::vector<double>` 数组读回，可记录 `/wheel_leg/rl/observation`；
配置输出 `type: array` 使用 double 数组初值。执行器、反馈 PD、物理与策略频率分别记录，
不能将 1000 Hz executor 当作 200 Hz Isaac 物理。管道验证不代替实车验收。
