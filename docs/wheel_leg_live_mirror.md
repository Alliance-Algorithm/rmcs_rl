# 轮腿实车状态三维镜像遥测

`rmcs_rl::WheelLegMirrorComponent` 将轮腿硬件反馈和状态估计参考结果发布为 ROS 2 消息，供本机 Isaac Sim 镜像按实测姿态显示。它和其他 RMCS 插件一样通过 YAML 组件列表加载，只有输入接口与 ROS 发布器。实现位于 `rmcs_ws/src/rmcs_rl/src/visualization/wheel_leg_mirror.cpp`。

## 配置与构建

以下配置已经加入组件，默认 100 Hz：

- `wheel-leg-infantry-traditional.yaml`
- `wheel-leg-infantry-prone-v4.yaml`
- `wheel-leg-infantry-wheelbipe.yaml`
- `wheel-leg-infantry-wheelbipe-recovery.yaml`

其他轮腿配置需要同时具备 `WheelLeg`、`WheelLegChassisController` 与 `WheelLegStateEstimator` 提供的接口，然后加入：

```yaml
rmcs_executor:
  ros__parameters:
    components:
      # 原有组件列表中追加
      - rmcs_rl::WheelLegMirrorComponent -> wheel_leg_mirror

wheel_leg_mirror:
  ros__parameters:
    enabled: true
    update_rate: 100.0
```

`enabled: false` 时组件不注册输入或 ROS 发布器。`update_rate` 支持 `(0,1000]` Hz，仍受执行器更新频率限制。原有 `value_broadcaster.forward_list` 继续负责标量显示，无需为关节与 IMU 添加标量转发项。

这里的 `enabled` 是遥测开关，与 `/chassis/control_enable` 和腿电机使能是三个不同状态。手动搬动失能的实车时，应保持 `wheel_leg_mirror.enabled: true`。当前轮腿硬件在失能时仍更新编码器/IMU，并以零增益、零力矩的反馈轮询帧读取 DM 轴；镜像不向电机发送命令。

使用默认配置时只追加组件列表中的一行即可，参数块可省略。传统控制和 RL 控制均可使用，镜像组件不要求加载 `RlBridge` 或策略模型。

之前自定义配置中的 `rmcs_core::broadcaster::WheelLegTelemetry -> wheel_leg_telemetry` 应替换为上述组件行，并将参数块改名为 `wheel_leg_mirror`。ROS 话题、单位、标定检查和 JSONL 记录格式保持一致。

在原构建环境运行：

```bash
build-rmcs --packages-select rmcs_rl rmcs_core rmcs_bringup
source rmcs_ws/install/setup.bash
```

部署后按原流程启动所选 RMCS 配置。例如现有传统配置入口为：

```bash
ros2 launch rmcs_bringup rmcs.launch.py robot:=wheel-leg-infantry-traditional
```

当前 `wheel-leg-infantry-wheelbipe-recovery` 已启用镜像。使用该配置时启动：

```bash
ros2 launch rmcs_bringup rmcs.launch.py robot:=wheel-leg-infantry-wheelbipe-recovery
```

无需使能底盘或策略即可显示姿态，只要板卡、电机和 IMU 仍供电并持续返回数据。GUI 会显示 `DISABLED (feedback still live)` 及 DM ready 状态。

在相同 ROS 2 环境、相同 domain 中启动 [Foxglove bridge](https://docs.foxglove.dev/docs/getting-started/frameworks/ros2)：

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

默认 WebSocket 地址为 `ws://实车IP:8765`。若使用自定义话题白名单，应包含下面六个话题。本机镜像仅订阅它们，不发控制命令。

## 消息约定

全部消息使用 `chassis_body` frame、rad / rad/s / m，QoS 为 best effort、volatile、深度 5。发布器以单调时钟节流，同一个执行器周期的五条消息具有完全相同的 ROS `header.stamp`。

| 话题，前缀 `/wheel_leg/telemetry/` | 消息 | 内容 |
|---|---|---|
| `joint_states` | `sensor_msgs/msg/JointState` | 四个腿部主动轴、两个轮子以及云台 yaw/pitch 的硬件层输出轴角与角速度 |
| `imu_body` | `sensor_msgs/msg/Imu` | 安装补偿后的姿态四元数、机体角速度，以及 body specific force（包含重力） |
| `imu_raw` | `sensor_msgs/msg/Imu` | 每个新加速度源样本；header 时间锚定 IMU 样本时间，供本机手动搬动位移积分 |
| `rl_projected_gravity` | `geometry_msgs/msg/Vector3Stamped` | 状态估计器的机体系单位重力向量，尚未做策略观测缩放 |
| `rl_angular_velocity` | `geometry_msgs/msg/Vector3Stamped` | 机体角速度，尚未做策略观测缩放 |
| `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 有效性、反馈年龄、标定身份、模型轴角、被动膝内角和腿长 |

`joint_states.name` 固定依次包含 `left_hip_joint, left_knee_joint, right_hip_joint, right_knee_joint, left_wheel, right_wheel, gimbal_yaw_joint, gimbal_pitch_joint`。接收端仍必须按名称寻址，因为训练模型的主动关节顺序不同；旧的六轴记录仍可回放，此时云台明确标记为未测量并使用名义零位。

硬件层已处理左右电机方向、轮子减速比 15.62 以及 IMU 安装变换；接收端不能重复处理。四个腿轴到模型坐标的公式为 `-q_api + kMotorOffsets[i]`，轮轴为原值。`left_knee_joint` / `right_knee_joint` 是主动传动轴，需闭链解算才能得到被动膝内角。

云台角度来自 `/gimbal/yaw/angle` 与 `/gimbal/pitch/angle`，速度来自对应的
`/velocity` 接口。yaw 直接对应模型 `M_joint1`；RMCS 的 pitch 约定为正角度向下，
模型 `M_joint2` 约定为正角度向上，因此镜像使用 `q_model_pitch = -q_rmcs_pitch`。
云台零点由 `wheel_leg` YAML 中的 `yaw_motor_zero_point` 和
`pitch_motor_zero_point` 处理一次，镜像不重复减零点。摩擦轮和拨弹轮速度保留在
diagnostics 中供 Foxglove 查看，不参与车体几何。

`imu_body` 的 ROS 四元数字段为 `x,y,z,w`，内部 Eigen 为 `w,x,y,z`。`linear_acceleration` 是安装补偿后的 body specific force（m/s²，包含重力）；没有有效样本时才将 `linear_acceleration_covariance[0] = -1`。`imu_raw` 只在加速度 sequence 变化时发布，保留源样本时间间隔。共同发布时间表示执行器快照；CAN 与 IMU 的采集仍异步，因此 diagnostics 还包含各自的单调时钟反馈年龄。

diagnostics 的 status 名称为 `wheel_leg/telemetry`，关键值包括：

- `schema=wheel_leg_telemetry_v1`、`coordinate=rmcs_hardware_output_shaft_v1`。
- `calibration_sha256`、`calibration_model_spec_sha256`，来自状态估计器提供的生产标定身份。
- `feedback_fresh`、`state_valid`、`reset_count`。
- `control_enabled`、`dm_control_ready`：仅供界面标注，不参与遥测发布和模型姿态更新的门控；旧记录没有这些字段时显示 UNKNOWN。
- `operator/ready`、`operator/remote_fresh`、`operator/vx`、`operator/vy`、`operator/wz`、`operator/height_m`、`operator/mode`：RMCS 已解释的遥控输入，供本机物理车复用，避免在 Python 重写双下/双中状态机。
- `remote/*`：原始双摇杆、左右拨杆、knob 和键盘 bitmask；不存在时表示该配置没有这些接口。
- `rl/state`、`recovery/phase`、`recovery/fault`：实车当前控制器状态，仅作对照。
- `imu/acceleration_sample_ns`、`imu/acceleration_sequence`、`imu/acceleration_convention=body_specific_force_including_gravity_mps2`。
- `imu_age_s`、`motor_age_s/<轴名>`；缺失或异常未来时间为 -1。
- `model_position_rad/<轴名>`、四个腿轴的 `model_offset_rad/<轴名>`。
- `gimbal_state_ready`、`gimbal/yaw_angle_rad`、`gimbal/yaw_velocity_rad_s`、
  `gimbal/pitch_angle_rad`、`gimbal/pitch_velocity_rad_s`。
- `gimbal/left_friction_velocity_rad_s`、`gimbal/right_friction_velocity_rad_s`、
  `gimbal/bullet_feeder_velocity_rad_s`。
- `left/passive_knee_rad`、`right/passive_knee_rad`、左右 `leg_length_m`。

标定仍由 `rmcs_core` 中的 `WheelLegStateEstimator` 持有。它额外提供 `/wheel_leg/state/model_offsets`（四元素 double 数组）、`/wheel_leg/state/calibration_sha256` 与 `/wheel_leg/state/calibration_model_spec_sha256`（string）。镜像通过这些组件接口读取实际标定，不依赖 `rmcs_core` 的私有 C++ 头文件，也不在 `rmcs_rl` 中复制标定常量。重置仍按 `/chassis/reset_count` 立即重发快照，静态标定身份不随重置改变。

## 本机镜像

相邻的 `isaac_wheeled_rl_train` 项目提供：

```bash
scripts/view_wheel_leg_live.sh --url ws://实车IP:8765 --record session.jsonl
scripts/view_wheel_leg_live.sh --replay session.jsonl
```

映射、独立几何重建、GUI 依赖与核对流程见该项目的 `docs/WHEEL_LEG_LIVE_MIRROR.md`。协议接收与记录实现复用本仓库 `rmcs_ws/src/rmcs_rl/tool/visualization/wheel_leg_telemetry.py`，构建时也安装到 `share/rmcs_rl/tool/visualization/`，无需在 GUI 主机安装 ROS。

镜像固定世界位置，只更新本机 IMU 积分得到的实测 XYZ、实测朝向、轮腿机构和云台 yaw/pitch 姿态。它不施加重力、接触、碰撞或其他动力学，不会把手持、悬空、倒置或快速转动车体拉回站立，也不会按倾角或角速度阻止显示。名义膝角/关节范围只给出诊断，不夹紧测量输入。闭链仅根据机构锚点求解；云台关节不参与轮腿闭链约束。

RMCS `state_valid=false` 或汇总 `feedback_fresh=false` 时，镜像继续使用单个传感器年龄合格的编码器和 IMU 显示当前姿态。`state_valid=false` 时将 RMCS 的对照解算标为不可用，不会把失效参考值用于几何重建。重力方向差只显示数值，不参与姿态更新。IMU 的原有姿态融合仍可能受手动加速影响，镜像显示的是它实际返回的姿态，不再额外做加速度或重力修正。

断线、缺帧、单个反馈过期、坏数值、标定不一致或无法闭合机构时仍冻结最后有效姿态并标红。这些是数据与几何完整性检查。原有实车控制使能、力矩、故障和超时保护保持原样；软件对齐检查不代表实车几何测量或动力学验证已经完成。

## 原始 IMU 的来源

`WheelLeg` 已将 `Bmi088Ekf::AccelerationSnapshot` 的 body specific force、host receive time、BoardClock sample time 和 sequence 分开发布。IMU 安装变换与可能的传感器杆臂修正只在硬件驱动处理一次；本组件不重新做安装补偿。加速度的 BoardClock 时间先转为纳秒，镜像端据此建立源时间轴；网络到达时间和 GUI 帧率不进入积分。

## 与本机 PhysX 双车对比

本仓库只提供遥测和本地策略宿主 `tool/simulation/wheel_leg_policy_host.cpp`，它通过 stdin/stdout 接收本机物理车反馈，加载本配置绑定的 `WheelLegStateEstimator`、`WheelLegRlController`、`WheelLegObservation` 和 `RlBridge`。它不加载 `rmcs_executor` 的硬件列表、不创建 CAN、不向实车输出控制。Isaac GUI 入口和 `Shift+R` 物理归位位于相邻 `isaac_wheeled_rl_train` 仓库。
