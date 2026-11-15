# 状态镜像

`rmcs_rl::StateMirror` 把 RMCS 接口的只读快照发布为 ROS 消息。它不依赖 RL 桥；来源接口、关节顺序和诊断键全部在 YAML 中配置，同一个插件可以服务不同车型。

## 参数

| 参数 | 含义 |
| --- | --- |
| `topic_prefix` | ROS 话题的绝对前缀，末尾不带 `/` |
| `frame_id` | 同一周期内所有消息共用的 frame |
| `rate` | 发布频率（Hz），由执行器 tick 分频得到，范围 `(0, update_rate]` |
| `joints.order` | 有序的关节名；整个 joints 块可以省略 |
| `joints.<name>.position`、`.velocity` | `double` 类型的来源接口，单位 rad、rad/s |
| `joints.<name>.stamp` | 可选，`std::uint64_t` 类型的单调时钟接收时刻（ns） |
| `imu.orientation`、`.angular_velocity` | 四元数与向量来源；整个 imu 块可以省略 |
| `imu.stamp` | 可选，接收时刻（ns） |
| `vectors.order`、`vectors.<name>` | 话题后缀与向量来源接口 |
| `values.<group>.<key>` | 接口路径；诊断键为 `group/key` |
| `constants.<group>.<key>` | 字符串常量；诊断键为 `group/key` |

**绑定规则**：
- 所有来源都在 `before_pairing()` 中作为可选输入绑定，没有生产者的来源不贡献任何值。
- 任一关节缺少 position 或 velocity 时，该周期不发布整条 JointState，避免把不完整的数据当成完整快照。
- 缺少 IMU 来源时不发布 IMU；缺少某个向量来源时不发布该话题。
- 接收时刻缺失、为 0 或晚于当前时刻时，年龄输出为 `-1`。接收年龄只反映传输是否存活，与状态估计是否有效无关。

**启动期类型检查**：有生产者、但类型不符合用途的来源，会在 `before_updating()` 中带着接口路径报错，例如关节角指向了 bool 接口、stamp 不是 `std::uint64_t`、values 指向无法格式化的类型。这样不会在运行中让控制线程抛异常。

## 话题与拼帧

**话题**（QoS 为深度 5、best effort、volatile）：
- `<prefix>/joint_states`：`sensor_msgs/msg/JointState`；
- `<prefix>/imu_body`：`sensor_msgs/msg/Imu`；
- 每个已配置的向量各一个 `geometry_msgs/msg/Vector3Stamped` 话题；
- `<prefix>/diagnostics`：`diagnostic_msgs/msg/DiagnosticArray`。

**消息约定**：
- 同一 tick 发布的所有消息使用相同的 ROS header stamp 与 frame。
- 关节名按 `joints.order` 排列；角度单位 rad，角速度 rad/s，接收年龄 s。
- IMU 不提供线加速度，`linear_acceleration_covariance[0] = -1`。

**诊断**：只有一个 status，`name = topic_prefix`，level 为 OK，message 为空。
- `schema` 为 `rmcs_state_mirror_v2`。
- `mirror/topics` 是本 stamp 实际发布的消息列表，逗号分隔，包含 diagnostics 本身。接收端据此拼帧，不会因为某个可选来源缺席而一直等待。
- 来源缺失的 values 直接省略。哪些项是必需的、如何提示告警，由各车型的适配层决定。

| 来源类型 | 诊断展开 |
| --- | --- |
| `double`、`bool`、`int`、`std::uint64_t`、`std::string` | `key`（bool 为 `true`/`false`） |
| `Switch`、`ChassisMode` | `key`，整数 |
| `Keyboard` | `key`，16 位掩码 |
| Eigen `Vector2d`/`Vector3d`、BaseLink `DirectionVector` | `key/x`、`/y`，三维时另有 `/z` |
| Eigen `Quaterniond` | `key/w`、`/x`、`/y`、`/z` |
| `std::array<double, 2/4/6>`、`std::vector<double>` | `key/0` 至 `/N-1` |
| `std::uint64_t` 作为 stamp | `age_s/<joint>` 或 `age_s/imu` |

在支持的 64 位 Linux 目标上，`std::size_t` 与 `std::uint64_t` 是同一个接口类型。诊断中的数值按往返精度输出。

## 配置示例

```yaml
rmcs_executor:
  ros__parameters:
    components:
      - rmcs_rl::StateMirror -> state_mirror
state_mirror:
  ros__parameters:
    topic_prefix: /vehicle/telemetry
    frame_id: base_link
    rate: 100.0
    joints:
      order: [drive]
      drive: {position: /drive/angle, velocity: /drive/velocity, stamp: /drive/received_ns}
    imu: {orientation: /imu/quaternion, angular_velocity: /imu/gyro, stamp: /imu/received_ns}
    vectors:
      order: [projected_gravity]
      projected_gravity: /state/gravity
    values:
      control: {enabled: /control/enable}
    constants:
      coordinate: hardware_output_shaft
```

**轮腿配置**：两份 bringup 文件给出完整的八关节配置，前缀 `/wheel_leg/telemetry`，frame `chassis_body`，沿用原有的两个 RL 向量话题名。

values 中另外包含：
- 控制就绪 `control/ready`；
- 发弹链路 `shooting/*`；
- 超电 `power/supercap_enabled`；
- 四个 DM 关节的状态码 `drives/*`：0 失能，1 使能，8–14 故障。

普通配置和 gimbal-only 配置都加载镜像。gimbal-only 下 RL 相关诊断与投影重力被省略，硬件关节、IMU 和陀螺照常发布。

## 接收、录制与回放

**`tool/mirror/state_mirror_client.py`**：不需要安装 ROS。
- `FoxgloveReader(url, latest, topic_prefix, frame_id=None)` 按需加载 `rosbags` 与 `websockets`，只订阅该前缀下支持的 CDR 通道，从不发布。
- `FrameAssembler` 按诊断中的话题清单与精确 header stamp 拼帧：不跨周期混合，最多保留 32 个未拼完的 stamp，同一连接内只接受严格递增的完整 stamp。

**`Frame`** 字段：`stamp_ns`、按名字的关节 `(position, velocity, age_s|None)`、可选 IMU（`quaternion_wxyz`、`gyro`、`age_s`）、按名字的向量、字符串诊断、`frame_id`。它只检查数值有限、四元数模长为 1。必需的关节名、坐标常量、模型偏置、模型/资产身份核对属于车型适配层。

**录制与回放**：
- `LatestFrame(record=..., metadata=...)` 以独占方式新建 JSONL 录包，schema 为 `rmcs_state_mirror_record_v2`，包含帧行与来源事件。
- `ReplayReader` 保留原始时间间隔，支持回放倍速与循环。
- `read_rows`、`read_record` 同时能转换 schema 为 `wheel_leg_mirror_record_v1` 的旧录包，关节与 IMU 数值原样保留。

| 旧 v1 字段 | v2 表示 |
| --- | --- |
| positions / velocities | 按名字的 `joints`，旧六轴顺序 |
| gimbal_positions / gimbal_velocities | 按名字的云台关节 |
| quaternion_wxyz / gyro_body | `imu` |
| gravity_rl / gyro_rl | `vectors.rl_projected_gravity` / `.rl_angular_velocity` |
| sensor_age_s | 关节 / IMU 年龄 |
| rmcs_model_positions | `values["state/position/<index>"]` |
| model_offsets | `values["model_offset/<joint>"]` |
| 其他诊断/参考字段 | 展平为 `values` 中的字符串项 |
| dm_control_ready | `values["control/ready"]`（v2 含义为：驱动与 IMU 均就绪） |

**v2 中不再包含的项**：被动膝角、腿长、标定哈希。这几项已经没有生产者，轮腿显示端会把 RMCS 对照标记为不可用，独立的闭链重建不受影响。如需恢复，在状态估计器中重新输出它们，然后在 YAML 的 `values` 中加一行即可，不需要改 C++。旧的 v1 接收端代码仍留在原路径，供旧安装使用；StateMirror v2 请使用通用客户端。

## 本地策略宿主

保留的 `wheel_leg_policy_host` 仍在使用历史上的 friend 访问适配器；改为正常加载的 `PipeFeedback` 插件是单独的后续工作。

宿主接受可选的 `jump_request_count` 输入。响应包含 `state`、`enable`、`policy_valid`、`healthy`、`action_seq`、`applied_action` 与 `effort_api`，已删除的 recovery/fault 接口不再输出。旧字段的使用者需要先迁移，才能把这个宿主与重构后的控制器一起使用。宿主只用于本地仿真。

The control thread only copies source values into preallocated snapshots. ROS messages and
string formatting run in the mirror's dedicated publisher thread. A busy publisher causes
that cycle to be dropped; the control thread never waits for publication.
