# rmcs_rl

RMCS 通用 **RL 策略部署**包（机器人无关，推理运行在 mini PC / Ubuntu 24.04）。
轮腿步兵、四足、其他轮式机器人只要满足同一策略合同即可复用；不需要就不引用（可选子模块）。

- `src/rl_controller.cpp` — `rmcs::rl::RlController` 组件（通用参数化，机器人描述全走 YAML）：
  - 观测构建（与训练合同对齐，布局见下）
  - ONNX Runtime CPU 推理（默认 50Hz 锁频）
  - PD（位置组 + 速度组，输出力矩，默认 1000Hz 执行环）
  - FSM（INIT/IDLE/PREPARE/RL）+ failSafe + RMCS reset 语义
- `src/onnxruntime_inference.hpp` — 极简 ONNX Runtime 封装（单输入 `obs` → 单输出 `actions`，float32）
- `models/` — 策略 ONNX 存放位置

## 机器人无关的参数化（YAML）

| 参数 | 含义 | 轮腿默认 |
|---|---|---|
| `joint_names` | 关节名（训练 DOF 序，N 个） | `[rf0, rf1, r_wheel, lf0, lf1, l_wheel]` |
| `joint_base_path` | 接口路径前缀 | `/wheel_leg` |
| `position_pd_joints` | 位置 PD 组（关节索引） | `[0,1,3,4]` |
| `velocity_pd_joints` | 速度 PD 组（关节索引；其位置观测在 obs 中置零） | `[2,5]` |
| `default_dof_pos` / `dof_pos_limits_*` | 位置 PD 组的目标中位/限位 | 轮腿训练值 |
| `position_action_scale` / `velocity_action_scale` | 动作→目标换算 | 0.5 / 10.0 |
| `position_kp/kd`、`velocity_kp/kd` | PD 增益（与训练一致） | 200/4、20/0.5 |
| `rl_obs_size` / `rl_action_size` | 模型合同尺寸（默认 10+3N / N） | 28 / 6 |

换机器人：改上述参数 + 换 `policy.onnx` 即可，代码零改动。

## 策略合同（models/policy.onnx 必须满足）

- 单输入 `obs`：float32，`[1, obs_size]`；单输出 `actions`：float32，`[1, action_size]`
- obs 布局：`cmd3 | height_cmd(×obs_height_scale) | ang_vel3(×obs_ang_vel_scale) | gravity3
  (×obs_gravity_scale) | joint_pos(N)(×obs_dof_pos_scale，velocity_pd 关节置零) |
  joint_vel(N)(×obs_dof_vel_scale) | last_actions(N)`
- actions 与 `joint_names` 序一一对应；位置组目标 = `position_action_scale×a + default_dof_pos`，
  速度组目标 = `velocity_action_scale×a`

## 与其他包的关系

| 包 | 内容 | 依赖 |
|---|---|---|
| `rmcs_core`（核心） | 各机器人硬件组件（如轮腿 `WheelLegInfantry`）、电机驱动（`dm_motor.hpp` 等） | 无本包依赖 |
| **本包**（RL 部署） | 推理 + 控制 + 模型 | 仅 `rmcs_executor` + `rclcpp` |

接口按路径配对（rmcs_executor 接口图）：本包读取 `{joint_base_path}/*/{angle,velocity}` 与
`{joint_base_path}/imu/*`（硬件组件输出），写回 `{joint_base_path}/*/control_torque`。

## 子模块用法（可选）

需要就用、不需要就不引用：

```bash
# 方式 A：作为 git 子模块（与 rmcs_auto_aim_v2 一致）
cd <RMCS 仓库>
git submodule add <本仓库 URL> rmcs_ws/src/rmcs_rl
git submodule update --init --recursive

# 方式 B：不想要就直接删目录（rmcs_core 不依赖本包，colcon 自动跳过）
rm -rf rmcs_ws/src/rmcs_rl
# 同时把 rmcs_bringup/config/wheel-leg-infantry.yaml 里的
#   rmcs::rl::RlController -> rl_controller
# 一行注释/删除即可（硬件组件 WheelLegInfantry 不受影响）
```

## 构建

```bash
source /opt/ros/jazzy/setup.bash
cd <RMCS>/rmcs_ws
colcon build --packages-select rmcs_rl   # 首次配置时自动下载 onnxruntime (~6MB)
```

## 运行期依赖

`libonnxruntime.so.1` 需在加载路径（`deploy/install_wheelleg.sh` 会自动装到 /usr/local/lib）。
