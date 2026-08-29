# rmcs_wheel_leg_rl

RMCS 轮腿步兵 **RL 策略部署**包（方案 A：推理在 mini PC / Ubuntu 24.04）。

- `src/wheel_leg_rl_controller.cpp` — `rmcs::rl::WheelLegRLController` 组件
  - 28D 观测构建（与 legged_gym `infantry_v4` 训练合同逐项对齐）
  - ONNX Runtime CPU 推理（默认 50Hz 锁频）
  - PD（腿位置 PD + 轮速度 PD，输出力矩，默认 1000Hz 执行环）
  - FSM（INIT/IDLE/PREPARE/RL）+ failSafe + RMCS reset 语义
- `src/onnxruntime_inference.hpp` — 极简 ONNX Runtime 封装（单输入 float32[1,28] `obs` → 单输出 float32[1,6] `actions`）
- `models/` — 策略 ONNX 存放位置（训练导出，`obs[1,28] -> actions[1,6]`）

## 与其他包的关系

| 包 | 内容 | 依赖 |
|---|---|---|
| `rmcs_core`（核心） | 硬件组件 `WheelLegInfantry`、达妙驱动 `dm_motor.hpp`、DJI/LK 驱动 | 无本包依赖 |
| **本包**（RL 部署） | 推理 + 控制 + 模型 | 仅 `rmcs_executor` + `rclcpp` |

接口按路径配对（rmcs_executor 接口图）：本包读取 `/wheel_leg/*/angle|velocity` 与
`/wheel_leg/imu/*`（硬件组件输出），写回 `/wheel_leg/*/control_torque`。

## 子模块用法（可选）

需要就用、不需要就不引用：

```bash
# 方式 A：作为 git 子模块（与 rmcs_auto_aim_v2 一致）
cd <RMCS 仓库>
git submodule add <本仓库 URL> rmcs_ws/src/rmcs_wheel_leg_rl
git submodule update --init --recursive

# 方式 B：不想要就直接删目录（rmcs_core 不依赖本包，colcon 自动跳过）
rm -rf rmcs_ws/src/rmcs_wheel_leg_rl
# 同时把 rmcs_bringup/config/wheel-leg-infantry.yaml 里的
#   rmcs::rl::WheelLegRLController -> wheel_leg_rl_controller
# 一行注释/删除即可（硬件组件 WheelLegInfantry 不受影响）
```

## 构建

```bash
source /opt/ros/jazzy/setup.bash
cd <RMCS>/rmcs_ws
colcon build --packages-select rmcs_wheel_leg_rl   # 首次配置时自动下载 onnxruntime (~6MB)
```

## 策略合同（models/policy.onnx 必须满足）

- 单输入 `obs`：float32，shape `[1,28]`
- 单输出 `actions`：float32，shape `[1,6]`
- 观测/动作语义见 `wheel_leg_rl_controller.cpp` 头部注释与
  `rmcs_bringup/config/wheel-leg-infantry.yaml` 参数表

## 运行期依赖

`libonnxruntime.so.1` 需在加载路径（`deploy/install_wheelleg.sh` 会自动装到 /usr/local/lib）。
