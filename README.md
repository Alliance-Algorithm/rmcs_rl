# rmcs_rl

RMCS（RoboMaster Control System）的**通用强化学习（RL）策略部署包**。

将训练好的 RL 策略（ONNX 模型）以标准 Component 的形式接入 RMCS 控制系统：
在 ROS2 上位机节点内完成"观测构建 → 推理 → 动作 → 低层控制"的完整闭环，
输出关节力矩命令交由机器人硬件组件下发。

## 设计原则

- **与机器人解耦**：策略合同（观测/动作格式）与参数（关节描述、PD 分组、缩放系数）全部通过YAML 配置。
- **与训练对齐**：观测构建与低层控制（PD）与训练环境保持同构，保证 sim-to-real 一致性。
- **可选集成**：本包不依赖 `rmcs_core`，以独立子模块形式引用。

## 工作方式

本包以 `rmcs::rl::RlController` 组件的形式接入 RMCS 的组件框架
（`rmcs_executor` 接口图，按路径配对，跨包通信）：

```
机器人硬件组件（rmcs_core 等）
   │  关节角度/速度、IMU（四元数/角速度）
   ▼
RlController
   ├─ 构建观测（按训练合同缩放，N 关节）
   ├─ ONNX Runtime CPU 推理（频率可配，默认 50Hz）
   ├─ 动作 → 目标换算 → PD（位置组 + 速度组，输出力矩）
   └─ FSM：INIT / IDLE / PREPARE / RL + failSafe + reset
   │
   ▼
  关节力矩 → 机器人硬件组件 → 电机
```

## 特性

- **观测构建**：与训练环境逐项对齐（指令、目标高度、角速度、重力投影、关节位置/速度、上一帧动作），
  缩放系数全部可配
- **推理**：ONNX Runtime（CPU），单输入 `obs` → 单输出 `actions`，模型合同严格校验，
  不匹配即拒绝进入 RL（安全退出）
- **低层控制**：关节分为"位置 PD 组"与"速度 PD 组"，动作分别换算为位置目标/速度目标后做 PD，
  输出力矩并限幅——与常见轮式/足式训练合同同构
- **状态机**：INIT(0)/IDLE(1)/PREPARE(2)/RL(3) 四态，PREPARE 负责从任意姿态插值到预备位，
  异常（观测非有限、推理失败等）自动 failSafe 退回 IDLE；支持 RMCS 统一的 reset 语义
- **参数化**：全部参数走 YAML（`rmcs_bringup` 配置），无硬编码机器人常量

## 快速开始

### 项目构建

先保证 `rmcs_executor` 正确构建，然后引用本包并构建：

```sh
# 进入工作空间的 src/ 目录下，引用本包（可选子模块，与 rmcs_auto_aim_v2 一致）
cd <RMCS 仓库>
git submodule add <本仓库 URL> rmcs_ws/src/rmcs_rl
git submodule update --init --recursive

# 构建依赖（RMCS 标准构建脚本，等同 colcon build）
build-rmcs
```

> 首次配置本包时，CMake 会自动下载官方 ONNX Runtime 预编译包（约 6MB，SHA256 校验）。
> 不需要本包时直接删除目录即可（`rmcs_core` 不依赖本包，构建自动跳过）。

### 放置策略模型

将训练导出的 `policy.onnx` 放到任意路径（如 `models/` 或 `/opt/rmcs/policies/`），
合同要求见 [models/README.md](models/README.md)。

### 3. 配置

在 `rmcs_bringup/config/<robot>.yaml` 中注册组件并填写参数：

```yaml
rmcs_executor:
  ros__parameters:
    update_rate: 1000.0
    components:
      - <机器人硬件组件> -> <实例名>
      - rmcs::rl::RlController -> rl_controller

rl_controller:
  ros__parameters:
    rl_model_path: "/path/to/policy.onnx"
    rl_inference_frequency: 50.0
    joint_names: [...]            # 关节名（训练 DOF 序）
    joint_base_path: "/<robot>/<subsystem>"   # 接口路径前缀，须与硬件组件输出一致
    position_pd_joints: [...]     # 位置 PD 组（关节索引）
    velocity_pd_joints: [...]     # 速度 PD 组（关节索引）
    # ... 其余参数见代码内注释与参数表
```

关键参数一览：

| 参数 | 含义 |
|---|---|
| `rl_model_path` | 策略 ONNX 路径 |
| `rl_inference_frequency` | 推理频率（Hz） |
| `rl_obs_size` / `rl_action_size` | 模型合同尺寸（默认 10+3N / N，N = 关节数） |
| `joint_names` | 关节名列表（训练 DOF 序） |
| `joint_base_path` | 关节接口路径前缀 |
| `position_pd_joints` / `velocity_pd_joints` | 位置 PD / 速度 PD 组关节索引 |
| `default_dof_pos` / `dof_pos_limits_*` | 位置组目标中位与限位 |
| `position_action_scale` / `velocity_action_scale` | 动作 → 目标换算系数 |
| `position_kp/kd`、`velocity_kp/kd` | PD 增益（与训练一致） |
| `obs_*_scale`、`clip_*` | 观测缩放与裁剪 |
| `auto_enter_rl`、`prepare_*` | FSM 行为 |

### 4. 运行

构建完成后（见"项目构建"），启动 RMCS：

```sh
launch-rmcs
```

启动后可通过 `ros2 topic echo` 查看 `{joint_base_path}/rl/observation`、
`{joint_base_path}/rl/action`、`{joint_base_path}/rl/state` 进行调试。

## 目录结构

```
rmcs_rl/
├── src/
│   ├── rl_controller.cpp          # 主组件（观测/推理/PD/FSM）
│   └── onnxruntime_inference.hpp  # ONNX Runtime 封装（合同校验）
├── models/                        # 策略模型与合同说明
├── package.xml
├── CMakeLists.txt                 # onnxruntime 自动下载（官方预编译包）
└── plugins.xml
```

## 运行期依赖

- `rmcs_executor`、`rclcpp`（ROS2 Jazzy）
- `libonnxruntime.so.1`（构建时由 CMake 自动下载官方预编译包；运行期需在加载路径中，
  例如安装到 `/usr/local/lib` 并 `ldconfig`）

## 相关

- 训练侧参考：`wheeled-legged_RL` / `legged_gym` 等策略训练仓库（导出 ONNX 即可部署）
- 集成示例：`rmcs_bringup/config/` 下各机器人配置
