# 架构与组件

## 定位

`rmcs_rl` 是 RMCS（RoboMaster Control System）的**通用强化学习策略部署包**：

- 将训练导出的 RL 策略（ONNX 模型）以 RMCS Component 形式接入控制链。
- 在 ROS 2 节点内完成“观测构建 → ONNX Runtime CPU 推理 → 动作换算 → PD → 关节力矩”。
- 与机器人解耦：策略合同、关节描述、PD 分组与缩放系数全部通过 YAML 配置。
- 不依赖 `rmcs_core`，作为独立子模块引用。

## 工作方式

```
机器人硬件组件（rmcs_core 等）
   │  关节角度/速度、IMU（四元数/角速度）
   ▼
RlController
   ├─ 构建观测（按训练合同缩放，N 关节）
   ├─ ONNX Runtime CPU 推理（频率可配，默认 100Hz）
   ├─ 动作 → 目标换算 → PD（位置组 + 速度组，输出力矩）
   └─ FSM：INIT / IDLE / PREPARE / RL + failSafe + reset
   │
   ▼
  关节力矩 → 机器人硬件组件 → 电机
```

## 组件

| 组件 | 文件 | 用途 |
|---|---|---|
| `rmcs::rl::RlController` | `src/rl_controller.cpp` | 核心策略部署控制器：观测构建、推理、PD、FSM |
| `rmcs::rl::RlDebugCommand` | `src/rl_debug_command.cpp` | 调试指令源：订阅 ROS topic 写 command 接口 |

## 状态机

`RlController` 使用四态 FSM：

| 状态 | 含义 |
|---|---|
| `INIT(0)` | 初始化 |
| `IDLE(1)` | 空闲，输出零力矩 |
| `PREPARE(2)` | 从任意姿态插值到预备位 |
| `RL(3)` | 策略闭环运行 |

异常（观测非有限、推理失败等）自动 `failSafe` 退回 `IDLE`；支持 RMCS 统一的 `reset` 语义。

## 目录结构

```
rmcs_rl/
├── CMakeLists.txt
├── package.xml
├── plugins.xml
├── README.md                # 入口说明
├── doc/                     # 架构、模型合同、部署文档
│   ├── architecture.md
│   ├── model-contract.md
│   └── deployment.md
├── config/executor.yaml     # RMCS 配置模板（复制到 rmcs_bringup/config/<robot>.yaml）
├── models/                  # 策略 ONNX 模型（随包安装到 share/rmcs_rl/models/）
├── src/                     # 全部实现与推理封装（保持扁平，无子目录）
│   ├── rl_controller.cpp
│   ├── rl_debug_command.cpp
│   └── onnxruntime_inference.hpp
└── tool/                    # 依赖安装与模型工具
    ├── install_rl_deps.sh
    ├── gen_synthetic_policy.py
    ├── gen_drive_policy.py
    └── check_policy_contract.py
```

> 模型与源码分离：`models/` 只放策略产物，`src/` 只放 C++ 实现；CMake 安装时模型进入
> `share/rmcs_rl/models/`，编译产物进入 `lib/`。头文件随实现放在 `src/`（包内使用，
> 不对外安装），与 `rmcs_executor` 内部头文件（如 `executor.hpp`）的约定一致。

## 接口约定

- 关节输入/输出路径由 `joint_base_path`、`joint_names` 和各组接口后缀共同组成。
- 调试 topic 与组件接口的详细字段见代码注释及 `rmcs_bringup` 配置示例。
