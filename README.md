# rmcs_rl

RMCS（RoboMaster Control System）的**通用强化学习（RL）策略部署包**。

将训练好的 RL 策略（ONNX 模型）以标准 Component 的形式接入 RMCS：
在 ROS 2 节点内完成“观测构建 → ONNX Runtime CPU 推理 → 动作换算 → PD → 关节力矩”，
输出关节力矩命令交由机器人硬件组件下发。

详细文档：

- [架构与组件](doc/architecture.md)
- [策略模型合同](doc/model-contract.md)
- [构建、配置与部署](doc/deployment.md)

## 通用设计（一次实现，多车复用）

`rmcs_rl` 只描述 **RL 部署的结构**，不绑定任何车型（deformable / wheel-leg /
后续新车型都只是配置实例）。核心链路：

```
关节/IMU 反馈（机器人硬件组件提供，接口路径可配）
   │
   ▼
RlController（rmcs_executor Component）
   ├─ 观测构建：cmd3 | height_cmd | ang_vel3 | gravity3
   │            | joint_pos(N) | joint_vel(N) | last_actions(action_size)
   │            （缩放系数全部 YAML 可配；velocity_pd 关节的位置观测置零）
   ├─ ONNX Runtime CPU 推理：obs[1, obs_size] → actions[1, action_size]，
   │            合同不符（名称/维度/类型）即拒绝进入 RL
   ├─ 动作换算：位置组 pos_target = position_action_scale·a + default_dof_pos；
   │            速度组 vel_target = velocity_action_scale·a
   ├─ PD → 力矩限幅（position_kp/kd、velocity_kp/kd、torque_max）
   └─ FSM：INIT / IDLE / PREPARE / RL + failSafe + reset
   │
   ▼
   关节力矩 → 机器人硬件组件 → 电机
```

要点：

- **机器人差异全部收敛到 YAML**：关节名、接口前缀/后缀、PD 分组、合同尺寸、
  缩放/增益/限位/预备位全部可配——代码不含车型常量。
- **策略模型与部署框架解耦**：任何训练侧仓库（Isaac Lab / legged_gym / …）
  导出符合 [doc/model-contract.md](doc/model-contract.md) 合同的 ONNX 即可部署。
- **台架调试方便**：`RlDebugCommand` 通过 topic 直接发 `vx/yaw_rate/height/state` 指令与复位，
  配合 `rl_publish_network_io` 输出观测/动作，不用改代码就能手动驱动 FSM 走完 PREPARE→RL。

### 接入一台新车型

1. 训练导出 ONNX（输入 `obs`、输出 `actions`，命名与合同一致），放入 `models/` 或任意路径；
2. 在 `rmcs_bringup/config/<robot>.yaml` 注册 `RlController`（调试期加
   `RlDebugCommand`），按 [doc/deployment.md](doc/deployment.md)
   填写 `joint_names`、PD 分组、缩放与限位；
3. `build-rmcs && launch-rmcs`：先在台架/安全环境用 `RlDebugCommand` 手动走
   PREPARE→RL 验证，再上真机。

## 快速开始

### 项目构建

先保证 `rmcs_executor` 正确构建，然后引用本包并构建：

```sh
cd <RMCS 仓库>
git submodule add <rmcs_rl 仓库 URL> rmcs_ws/src/rmcs_rl
git submodule update --init --recursive
build-rmcs
```

> 首次配置时，CMake 会自动下载官方 ONNX Runtime CPU 预编译包（约 6MB，SHA256 校验）。

### 放置模型

将训练导出的 `policy.onnx` 放到 `models/policy.onnx`（或通过 `rl_model_path` 指向任意路径）。
模型随包安装到 `share/rmcs_rl/models/`，`sync-remote` 时会自动同步到运行机。
合同见 [doc/model-contract.md](doc/model-contract.md)。

### 配置与运行

在 `rmcs_bringup/config/<robot>.yaml` 中注册组件：

```yaml
rmcs_executor:
  ros__parameters:
    update_rate: 1000.0
    components:
      - rmcs::rl::RlController -> rl_controller

rl_controller:
  ros__parameters:
    rl_model_path: "models/policy.onnx"
    # ... 其他参数见 doc/deployment.md
```

构建后启动：

```sh
launch-rmcs
```

详细参数、接口、运行期依赖与部署流程见 [doc/deployment.md](doc/deployment.md)。

## 项目架构

```
rmcs_rl/
├── README.md            # 入口说明（本页）
├── doc/                 # 架构 / 模型合同 / 部署文档
├── config/executor.yaml # RMCS 配置模板（复制到 rmcs_bringup/config/<robot>.yaml）
├── models/              # 策略 ONNX（与源码分离，安装到 share/rmcs_rl/models/）
├── src/                 # 全部实现与推理封装（组件 cpp + 单头文件推理封装）
│   ├── rl_controller.cpp
│   ├── rl_debug_command.cpp
│   └── onnxruntime_inference.hpp
├── tool/                # 运行依赖安装、模型生成与合同校验
├── package.xml
├── plugins.xml
└── CMakeLists.txt
```

组件职责：

- `RlController`：核心策略部署控制器（观测构建/推理/动作换算/PD/FSM）
- `RlDebugCommand`：调试指令源（ROS topic → command 接口）

## 相关

- 训练侧参考：`wheeled-legged_RL` / `legged_gym` 等策略训练仓库（导出 ONNX 即可部署）
- 集成示例：`rmcs_bringup/config/` 下各机器人配置
