# 构建、配置与部署

## 项目构建

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

## 放置策略模型

将训练导出的 `policy.onnx` 放到本包 `models/` 目录（命名 `policy.onnx`）。
模型随包安装进 `install/`，`sync-remote` 时自动同步到运行机——无需单独 scp。
合同要求见 [model-contract.md](model-contract.md)。

YAML 中 `rl_model_path` 写相对路径（相对本包 share 目录，开发/运行环境一致）：

```yaml
rl_model_path: "models/policy.onnx"
```

## 配置

在 `rmcs_bringup/config/<robot>.yaml` 中注册组件并填写参数。
完整带注释的可复制模板见 [`config/executor.yaml`](../config/executor.yaml)
（含 RlController / RlDebugCommand 全部参数与实车参考）。

```yaml
rmcs_executor:
  ros__parameters:
    update_rate: 1000.0
    components:
      - <机器人硬件组件> -> <实例名>
      - rmcs::rl::RlController -> rl_controller

rl_controller:
  ros__parameters:
    rl_model_path: "models/policy.onnx"
    rl_inference_frequency: 50.0
    rl_obs_size: 28                # 必填：obs 维度（与模型 shape 及观测布局一致）
    rl_action_size: 6              # 必填：action 维度（与模型 shape 一致）
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
| `rl_obs_size` / `rl_action_size` | 模型合同 obs/action 维度（**必填**，无默认；须与模型 shape 及观测布局一致） |
| `joint_names` | 关节名列表（训练 DOF 序） |
| `joint_base_path` | 关节接口路径前缀 |
| `position_pd_joints` / `velocity_pd_joints` | 位置 PD / 速度 PD 组关节索引 |
| `default_dof_pos` / `dof_pos_limits_*` | 位置组目标中位与限位 |
| `position_action_scale` / `velocity_action_scale` | 动作 → 目标换算系数 |
| `position_kp/kd`、`velocity_kp/kd` | PD 增益（与训练一致） |
| `obs_*_scale`、`clip_*` | 观测缩放与裁剪 |
| `auto_enter_rl`、`prepare_*` | FSM 行为 |

## 运行

构建完成后（见“项目构建”），启动 RMCS：

```sh
launch-rmcs
```

启动后可通过 `ros2 topic echo` 查看 `{joint_base_path}/rl/observation`、
`{joint_base_path}/rl/action`、`{joint_base_path}/rl/state` 进行调试。

## 运行期依赖

- `rmcs_executor`、`rclcpp`（ROS2 Jazzy）
- `libonnxruntime.so.1`：构建时由 CMake 自动下载官方预编译包（进 `build/`）；
  运行侧（mini PC / 运行容器）需另行安装

## 部署流程（开发容器 → 运行机）

```sh
# 1. 开发容器内：模型放入 models/，构建
cp policy.onnx src/rmcs_rl/models/policy.onnx
build-rmcs

# 2. 同步安装树到运行机（unison，含模型与配置；onnxruntime 不在其中）
sync-remote

# 3. 运行机一键安装 onnxruntime（仅首次/换版本时）
bash tool/install_rl_deps.sh remote
```

> 之后更新模型/代码只需重复 1+2（模型随 install/ 同步，无需单独 scp）；
> 仅当更换 onnxruntime 版本时才需重跑第 3 步。

## 相关

- 训练侧参考：`wheeled-legged_RL` / `legged_gym` 等策略训练仓库（导出 ONNX 即可部署）
- 集成示例：`rmcs_bringup/config/` 下各机器人配置
