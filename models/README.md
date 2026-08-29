# 策略模型

把训练导出的策略命名为 `policy.onnx` 放到本目录。

本目录随包安装进 `install/share/rmcs_rl/models/`，`sync-remote` 同步时自动带到运行机；
YAML 配置 `rl_model_path: "models/policy.onnx"`（相对路径，相对本包 share 目录解析）。
也支持任意绝对路径（`rl_model_path` 指向即可，此时需自行分发模型文件）。

## 合同（必须严格满足）

| 项 | 要求 |
|---|---|
| 输入 | 单 tensor，名 `obs`，float32，shape `[1, obs_size]` |
| 输出 | 单 tensor，名 `actions`，float32，shape `[1, action_size]` |
| 尺寸 | 默认 `obs_size = 10 + 3N`、`action_size = N`（N = 关节数），可用 YAML `rl_obs_size` / `rl_action_size` 覆盖 |

## 观测布局（默认合同）

```
obs = cmd3 | height_cmd | ang_vel3 | gravity3 | joint_pos(N) | joint_vel(N) | last_actions(N)
```

| 段 | 含义 | 缩放 |
|---|---|---|
| `cmd3` | 前进速度、横向速度、偏航角速度指令 | ×1.0（已限幅） |
| `height_cmd` | 目标高度 | `obs_height_scale` |
| `ang_vel3` | 机体角速度（机体系，IMU） | `obs_ang_vel_scale` |
| `gravity3` | 重力在世界系 `[0,0,-1]` 到机体系的投影（四元数旋转） | `obs_gravity_scale` |
| `joint_pos(N)` | 各关节当前角度 − 默认中位；**速度 PD 组关节置零** | `obs_dof_pos_scale` |
| `joint_vel(N)` | 各关节当前角速度 | `obs_dof_vel_scale` |
| `last_actions(N)` | 上一帧策略动作 | ×1.0 |

关节序与 `joint_names` 一致（训练 DOF 序）。

## 动作语义（默认合同）

- **位置 PD 组**：`pos_target = position_action_scale × a + default_dof_pos`
- **速度 PD 组**：`vel_target = velocity_action_scale × a`

再经 PD（`position_kp/kd`、`velocity_kp/kd`）转为力矩并限幅（`position_torque_max`、
`velocity_torque_max`）。所有系数均与训练环境一致（见 YAML 配置）。

> 具体缩放与默认值以对应机器人的训练配置为准，通过 YAML 传入，代码不写死。

## 导出示例（训练后）

```python
# 在训练侧将策略（actor）导出为 ONNX，输入 obs、输出 actions
import torch

# ... 按实际训练代码加载 actor 后：
dummy = torch.zeros(1, obs_size)
torch.onnx.export(actor, dummy, "policy.onnx",
                  input_names=["obs"], output_names=["actions"],
                  dynamic_axes={"obs": {0: 1}, "actions": {0: 1}})
```

> 名称/类型/shape 与合同不符时，`RlController` 拒绝进入 RL 状态并安全退出。
