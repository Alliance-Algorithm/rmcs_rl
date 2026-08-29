# 策略模型

把训练导出的 `policy.onnx` 放到本目录（或任意路径，YAML `rl_model_path` 指向即可）。

## 合同（必须严格满足）

| 项 | 值 |
|---|---|
| 输入 | 单 tensor `obs`，float32，`[1, 28]` |
| 输出 | 单 tensor `actions`，float32，`[1, 6]` |
| DOF 序 | `[rf0, rf1, r_wheel, lf0, lf1, l_wheel]` |
| 观测 | cmd3 \| height_cmd(×5) \| ang_vel3(×0.5) \| gravity3 \| joint_pos6(×1, 轮置零) \| joint_vel6(×0.1) \| last_actions6 |
| 动作 | 腿位置目标 = 0.5×a + default_dof_pos；轮速度目标 = 10×a |

参考：训练环境 `legged_gym/envs/infantry_v4`（Isaac Gym Preview 4 + legged_gym + rsl_rl）。
导出示例（训练后）：

```bash
# 在训练侧将 policy.pt 导出为 ONNX（actor 部分，输入 obs、输出 actions）
python -c "
import torch
from rsl_rl.runners import OnPolicyRunner
...  # 按实际训练代码加载 actor_critic 后：
dummy = torch.zeros(1, 28)
torch.onnx.export(actor, dummy, 'policy.onnx',
                  input_names=['obs'], output_names=['actions'],
                  dynamic_axes={'obs': {0: 1}, 'actions': {0: 1}})
"
```

> 名称/类型/shape 不匹配时，`WheelLegRLController` 拒绝进入 RL 状态并安全退出。
