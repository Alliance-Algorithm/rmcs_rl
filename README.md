# rmcs_rl

RMCS（RoboMaster Control System）的 **RL 策略桥**：在「传统 RMCS 结构」与「独立进程里的 ONNX 策略」之间做翻译。

桥只做一件事：**按 YAML 词条把 RMCS 侧接口拼成有序 obs，发出去；把收到的 action 逐项写回 RMCS 侧的 RL 专属接口。**
推理在另一个进程（`policy_server`），PD / FSM / 关节语义在 RMCS 侧消费组件（P1，见「现状与边界」）。

详细文档：

- [架构与组件](planning/docs/architecture.md)
- [桥式重构设计（v2，定稿方案）](planning/docs/bridge-design.md)
- [构建、配置与部署](planning/docs/deployment.md)
- [策略模型合同（张量形状 + 元数据要求）](planning/docs/model-contract.md)

## 两条硬边界

| 边界 | 通道 | 数量 |
|---|---|---|
| 桥 ↔ **RMCS 侧**（executor 内的组件） | 只有 `register_input` / `register_output` 接口，**没有任何 topic** | N 个观测输入 + M 个动作输出 + 使能/复位/事实位 |
| 桥 ↔ **策略进程**（不是 RMCS 组件，独立可执行文件） | ROS topic | **2 条必需**（`<rl_base>/obs`、`<rl_base>/action`）+ 1 条可选运维（`<rl_base>/policy_status`，缺省关） |

topic 存在的**唯一**原因是「策略进程独立」。想消除这 3 条 topic，只能把 ONNX 放回控制进程（进程内推理）——
那正是桥要避免的事（ORT 会跟着 1 kHz 控制进程一起加载/崩溃/重启）。

## 组件与可执行文件

| 名称 | 位置 | 依赖 ONNX Runtime | 职责 |
|---|---|---|---|
| `rmcs_rl::RlBridge` | 库 `rmcs_rl_bridge` | **否** | 观测组装、定频发布、动作回写、`valid/healthy/action_age` 事实位、合同指纹 |
| `rmcs_rl::PolicyServerLauncher` | 库 `rmcs_rl_bridge` | **否** | 随 executor 生命周期 fork/exec 独立的 `policy_server` 子进程，可退避重启、防孤儿 |
| `policy_server` | 可执行文件（`ros2 run rmcs_rl policy_server`） | **是**（唯一链接 ORT 的可执行文件） | 收一帧 obs → 归一化 → ONNX → 回一帧 action（纯反应式，无定时器） |
| 消息 `rmcs_rl/msg/*` | 本包 `msg/`（rosidl 生成，类型全名如 `rmcs_rl/msg/Observation`） | 否 | `Observation` / `Action` / `PolicyStatus` |

## 数据流

```
RMCS 侧 output 接口 ──(桥：按词条拼 obs)──> obs 向量 ──topic <rl>/obs──> policy_server ──ONNX──> action
        ▲                                                                                       │
        └─ RMCS 侧 consumer（P1：FSM/PD/限位，写电机 control_*）<──(桥：逐项回写)── action 向量 <─┘
```

## 部署到真机

本包**不含台架夹具**：链路必须挂到真实 RMCS 组件上跑。简要步骤（完整流程见
[planning/docs/deployment.md](planning/docs/deployment.md)）：

1. 改 `config/executor.yaml`：观测/动作词条与 `joint_*` 指向真机的接口路径，
   `policy_server.rl_model_path` 指向已盖章的模型；
2. 把该文件复制到 `rmcs_bringup/config/<robot>.yaml`，或真机上 `--params-file` 直接加载；
3. 真机起两个进程：

```sh
ros2 run rmcs_executor rmcs_executor --ros-args --params-file <deploy.yaml>
ros2 run rmcs_rl policy_server --ros-args --params-file <deploy.yaml>
```

`layout_hash` 两侧必须一致；`valid=0` 的原因会直接打在桥的日志里（见
[planning/docs/deployment.md](planning/docs/deployment.md) 的排障表）。注意 P1 的 RMCS 侧 consumer 尚未实现，
真机上桥恒 `valid=0`、不会输出权威动作（见「现状与边界」）。

## 加一台新车型 / 加一条观测词条

1. 在部署 YAML 的 `rl_bridge.ros__parameters` 里写 `observation_terms`（**顺序就是 obs 索引顺序**）
   与 `action_terms`（`index=` 必须恰好是 `0..M-1` 的完整置换）。
2. 只写路径 + 取值语义，**不要写 `type=`**：接口的 C++ 类型由桥在 `before_pairing` 里按 `output_map`
   的 `typeid` 自省（`take=x` / `take=vec3` / `transform=projected_gravity` 决定取值语义与 dim）。
   `type=` 是逃生口，只在自省失败（提示「无 producer 或类型不在候选表」）时才显式写。
3. 词条引用的接口**必须在启动时已存在**（配对只在启动做一次，运行期热加接口不支持）；
   确实可能缺失的可选接口用 `default=` 兜底（仅标量路径词条）。
4. 尺寸自检：`Σdim(observation_terms)` 与 `rl_obs_size`、`action_terms` 条数与 `rl_action_size` 写了就必须一致。
5. 改完重新盖章 + 重新校验，**然后重启 executor**（新接口/新词条都要重新配对）：

```sh
python3 src/rmcs_rl/tool/stamp_layout_metadata.py --model <model.onnx> --from-config <deploy.yaml> --node rl_bridge
python3 src/rmcs_rl/tool/check_policy_contract.py <model.onnx> --config <deploy.yaml> --node rl_bridge
```

> 词条**增删或维度变化**等于换了策略输入维度 → 必须重新导出模型（盖章只改 metadata，不改张量 shape）。
> 仅「维度不变」的改名/换实现（例如 v1 的类型前缀 id → v2 去类型化 id）才只需要重盖章，不需要重训。

## 工具链回归

```sh
# 不需要 executor：词条语法自检 + gen/stamp/check 正例 + 6 个负例
bash src/rmcs_rl/tool/test_layout_contract.sh
```

实测 **PASS=21 FAIL=0**。CI 里也会跑这一项，外加对 `models/*.onnx` 逐个做模型自检。

## 工具链

| 工具 | 用途 |
|---|---|
| `tool/rl_layout.py` | 词条语法 / 规范串 / `layout_hash` 的**单一真源**（`python3 tool/rl_layout.py` 跑自检） |
| `tool/stamp_layout_metadata.py` | 部署 YAML → 写进 ONNX `metadata_props`（`rmcs_obs_layout` / `rmcs_actions_layout` / `policy_layout_hash` …） |
| `tool/check_policy_contract.py` | 两种模式：`MODEL`（模型自检：张量契约 + metadata 自洽，不比对 YAML，CI 用）/ `MODEL --config X [--node N] [--print-layout] [--expect-model-id H]`（YAML ↔ 张量 ↔ metadata 完整校验） |
| `tool/gen_synthetic_policy.py` | `--from-config X -o M.onnx`：生成零动作合成模型（`obs[1,N] → actions[1,M]`；ONNX 后端强制 `ir_version=10`，因为 ORT 1.20 拒收 IR 13） |
| `tool/test_layout_contract.sh` | Python 侧布局契约回归（含 6 个必须 FAIL 的负例） |
| `tool/install_rl_deps.sh` | `bash tool/install_rl_deps.sh local\|remote`：装 `libonnxruntime.so.1`（仅 `policy_server` 需要） |

## 现状与边界

- **P0 已完成**：桥、策略进程、消息定义、合同指纹（v2）、工具链。
- **P1 进行中**：RMCS 侧 consumer（FSM / PREPARE / kp,kd / 限位 / NaN 让位）。
  deformable 已有消费侧落地（`DeformableRlSuspension` / 仲裁）；wheel-leg 尚未迁到桥式，
  因此轮腿上桥若挂载且 `enable_default: false` → **桥恒 `valid=0`**，不会输出权威动作。
- 桥**从不写电机 `control_*` 接口**：executor 禁止同名 output，电机控制权始终在 RMCS 侧消费组件手上，
  所以「RL 接管 / 退回传统控制」不需要仲裁组件（consumer 用写 NaN 让位）。
- 观测接口运行期热加不支持（配对只在启动做一次）；`contract_ok` 一旦因合同不符锁存为 false，
  当前实现**只能靠重启 executor 进程恢复**。

## 目录结构

```
rmcs_rl/
├── README.md                 # 本页（入口）
├── config/
│   └── executor.yaml         # 实车配置模板（轮腿；复制到 rmcs_bringup/config/<robot>.yaml）
├── planning/
│   └── docs/
│       ├── architecture.md       # 进程/组件、数据流、接口清单、valid 状态机
│       ├── bridge-design.md      # 重构定稿方案（权威设计）
│       ├── deployment.md         # 构建 → 模型 → 配置 → 运行 → 验证 → 交接
│       ├── model-contract.md     # 模型张量合同与 metadata 清单
│       └── deformable-rl-pipeline.md  # deformable 消费侧落地说明
├── models/                   # 策略 ONNX（安装到 share/rmcs_rl/models/）
├── msg/                      # Observation / Action / PolicyStatus（rosidl 生成，类型名 rmcs_rl/msg/*）
├── src/
│   ├── rl_bridge.cpp         # 桥
│   ├── rl_layout.hpp         # FNV-1a64 / layout_hash / model_id（与 tool/rl_layout.py 同构）
│   ├── policy_server.cpp     # 策略进程（独立可执行文件，非 executor 组件）
│   ├── policy_server_launcher.cpp  # PolicyServerLauncher 组件（拉起/重启策略进程）
│   └── onnxruntime_inference.hpp
├── tool/                     # 见上表
├── plugins.xml               # rmcs_rl_bridge 的 pluginlib 导出
└── CMakeLists.txt
```

## 相关

- 集成示例：`rmcs_bringup/config/deformable-infantry-omni-rl.yaml`（桥式 + launcher）；
  `wheel-leg-infantry-rl.yaml` 尚未挂 RL 桥（待迁到桥式）
- 训练侧：任何能导出 `obs[1,N] → actions[1,M]` 且带 `rmcs_obs_layout` / `rmcs_actions_layout` metadata 的
  ONNX 仓库（Isaac Lab / legged_gym / rsl_rl …）都可以接
