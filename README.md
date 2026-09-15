# rmcs_rl

RMCS（RoboMaster Control System）的 **RL 策略桥**：在「传统 RMCS 结构」与「独立进程里的 ONNX 策略」之间做翻译。

桥只做一件事：**按 YAML 词条把 RMCS 侧接口拼成有序 obs，发出去；把收到的 action 逐项写回 RMCS 侧的 RL 专属接口。**
推理在另一个进程（`policy_server`），PD / FSM / 关节语义在 RMCS 侧消费组件（P1，见「现状与边界」）。

详细文档：

- [架构与组件](doc/architecture.md)
- [桥式重构设计（v2，定稿方案）](doc/bridge-design.md)
- [构建、配置与部署](doc/deployment.md)
- [策略模型合同（张量形状 + 元数据要求）](doc/model-contract.md)

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
| `rmcs::rl::RlBridge` | 库 `rmcs_rl_bridge` | **否** | 观测组装、定频发布、动作回写、`valid/healthy/action_age` 事实位、合同指纹 |
| `policy_server` | 可执行文件（`ros2 run rmcs_rl policy_server`） | **是**（唯一链接 ORT 的可执行文件） | 收一帧 obs → 归一化 → ONNX → 回一帧 action（纯反应式，无定时器） |
| `rmcs::rl::testing::SyntheticRobot` | 库 `rmcs_rl_testing` | 否 | **仅测试用**的合成观测源（台架/无硬件时验证链路） |
| `rmcs::rl::RlController` | 库 `rmcs_rl_legacy` | 是 | **遗留控制器**（旧配置格式，P1 切换完成后删除） |
| 消息 `rmcs_rl_msgs` | 独立包 | 否 | `Observation` / `Action` / `PolicyStatus` |

> ⚠️ `RlController` 是遗留路径：它自带 ONNX 推理、PD、FSM，配置键（`rl_inference_frequency`、
> `position_pd_joints`、`action_terms: joint=... mode=... kp=...`）与本文档描述的桥式配置**完全不同**，
> 不要混用、也不要与 `RlBridge` 同时挂载。桥式链路取代它之后即删除。

## 数据流

```
RMCS 侧 output 接口 ──(桥：按词条拼 obs)──> obs 向量 ──topic <rl>/obs──> policy_server ──ONNX──> action
        ▲                                                                                       │
        └─ RMCS 侧 consumer（P1：FSM/PD/限位，写电机 control_*）<──(桥：逐项回写)── action 向量 <─┘
```

## 60 秒快速开始（容器内，验证 obs→action 链路）

下面所有命令都在 dev 容器里跑（工作目录 `rmcs_ws`），用的夹具是 `config/bridge_test.yaml`
（合成观测源 + 桥 + `policy_server` 三段参数都在同一个文件里，两个节点都能直接吃）：

```sh
# 0) 环境（容器内）；install/ 还没构建就先构建：
#    colcon build --packages-up-to rmcs_rl --symlink-install --merge-install
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# 1) 由部署 YAML 生成一个零动作合成策略（只为打通链路，不是真策略）
mkdir -p /tmp/rmcs_rl_accept
python3 src/rmcs_rl/tool/gen_synthetic_policy.py --from-config src/rmcs_rl/config/bridge_test.yaml \
    --node rl_bridge -o /tmp/rmcs_rl_accept/bridge_test.onnx

# 2) 盖章：把 YAML 推导出的 v2 规范串 + layout_hash 写进模型 metadata
python3 src/rmcs_rl/tool/stamp_layout_metadata.py --model /tmp/rmcs_rl_accept/bridge_test.onnx \
    --from-config src/rmcs_rl/config/bridge_test.yaml --node rl_bridge

# 3) 校验三方一致（YAML / 张量形状 / metadata），末尾应打印 == SUMMARY: PASS ==
python3 src/rmcs_rl/tool/check_policy_contract.py /tmp/rmcs_rl_accept/bridge_test.onnx \
    --config src/rmcs_rl/config/bridge_test.yaml --node rl_bridge

# 4) 终端 A：executor（合成观测源 + 桥）。日志里应看到 layout_hash=<16 位 hex>
ros2 run rmcs_executor rmcs_executor --ros-args \
    --params-file src/rmcs_rl/config/bridge_test.yaml

# 5) 终端 B：策略进程。日志里的 layout_hash 必须与终端 A 的完全相同
ros2 run rmcs_rl policy_server --ros-args \
    --params-file src/rmcs_rl/config/bridge_test.yaml
```

看到 `[rl_bridge]: valid=1 (obs_seq=... model_id=... action_age=...)` 即链路通。
`valid=0` 的原因会直接打在日志里（见 [doc/deployment.md](doc/deployment.md) 的排障表）。

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

## P0 验收

```sh
# 端到端验收：合成策略 → 盖章 → 起 executor + policy_server → 断言 valid/失效语义/错配拒答/去类型化回归
bash src/rmcs_rl/tool/p0_acceptance.sh

# Python 工具链回归（不需要 executor）：词条语法自检 + gen/stamp/check 正例 + 6 个负例
bash src/rmcs_rl/tool/test_layout_contract.sh
```

`p0_acceptance.sh` 当前 **PASS=23 FAIL=0**（日志在 `/tmp/rmcs_rl_accept/logs/`）；
`test_layout_contract.sh` 实测 **PASS=25 FAIL=0**。

## 工具链

| 工具 | 用途 |
|---|---|
| `tool/rl_layout.py` | 词条语法 / 规范串 / `layout_hash` 的**单一真源**（`python3 tool/rl_layout.py` 跑自检） |
| `tool/stamp_layout_metadata.py` | 部署 YAML → 写进 ONNX `metadata_props`（`rmcs_obs_layout` / `rmcs_actions_layout` / `policy_layout_hash` …） |
| `tool/check_policy_contract.py` | `MODEL --config X [--node N] [--print-layout] [--expect-model-id H]`：校验 YAML ↔ 张量 ↔ metadata 三方一致 |
| `tool/gen_synthetic_policy.py` | `--from-config X -o M.onnx`：生成零动作合成模型（`obs[1,N] → actions[1,M]`；ONNX 后端强制 `ir_version=10`，因为 ORT 1.20 拒收 IR 13） |
| `tool/p0_probe.py` | 采集桥侧 topic（obs + 转发的 float 事实位）输出一行 JSON，供验收脚本断言 |
| `tool/test_layout_contract.sh` | Python 侧布局契约回归（含 6 个必须 FAIL 的负例） |
| `tool/p0_acceptance.sh` | P0 端到端验收 |
| `tool/install_rl_deps.sh` | `bash tool/install_rl_deps.sh local\|remote`：装 `libonnxruntime.so.1`（仅 `policy_server` 需要） |

## 现状与边界

- **P0 已完成**：桥、策略进程、消息包、合同指纹（v2）、工具链、端到端验收。
- **P1 未实现**：RMCS 侧 consumer（FSM / PREPARE / kp,kd / 限位 / NaN 让位）与 `RlController` → 桥的切换。
  目前没有组件写 `/wheel_leg/rl/enable`，也没有组件消费 `/wheel_leg/rl/action/*`，因此实车上
  `enable_default: false` → **桥恒 `valid=0`**，不会输出权威动作（这是刻意的：没人宣告权威就不许输出）。
- 桥**从不写电机 `control_*` 接口**：executor 禁止同名 output，电机控制权始终在 RMCS 侧消费组件手上，
  所以「RL 接管 / 退回传统控制」不需要仲裁组件（P1 的 consumer 用写 NaN 让位）。
- `RlController`（库 `rmcs_rl_legacy`）**仍然是唯一的整机可用路径**，P1 完成后删除；
  它带 ORT 进控制进程，配置格式与桥式完全不同，不要混配。
- 观测接口运行期热加不支持（配对只在启动做一次）；`contract_ok` 一旦因合同不符锁存为 false，
  当前实现**只能靠重启 executor 进程恢复**。

## 目录结构

```
rmcs_rl/
├── README.md                 # 本页（入口）
├── config/
│   ├── executor.yaml         # 实车配置模板（轮腿；复制到 rmcs_bringup/config/<robot>.yaml）
│   └── bridge_test.yaml      # 端到端测试夹具（合成观测源 + 桥 + policy_server）
├── doc/
│   ├── architecture.md       # 进程/组件、数据流、接口清单、valid 状态机
│   ├── bridge-design.md      # 重构定稿方案（权威设计）
│   ├── deployment.md         # 构建 → 模型 → 配置 → 运行 → 验证 → 交接
│   └── model-contract.md     # 模型张量合同与 metadata 清单
├── models/                   # 策略 ONNX（安装到 share/rmcs_rl/models/）
├── src/
│   ├── rl_bridge.cpp         # 桥
│   ├── rl_layout.hpp         # FNV-1a64 / layout_hash / model_id（与 tool/rl_layout.py 同构）
│   ├── policy_server.cpp     # 策略进程
│   ├── onnxruntime_inference.hpp
│   ├── rl_controller.cpp     # 遗留控制器（P1 删除）
│   └── testing/synthetic_robot.cpp
├── tool/                     # 见上表
├── plugins.xml               # rmcs_rl_bridge / rmcs_rl_testing / rmcs_rl_legacy 的 pluginlib 导出
└── CMakeLists.txt
```

## 相关

- 集成示例：`rmcs_bringup/config/wheel-leg-infantry-rl.yaml`（当前是遗留 `RlController` 配置）
- 训练侧：任何能导出 `obs[1,N] → actions[1,M]` 且带 `rmcs_obs_layout` / `rmcs_actions_layout` metadata 的
  ONNX 仓库（Isaac Lab / legged_gym / rsl_rl …）都可以接
