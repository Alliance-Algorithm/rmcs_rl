# 管道反馈

`rmcs_rl::PipeFeedback` 在正常执行器中提供配置化的仿真反馈。主组件读取 stdin 并发布内部接口，`<实例名>_command` partner 读取控制结果并写入 stdout，执行器按真实依赖图安排顺序。组件不加载硬件，也不自行定义执行器。

执行器使用 `clock: stepped`，按固定周期推进 `/predefined/timestamp`，由读取管道的阻塞操作决定实际节拍。`RlBridge` 使用这一时间戳提交观测并判断动作年龄；异步推理仍由独立 worker 完成，结果到达的 tick 取决于实际推理耗时。执行器必须在组件 `before_updating()` 之前发布有效的 `update_rate`。

```yaml
rmcs_executor:
  ros__parameters:
    update_rate: 1000.0
    clock: stepped
    components:
      - rmcs_rl::PipeFeedback -> pipe_feedback
      # 添加只消费下面反馈接口的仿真控制组件。
pipe_feedback:
  ros__parameters:
    outputs:
      order: [position]
      position: {path: /simulation/position, type: double, initial: 0.0}
    inputs: [/simulation/position]
```

`outputs.order` 必须非空，每项指定 `path`、`type` 和 `initial`。输出类型与 JSON 表示如下；`inputs` 根据生产者的实际类型绑定，支持同一组类型，缺少生产者、事件接口或不支持的类型会在启动时报错。

| 类型 | C++ 类型 | JSON |
|---|---|---|
| `double` | `double` | 数值 |
| `bool` | `bool` | 布尔 |
| `int` | `int` | 整数 |
| `uint64` | `std::uint64_t`，本平台也包括 `std::size_t` | 非负整数 |
| `vector3` | `Eigen::Vector3d` | `[x, y, z]` |
| `direction_vector` | `BaseLink::DirectionVector` | `[x, y, z]` |
| `quaternion_wxyz` | `Eigen::Quaterniond` | `[w, x, y, z]` |
| `chassis_mode` | `rmcs_msgs::ChassisMode` | 整数 |

启动时输出 `{"ready":true}`。之后每行输入一个对象：键为配置的接口路径，值覆盖对应反馈；省略的接口保留上一值。`ticks` 是可选的正整数，缺省为 1。例：

```json
{"ticks":5,"/simulation/position":0.1}
```

该行反馈连续执行 5 tick。最后一个 tick 中，command partner 将 `inputs` 列出的接口输出为一行 JSON，并刷新 stdout。未知输出路径、错误向量长度或非法计数会使执行器报错退出。NaN 控制量按 JSON 库规则输出为 `null`，由车型适配层解释。

stdin 到达 EOF 时调用 `rclcpp::shutdown()`，不额外输出上一批结果。调用方应保留 stderr 以诊断启动或运行错误，并在退出时关闭 stdin。

轮腿五组件配置见 `config/simulation/wheel-leg-pipe.yaml`。其中模型和合同仍须替换为通过验收的同一份正式导出；未提供匹配模型时启动检查会失败。实车 bringup 继续使用硬件反馈配置。
