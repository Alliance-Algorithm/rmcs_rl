/// RL 调试指令源组件（验证/调试专用，正式控制链不应使用）
///
/// 订阅 ROS topic 写入 RlController 的命令接口：
///   {command_base}/debug/command   std_msgs::msg::Float64MultiArray [vx, yaw_rate, height, state]
///   {command_base}/debug/reset     std_msgs::msg::Bool（true 沿 → reset_count 递增）
///
/// 输出：
///   {command_base}/command/{vx,yaw_rate,height,state}
///   {command_base}/reset_count（std::size_t，变化触发 RlController 干净复位）
#include <atomic>
#include <cstddef>
#include <string>

#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace rmcs::rl {

class RlDebugCommand
    : public rmcs_executor::Component
    , public rclcpp::Node {

public:
    explicit RlDebugCommand()
        : Node{
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {
        command_base_ = param_or<std::string>("command_base", "/wheel_leg");

        register_output(command_base_ + "/command/vx", command_vx_output_, 0.0);
        register_output(command_base_ + "/command/yaw_rate", command_yaw_rate_output_, 0.0);
        register_output(command_base_ + "/command/height", command_height_output_, 0.0);
        register_output(command_base_ + "/command/state", command_state_output_, 0);
        register_output(command_base_ + "/reset_count", reset_count_output_, std::size_t{0});

        command_subscription_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            command_base_ + "/debug/command", 1,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                const auto& data = msg->data;
                if (data.size() > 0)
                    command_vx_.store(data[0], std::memory_order::relaxed);
                if (data.size() > 1)
                    command_yaw_rate_.store(data[1], std::memory_order::relaxed);
                if (data.size() > 2)
                    command_height_.store(data[2], std::memory_order::relaxed);
                if (data.size() > 3)
                    command_state_.store(static_cast<int>(data[3]), std::memory_order::relaxed);
            });
        reset_subscription_ = create_subscription<std_msgs::msg::Bool>(
            command_base_ + "/debug/reset", 1,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                if (msg->data)
                    reset_count_.fetch_add(1, std::memory_order::relaxed);
            });
    }

    void update() override {
        *command_vx_output_ = command_vx_.load(std::memory_order::relaxed);
        *command_yaw_rate_output_ = command_yaw_rate_.load(std::memory_order::relaxed);
        *command_height_output_ = command_height_.load(std::memory_order::relaxed);
        *command_state_output_ = command_state_.load(std::memory_order::relaxed);
        *reset_count_output_ = reset_count_.load(std::memory_order::relaxed);
    }

private:
    template <typename T>
    T param_or(const std::string& name, const T& default_value) {
        T value;
        if (get_parameter(name, value))
            return value;
        RCLCPP_WARN(get_logger(), "Parameter '%s' not set, using default", name.c_str());
        return default_value;
    }

    std::string command_base_ = "/wheel_leg";
    std::atomic<double> command_vx_{0.0};
    std::atomic<double> command_yaw_rate_{0.0};
    std::atomic<double> command_height_{0.0};
    std::atomic<int> command_state_{0};
    std::atomic<std::size_t> reset_count_{0};

    rmcs_executor::Component::OutputInterface<double> command_vx_output_;
    rmcs_executor::Component::OutputInterface<double> command_yaw_rate_output_;
    rmcs_executor::Component::OutputInterface<double> command_height_output_;
    rmcs_executor::Component::OutputInterface<int> command_state_output_;
    rmcs_executor::Component::OutputInterface<std::size_t> reset_count_output_;

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr command_subscription_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reset_subscription_;
};

} // namespace rmcs::rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs::rl::RlDebugCommand, rmcs_executor::Component)
