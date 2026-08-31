/// RL 部署管线测试桩（无硬件）：模拟 deformable 底盘硬件接口
///
/// 与真实硬件同构的 Status/Command 双组件架构（create_partner_component），
/// 以打破 rmcs_executor 接口图的环（状态输出与命令输入分属 partner 两侧）：
///
///   Status（rl_fake_plant）：输出关节 physical_angle/physical_velocity、IMU 姿态
///   Command（rl_fake_plant_command，partner）：输入 control_torque 与 rl/observation、
///     rl/action、rl/state，发布 ROS topic 供无硬件观测
///
/// 简单动力学（可选，默认开）：q̈ = tau/J - damp·q̇，可观察 PREPARE/RL 状态下关节运动。
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>

namespace rmcs::rl {

class RlFakePlant
    : public rmcs_executor::Component
    , public rclcpp::Node {

public:
    /// 命令侧（partner）：消费 RL 输出，转发为 ROS topic
    class Command
        : public rmcs_executor::Component
        , public rclcpp::Node {

    public:
        Command(const std::string& base_path, const std::vector<std::string>& joint_names)
            : Node{
                  get_component_name(),
                  rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)}
            , dof_{joint_names.size()} {
            joint_torque_input_ = std::make_unique<rmcs_executor::Component::InputInterface<double>[]>(dof_);
            for (std::size_t i = 0; i < dof_; ++i) {
                const std::string base = base_path + "/" + joint_names[i];
                register_input(base + "/control_torque", joint_torque_input_[i], false);
            }
            register_input(base_path + "/rl/observation", rl_observation_input_, false);
            register_input(base_path + "/rl/action", rl_action_input_, false);
            register_input(base_path + "/rl/state", rl_state_input_, false);

            torque_publisher_ = create_publisher<std_msgs::msg::Float64MultiArray>(
                base_path + "/fake/torques", 1);
            obs_publisher_ = create_publisher<std_msgs::msg::Float64MultiArray>(
                base_path + "/fake/observation", 1);
            action_publisher_ = create_publisher<std_msgs::msg::Float64MultiArray>(
                base_path + "/fake/action", 1);
            state_publisher_ = create_publisher<std_msgs::msg::Int32>(
                base_path + "/fake/state", 1);
        }

        void update() override {
            // 限频 20Hz 发布，避免 1000Hz 更新周期下 topic 洪泛
            const auto now = std::chrono::steady_clock::now();
            if (now - last_publish_ <= std::chrono::milliseconds(50))
                return;
            last_publish_ = now;
            auto torques = std_msgs::msg::Float64MultiArray();
            auto obs = std_msgs::msg::Float64MultiArray();
            auto act = std_msgs::msg::Float64MultiArray();
            for (std::size_t i = 0; i < dof_; ++i)
                torques.data.push_back(
                    joint_torque_input_[i].ready() ? *joint_torque_input_[i] : 0.0);
            if (rl_observation_input_.ready())
                obs.data = *rl_observation_input_;
            if (rl_action_input_.ready())
                act.data = *rl_action_input_;
            torque_publisher_->publish(torques);
            obs_publisher_->publish(obs);
            action_publisher_->publish(act);
            auto state = std_msgs::msg::Int32();
            state.data = rl_state_input_.ready() ? *rl_state_input_ : -1;
            state_publisher_->publish(state);
        }

        // 供 Status 侧动力学读取
        std::unique_ptr<rmcs_executor::Component::InputInterface<double>[]> joint_torque_input_;
        rmcs_executor::Component::InputInterface<std::vector<double>> rl_observation_input_;
        rmcs_executor::Component::InputInterface<std::vector<double>> rl_action_input_;
        rmcs_executor::Component::InputInterface<int> rl_state_input_;

    private:
        std::size_t dof_ = 0;
        std::chrono::steady_clock::time_point last_publish_{};
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr obs_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr action_publisher_;
        rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_publisher_;
    };

public:
    explicit RlFakePlant()
        : Node{
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)} {
        joint_names_ = param_or(
            "joint_names",
            std::vector<std::string>{
                "left_front_joint", "left_back_joint", "right_back_joint", "right_front_joint"});
        joint_base_path_ = param_or<std::string>("joint_base_path", "/chassis");
        const std::size_t dof = joint_names_.size();
        if (dof == 0 || dof > 32)
            throw std::invalid_argument("joint_names must contain 1..32 joints");

        angle_suffix_ = param_or<std::string>("angle_suffix", "/physical_angle");
        velocity_suffix_ = param_or<std::string>("velocity_suffix", "/physical_velocity");

        const std::vector<double> init_angles =
            param_or("fake_joint_angles", std::vector<double>(dof, 0.0));
        joint_angle_.assign(dof, 0.0);
        joint_velocity_.assign(dof, 0.0);
        for (std::size_t i = 0; i < dof && i < init_angles.size(); ++i)
            joint_angle_[i] = init_angles[i];

        dynamics_enabled_ = param_or("fake_dynamics", true);
        inertia_ = param_or("fake_inertia", 0.05);
        damping_ = param_or("fake_damping", 2.0);
        imu_roll_rate_ = param_or("fake_imu_roll_rate", 0.0);   // rad/s
        imu_pitch_rate_ = param_or("fake_imu_pitch_rate", 0.0); // rad/s

        command_ = create_partner_component<Command>(
            get_component_name() + "_command", joint_base_path_, joint_names_);

        joint_angle_output_ = std::make_unique<rmcs_executor::Component::OutputInterface<double>[]>(dof);
        joint_velocity_output_ = std::make_unique<rmcs_executor::Component::OutputInterface<double>[]>(dof);
        for (std::size_t i = 0; i < dof; ++i) {
            const std::string base = joint_base_path_ + "/" + joint_names_[i];
            register_output(base + angle_suffix_, joint_angle_output_[i], 0.0);
            register_output(base + velocity_suffix_, joint_velocity_output_[i], 0.0);
        }
        register_output(
            joint_base_path_ + "/imu/quaternion", imu_quaternion_output_,
            Eigen::Quaterniond::Identity());
        register_output(
            joint_base_path_ + "/imu/angular_velocity", imu_angular_velocity_output_,
            Eigen::Vector3d::Zero());
    }

    void update() override {
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::clamp(
            std::chrono::duration<double>(now - last_update_).count(), 0.0, 0.05);
        last_update_ = now;

        // simple joint dynamics: q̈ = tau/J - damp·q̇（力矩来自 Command 侧输入）
        if (dynamics_enabled_) {
            for (std::size_t i = 0; i < joint_angle_.size(); ++i) {
                const auto& torque_input = command_->joint_torque_input_[i];
                const double tau = torque_input.ready() ? *torque_input : 0.0;
                joint_velocity_[i] += (tau / inertia_ - damping_ * joint_velocity_[i]) * dt;
                joint_angle_[i] += joint_velocity_[i] * dt;
                joint_angle_[i] = std::clamp(joint_angle_[i], 0.0, 1.05);
            }
        }
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "fake q=[%.3f %.3f %.3f %.3f] v=[%.1f %.1f %.1f %.1f] tau0=%.3f ready=%d",
            joint_angle_[0], joint_angle_[1], joint_angle_[2], joint_angle_[3],
            joint_velocity_[0], joint_velocity_[1], joint_velocity_[2], joint_velocity_[3],
            command_->joint_torque_input_[0].ready() ? *command_->joint_torque_input_[0] : 0.0,
            command_->joint_torque_input_[0].ready() ? 1 : 0);
        for (std::size_t i = 0; i < joint_angle_.size(); ++i) {
            *joint_angle_output_[i] = joint_angle_[i];
            *joint_velocity_output_[i] = joint_velocity_[i];
        }

        // IMU：水平姿态 + 可配置角速度（验证观测的角速度段）
        *imu_quaternion_output_ = Eigen::Quaterniond::Identity();
        *imu_angular_velocity_output_ = Eigen::Vector3d(imu_roll_rate_, imu_pitch_rate_, 0.0);
    }

private:
    template <typename T>
    T param_or(const std::string& name, const T& default_value) {
        T value;
        try {
            if (get_parameter(name, value))
                return value;
        } catch (const rclcpp::exceptions::InvalidParameterValueException&) {
            // rcl 参数文件中的空列表会以 PARAMETER_NOT_SET 落盘，get_parameter 抛异常；
            // 视为未设置，回落默认值。
        }
        RCLCPP_WARN(get_logger(), "Parameter '%s' not set, using default", name.c_str());
        return default_value;
    }

    std::vector<std::string> joint_names_;
    std::string joint_base_path_ = "/chassis";
    std::string angle_suffix_ = "/physical_angle";
    std::string velocity_suffix_ = "/physical_velocity";
    bool dynamics_enabled_ = true;
    double inertia_ = 0.05;
    double damping_ = 2.0;
    double imu_roll_rate_ = 0.0;
    double imu_pitch_rate_ = 0.0;

    std::vector<double> joint_angle_;
    std::vector<double> joint_velocity_;

    std::shared_ptr<Command> command_;
    std::unique_ptr<rmcs_executor::Component::OutputInterface<double>[]> joint_angle_output_;
    std::unique_ptr<rmcs_executor::Component::OutputInterface<double>[]> joint_velocity_output_;
    rmcs_executor::Component::OutputInterface<Eigen::Quaterniond> imu_quaternion_output_;
    rmcs_executor::Component::OutputInterface<Eigen::Vector3d> imu_angular_velocity_output_;

    std::chrono::steady_clock::time_point last_update_{};
};

} // namespace rmcs::rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs::rl::RlFakePlant, rmcs_executor::Component)
