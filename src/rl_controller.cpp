#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>

#include "onnxruntime_inference.hpp"

namespace rmcs::rl {

class RlController
    : public rmcs_executor::Component
    , public rclcpp::Node {

    enum class State : std::uint8_t { kInit = 0, kIdle = 1, kPrepare = 2, kRl = 3 };

public:
    explicit RlController()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {

        joint_names_ = param_or("joint_names", std::vector<std::string>{});
        joint_base_path_ = param_or<std::string>("joint_base_path", "");
        position_pd_joints_ = param_or(
            "position_pd_joints", std::vector<std::int64_t>{});
        velocity_pd_joints_ = param_or(
            "velocity_pd_joints", std::vector<std::int64_t>{});

        if (joint_base_path_.empty())
            throw std::invalid_argument(
                "joint_base_path must be configured for this robot (e.g. /chassis, /wheel_leg)");

        position_group_angle_suffix_ =
            param_or<std::string>("position_group_angle_suffix", "/angle");
        position_group_velocity_suffix_ =
            param_or<std::string>("position_group_velocity_suffix", "/velocity");
        velocity_group_angle_suffix_ =
            param_or<std::string>("velocity_group_angle_suffix", "/angle");
        velocity_group_velocity_suffix_ =
            param_or<std::string>("velocity_group_velocity_suffix", "/velocity");

        const std::size_t dof = joint_names_.size();
        if (dof == 0 || dof > 32)
            throw std::invalid_argument(
                "joint_names must be configured (1..32 joints, training DOF order)");
        for (const auto idx : position_pd_joints_)
            if (idx < 0 || static_cast<std::size_t>(idx) >= dof)
                throw std::invalid_argument("position_pd_joints out of range");
        for (const auto idx : velocity_pd_joints_)
            if (idx < 0 || static_cast<std::size_t>(idx) >= dof)
                throw std::invalid_argument("velocity_pd_joints out of range");
        if (position_pd_joints_.empty() && velocity_pd_joints_.empty())
            throw std::invalid_argument(
                "position_pd_joints/velocity_pd_joints: at least one PD group must be configured");

        dof_ = dof;

        const auto configured_obs = require_param<std::int64_t>("rl_obs_size");
        const auto configured_act = require_param<std::int64_t>("rl_action_size");
        if (configured_obs <= 0 || configured_act <= 0)
            throw std::invalid_argument("rl_obs_size / rl_action_size must be positive");
        rl_obs_size_ = static_cast<std::size_t>(configured_obs);
        rl_action_size_ = static_cast<std::size_t>(configured_act);
        std::int64_t max_grouped_joint = -1;
        for (const auto idx : position_pd_joints_)
            max_grouped_joint = std::max(max_grouped_joint, idx);
        for (const auto idx : velocity_pd_joints_)
            max_grouped_joint = std::max(max_grouped_joint, idx);
        if (configured_act < max_grouped_joint + 1)
            throw std::invalid_argument(
                "rl_action_size=" + std::to_string(configured_act)
                + " too small: PD group joints need action slots up to index "
                + std::to_string(max_grouped_joint));
        const std::size_t layout_obs_size = 10 + 2 * dof + rl_action_size_;
        if (rl_obs_size_ != layout_obs_size)
            throw std::invalid_argument(
                "rl_obs_size=" + std::to_string(rl_obs_size_)
                + " inconsistent with this controller's observation layout length "
                  "(10 + 2*" + std::to_string(dof) + " + rl_action_size="
                + std::to_string(layout_obs_size) + "); see doc/model-contract.md");
        default_dof_pos_ = param_or("default_dof_pos", std::vector<double>(dof, 0.0));
        dof_pos_limits_lower_ = param_or("dof_pos_limits_lower", std::vector<double>(dof, -100.0));
        dof_pos_limits_upper_ = param_or("dof_pos_limits_upper", std::vector<double>(dof, 100.0));
        position_action_scale_ = param_or("position_action_scale", 1.0);
        velocity_action_scale_ = param_or("velocity_action_scale", 10.0);
        max_velocity_ = param_or("max_velocity", 100.0);
        position_kp_ = param_or("position_kp", 200.0);
        position_kd_ = param_or("position_kd", 4.0);
        velocity_kp_ = param_or("velocity_kp", 20.0);
        velocity_kd_ = param_or("velocity_kd", 0.5);
        position_torque_max_ = param_or("position_torque_max", 20.0);
        velocity_torque_max_ = param_or("velocity_torque_max", 6.0);

        obs_height_scale_ = param_or("obs_height_scale", 5.0);
        obs_ang_vel_scale_ = param_or("obs_ang_vel_scale", 0.5);
        obs_gravity_scale_ = param_or("obs_gravity_scale", 1.0);
        obs_dof_pos_scale_ = param_or("obs_dof_pos_scale", 1.0);
        obs_dof_vel_scale_ = param_or("obs_dof_vel_scale", 0.1);
        clip_observations_ = param_or("clip_observations", 100.0);
        clip_actions_ = param_or("clip_actions", 100.0);

        motion_linear_x_min_ = param_or("motion_linear_x_min", 0.0);
        motion_linear_x_max_ = param_or("motion_linear_x_max", 0.0);
        motion_angular_z_min_ = param_or("motion_angular_z_min", 0.0);
        motion_angular_z_max_ = param_or("motion_angular_z_max", 0.0);
        command_height_min_ = param_or("command_height_min", 0.0);
        command_height_max_ = param_or("command_height_max", 10.0);
        default_command_height_ = param_or("default_command_height", 0.0);

        auto_enter_rl_ = param_or("auto_enter_rl", false);
        prepare_dof_pos_ = param_or(
            "prepare_dof_pos",
            std::vector<double>(position_pd_joints_.size(), 0.0));
        if (!position_pd_joints_.empty()
            && prepare_dof_pos_.size() != position_pd_joints_.size())
            throw std::invalid_argument(
                "prepare_dof_pos size must match position_pd_joints size");
        prepare_kp_ = param_or("prepare_kp", 80.0);
        prepare_kd_ = param_or("prepare_kd", 2.0);
        prepare_max_velocity_ = param_or("prepare_max_velocity", 1.0);
        prepare_reach_threshold_ = param_or("prepare_reach_threshold", 0.02);

        rl_model_path_ = param_or<std::string>("rl_model_path", "");
        rl_inference_frequency_ = param_or("rl_inference_frequency", 100.0);
        rl_publish_network_io_ = param_or("rl_publish_network_io", false);

        joint_angle_input_ = std::make_unique<rmcs_executor::Component::InputInterface<double>[]>(dof);
        joint_velocity_input_ = std::make_unique<rmcs_executor::Component::InputInterface<double>[]>(dof);
        joint_control_torque_output_ = std::make_unique<rmcs_executor::Component::OutputInterface<double>[]>(dof);
        for (std::size_t i = 0; i < dof; ++i) {
            const std::string base = joint_base_path_ + "/" + joint_names_[i];
            const bool is_position_joint = std::find(
                position_pd_joints_.begin(), position_pd_joints_.end(),
                static_cast<std::int64_t>(i))
                != position_pd_joints_.end();
            register_input(
                base
                    + (is_position_joint ? position_group_angle_suffix_
                                         : velocity_group_angle_suffix_),
                joint_angle_input_[i]);
            register_input(
                base
                    + (is_position_joint ? position_group_velocity_suffix_
                                         : velocity_group_velocity_suffix_),
                joint_velocity_input_[i]);
            register_output(base + "/control_torque", joint_control_torque_output_[i], 0.0);
        }

        register_input(joint_base_path_ + "/imu/quaternion", imu_quaternion_);
        register_input(joint_base_path_ + "/imu/angular_velocity", imu_angular_velocity_);

        register_input(joint_base_path_ + "/command/vx", command_vx_, false);
        register_input(joint_base_path_ + "/command/yaw_rate", command_yaw_rate_, false);
        register_input(joint_base_path_ + "/command/height", command_height_, false);
        register_input(joint_base_path_ + "/command/state", command_state_, false);
        register_input(joint_base_path_ + "/reset_count", reset_count_, false);

        if (rl_publish_network_io_) {
            register_output(
                joint_base_path_ + "/rl/observation", rl_observation_output_, std::vector<double>{});
            register_output(
                joint_base_path_ + "/rl/action", rl_action_output_, std::vector<double>{});
            observation_publisher_ = create_publisher<std_msgs::msg::Float64MultiArray>(
                joint_base_path_ + "/rl/observation", 1);
            action_publisher_ = create_publisher<std_msgs::msg::Float64MultiArray>(
                joint_base_path_ + "/rl/action", 1);
            state_publisher_ = create_publisher<std_msgs::msg::Int32>(
                joint_base_path_ + "/rl/state", 1);
        }
        register_output(joint_base_path_ + "/rl/state", rl_state_output_, 0);

        action_.assign(rl_action_size_, 0.0);
        last_actions_.assign(rl_action_size_, 0.0);
        prepare_pos_.assign(position_pd_joints_.size(), 0.0);

        inference_ready_ = false;
        const std::string resolved_model_path = resolve_model_path_(rl_model_path_);
        if (!resolved_model_path.empty()) {
            inference_ready_ = inference_.load(OnnxRuntimeInference::Config{
                .model_path = resolved_model_path,
                .input_name = "obs",
                .output_name = "actions",
                .input_size = rl_obs_size_,
                .output_size = rl_action_size_,
            });
            if (inference_ready_) {
                RCLCPP_INFO(
                    get_logger(),
                    "RL policy loaded: %s ([1,%zu] -> [1,%zu], %.1f Hz, %zu joints)",
                    resolved_model_path.c_str(), rl_obs_size_, rl_action_size_,
                    rl_inference_frequency_, dof);
            } else {
                RCLCPP_ERROR(
                    get_logger(), "Failed to load RL policy '%s'; RL state unavailable",
                    resolved_model_path.c_str());
            }
        } else {
            RCLCPP_ERROR(get_logger(), "rl_model_path not set; RL state unavailable");
        }

        height_ = default_command_height_;
    }

    void update() override {
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::clamp(
            std::chrono::duration<double>(now - last_update_time_).count(), 0.0, 0.1);
        last_update_time_ = now;

        if (reset_count_.ready() && *reset_count_ != last_reset_count_) {
            last_reset_count_ = *reset_count_;
            reset_runtime_();
            return;
        }

        read_commands_();
        update_state_machine_();

        switch (state_) {
        case State::kInit:
        case State::kIdle:
            write_zero_outputs_();
            break;
        case State::kPrepare:
            prepare_step_(dt);
            break;
        case State::kRl:
            rl_step_();
            break;
        }

        *rl_state_output_ = static_cast<int>(state_);

        if (rl_publish_network_io_ && should_publish_io_()) {
            std_msgs::msg::Int32 state_msg;
            state_msg.data = static_cast<int>(state_);
            state_publisher_->publish(state_msg);
        }

        if (dof_ > 0) {
            std::ostringstream oss;
            oss << "joint_q_deg=[";
            for (std::size_t i = 0; i < dof_; ++i) {
                if (i > 0)
                    oss << ' ';
                oss << read_joint_angle_(i) * (180.0 / std::numbers::pi);
            }
            oss << "] (state=" << static_cast<int>(state_) << ")";
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000, "%s", oss.str().c_str());
        }
    }

private:
    template <typename T>
    T param_or(const std::string& name, const T& default_value) {
        T value;
        try {
            if (get_parameter(name, value))
                return value;
        } catch (const rclcpp::exceptions::InvalidParameterValueException&) {
        }
        RCLCPP_WARN(get_logger(), "Parameter '%s' not set, using default", name.c_str());
        return default_value;
    }

    template <typename T>
    T require_param(const std::string& name) {
        T value{};
        try {
            if (get_parameter(name, value))
                return value;
        } catch (const std::exception& error) {
            throw std::invalid_argument(
                "required parameter '" + name + "' is invalid: " + error.what());
        }
        throw std::invalid_argument(
            "missing required parameter '" + name + "' (no default; configure per robot)");
    }

    static std::string resolve_model_path_(const std::string& path) {
        if (path.empty() || path.front() == '/')
            return path;
        try {
            return ament_index_cpp::get_package_share_directory("rmcs_rl") + "/" + path;
        } catch (const std::exception&) {
            return path;
        }
    }

    void read_commands_() {
        const double vx = command_vx_.ready() ? *command_vx_ : 0.0;
        const double yaw = command_yaw_rate_.ready() ? *command_yaw_rate_ : 0.0;
        const double height = command_height_.ready() ? *command_height_ : default_command_height_;
        vx_ = std::clamp(vx, motion_linear_x_min_, motion_linear_x_max_);
        yaw_rate_ = std::clamp(yaw, motion_angular_z_min_, motion_angular_z_max_);
        height_ = std::clamp(height, command_height_min_, command_height_max_);
    }

    bool resolve_state_command_(int raw, State& target) const {
        switch (raw) {
        case 0: target = State::kInit; return true;
        case 1: target = State::kIdle; return true;
        case 2: target = State::kPrepare; return true;
        case 3:
            target = inference_ready_ ? State::kRl : State::kIdle;
            return inference_ready_;
        default: return false;
        }
    }

    void update_state_machine_() {
        if (command_state_.ready()) {
            const int raw = *command_state_;
            State target;
            if (resolve_state_command_(raw, target)) {
                if (target == State::kRl && state_ != State::kRl
                    && !(state_ == State::kPrepare && prepare_reached_)) {
                    RCLCPP_WARN_THROTTLE(
                        get_logger(), *get_clock(), 1000,
                        "Refusing RL: state=%d prepare_reached=%d joint_q=[%.3f %.3f %.3f %.3f] "
                        "(send 2 first, wait PREPARE done, then 3)",
                        static_cast<int>(state_), prepare_reached_ ? 1 : 0,
                        read_joint_angle_(0), read_joint_angle_(1), read_joint_angle_(2),
                        read_joint_angle_(3));
                    target = state_;
                }
                if (target != state_)
                    enter_state_(target);
            } else {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 1000, "Invalid state command %d (use 0..3)", raw);
            }
        }
        if (auto_enter_rl_ && inference_ready_ && state_ == State::kPrepare && prepare_reached_) {
            enter_state_(State::kRl);
            auto_enter_rl_ = false;
        }
    }

    void enter_state_(State target) {
        state_ = target;
        reset_policy_runtime_();
        if (target == State::kPrepare) {
            for (std::size_t k = 0; k < position_pd_joints_.size(); ++k)
                prepare_pos_[k] = read_joint_angle_(static_cast<std::size_t>(position_pd_joints_[k]));
            prepare_reached_ = false;
        }
        RCLCPP_INFO(get_logger(), "Entering state %d", static_cast<int>(target));
    }

    bool is_velocity_pd_joint_(std::size_t joint_index) const {
        return std::find(
                   velocity_pd_joints_.begin(), velocity_pd_joints_.end(),
                   static_cast<std::int64_t>(joint_index))
            != velocity_pd_joints_.end();
    }

    bool build_observation_(std::vector<double>& obs) {
        obs.assign(rl_obs_size_, 0.0);
        std::size_t k = 0;
        obs[k++] = vx_;
        obs[k++] = 0.0;
        obs[k++] = yaw_rate_;
        obs[k++] = height_ * obs_height_scale_;
        const Eigen::Vector3d ang_vel = *imu_angular_velocity_;
        for (std::size_t i = 0; i < 3; ++i)
            obs[k++] = ang_vel[i] * obs_ang_vel_scale_;
        const Eigen::Vector3d gravity = *imu_quaternion_ * Eigen::Vector3d(0.0, 0.0, -1.0);
        for (std::size_t i = 0; i < 3; ++i)
            obs[k++] = gravity[i] * obs_gravity_scale_;
        for (std::size_t i = 0; i < dof_; ++i) {
            obs[k++] = is_velocity_pd_joint_(i)
                ? 0.0
                : (read_joint_angle_(i) - default_dof_pos_[i]) * obs_dof_pos_scale_;
        }
        for (std::size_t i = 0; i < dof_; ++i)
            obs[k++] = read_joint_velocity_(i) * obs_dof_vel_scale_;
        for (std::size_t i = 0; i < rl_action_size_; ++i)
            obs[k++] = last_actions_[i];

        if (k != rl_obs_size_)
            return false;

        for (double& value : obs)
            value = std::clamp(value, -clip_observations_, clip_observations_);
        return std::all_of(obs.begin(), obs.end(), [](double v) { return std::isfinite(v); });
    }

    bool should_infer_() {
        const auto now = std::chrono::steady_clock::now();
        if (!last_inference_time_initialized_) {
            last_inference_time_ = now;
            last_inference_time_initialized_ = true;
            return true;
        }
        const double period = 1.0 / std::max(rl_inference_frequency_, 1.0);
        if (std::chrono::duration<double>(now - last_inference_time_).count() < period)
            return false;
        last_inference_time_ = now;
        return true;
    }

    bool should_publish_io_() {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double, std::milli>(now - last_io_publish_time_).count() < 10.0)
            return false;
        last_io_publish_time_ = now;
        return true;
    }

    void rl_step_() {
        if (!inference_ready_) {
            fail_safe_("policy session is not ready");
            return;
        }
        std::vector<double> obs;
        if (!build_observation_(obs)) {
            fail_safe_("policy observation size/validity mismatch");
            return;
        }
        if (rl_publish_network_io_)
            (*rl_observation_output_) = obs;

        if (should_infer_()) {
            std::vector<float> obs_f(obs.begin(), obs.end());
            std::vector<float> act_f(rl_action_size_, 0.0F);
            if (!inference_.run(obs_f, act_f)) {
                fail_safe_("ONNX Runtime rejected the policy input");
                return;
            }
            for (std::size_t i = 0; i < rl_action_size_; ++i)
                action_[i] = std::clamp(static_cast<double>(act_f[i]), -clip_actions_, clip_actions_);
            last_actions_ = action_;
            if (rl_publish_network_io_) {
                (*rl_action_output_) = action_;
                std_msgs::msg::Float64MultiArray obs_msg;
                obs_msg.data = obs;
                observation_publisher_->publish(obs_msg);
                std_msgs::msg::Float64MultiArray act_msg;
                act_msg.data = action_;
                action_publisher_->publish(act_msg);
            }
        }

        apply_pd_();
    }

    void apply_pd_() {
        std::vector<double> torques(dof_, 0.0);

        for (const auto idx : position_pd_joints_) {
            const std::size_t i = static_cast<std::size_t>(idx);
            const double target = std::clamp(
                position_action_scale_ * action_[i] + default_dof_pos_[i],
                dof_pos_limits_lower_[i], dof_pos_limits_upper_[i]);
            const double tau = position_kp_ * (target - read_joint_angle_(i))
                               - position_kd_ * read_joint_velocity_(i);
            torques[i] = std::clamp(tau, -position_torque_max_, position_torque_max_);
        }
        for (const auto idx : velocity_pd_joints_) {
            const std::size_t i = static_cast<std::size_t>(idx);
            const double vel_target = std::clamp(
                velocity_action_scale_ * action_[i], -max_velocity_, max_velocity_);
            const double tau = velocity_kp_ * (vel_target - read_joint_velocity_(i))
                               - velocity_kd_ * read_joint_velocity_(i);
            torques[i] = std::clamp(tau, -velocity_torque_max_, velocity_torque_max_);
        }

        write_outputs_(torques);
    }

    void prepare_step_(double dt) {
        std::vector<double> torques(dof_, 0.0);
        bool reached = true;
        for (std::size_t k = 0; k < position_pd_joints_.size(); ++k) {
            const std::size_t i = static_cast<std::size_t>(position_pd_joints_[k]);
            const double target = prepare_dof_pos_[k];
            const double step = prepare_max_velocity_ * dt;
            const double diff = target - prepare_pos_[k];
            if (std::abs(diff) > step) {
                prepare_pos_[k] += std::copysign(step, diff);
                reached = false;
            } else {
                prepare_pos_[k] = target;
            }
            const double tau = prepare_kp_ * (prepare_pos_[k] - read_joint_angle_(i))
                               - prepare_kd_ * read_joint_velocity_(i);
            torques[i] = std::clamp(tau, -position_torque_max_, position_torque_max_);
            if (std::abs(prepare_pos_[k] - read_joint_angle_(i)) > prepare_reach_threshold_)
                reached = false;
        }
        prepare_reached_ = reached;
        write_outputs_(torques);
    }

    void write_outputs_(const std::vector<double>& torques) {
        for (std::size_t i = 0; i < dof_; ++i) {
            const double value = std::isfinite(torques[i]) ? torques[i] : 0.0;
            *joint_control_torque_output_[i] = value;
        }
    }

    void write_zero_outputs_() { write_outputs_(std::vector<double>(dof_, 0.0)); }

    void fail_safe_(const std::string& reason) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "RL safety stop: %s", reason.c_str());
        write_zero_outputs_();
        reset_policy_runtime_();
        if (state_ != State::kIdle) {
            state_ = State::kIdle;
            RCLCPP_INFO(get_logger(), "Fail-safe: forced to IDLE");
        }
    }

    void reset_policy_runtime_() {
        std::fill(last_actions_.begin(), last_actions_.end(), 0.0);
        std::fill(action_.begin(), action_.end(), 0.0);
        std::fill(prepare_pos_.begin(), prepare_pos_.end(), 0.0);
        prepare_reached_ = false;
        last_inference_time_initialized_ = false;
    }

    void reset_runtime_() {
        write_zero_outputs_();
        reset_policy_runtime_();
        state_ = State::kIdle;
        vx_ = 0.0;
        yaw_rate_ = 0.0;
        height_ = default_command_height_;
        RCLCPP_INFO(get_logger(), "RMCS reset: outputs stopped, state cleared");
    }

    double read_joint_angle_(std::size_t index) const {
        return joint_angle_input_[index].ready() ? *joint_angle_input_[index] : 0.0;
    }
    double read_joint_velocity_(std::size_t index) const {
        return joint_velocity_input_[index].ready() ? *joint_velocity_input_[index] : 0.0;
    }

    std::vector<std::string> joint_names_;
    std::string joint_base_path_;
    std::vector<std::int64_t> position_pd_joints_;
    std::vector<std::int64_t> velocity_pd_joints_;
    std::string position_group_angle_suffix_ = "/angle";
    std::string position_group_velocity_suffix_ = "/velocity";
    std::string velocity_group_angle_suffix_ = "/angle";
    std::string velocity_group_velocity_suffix_ = "/velocity";
    std::size_t dof_ = 0;

    std::vector<double> default_dof_pos_;
    std::vector<double> dof_pos_limits_lower_;
    std::vector<double> dof_pos_limits_upper_;
    double position_action_scale_ = 1.0;
    double velocity_action_scale_ = 10.0;
    double max_velocity_ = 100.0;
    double position_kp_ = 200.0;
    double position_kd_ = 4.0;
    double velocity_kp_ = 20.0;
    double velocity_kd_ = 0.5;
    double position_torque_max_ = 20.0;
    double velocity_torque_max_ = 6.0;

    std::size_t rl_obs_size_ = 0;
    std::size_t rl_action_size_ = 0;
    double obs_height_scale_ = 5.0;
    double obs_ang_vel_scale_ = 0.5;
    double obs_gravity_scale_ = 1.0;
    double obs_dof_pos_scale_ = 1.0;
    double obs_dof_vel_scale_ = 0.1;
    double clip_observations_ = 100.0;
    double clip_actions_ = 100.0;

    double motion_linear_x_min_ = 0.0;
    double motion_linear_x_max_ = 0.0;
    double motion_angular_z_min_ = 0.0;
    double motion_angular_z_max_ = 0.0;
    double command_height_min_ = 0.0;
    double command_height_max_ = 10.0;
    double default_command_height_ = 0.0;

    bool auto_enter_rl_ = false;
    std::vector<double> prepare_dof_pos_;
    double prepare_kp_ = 80.0;
    double prepare_kd_ = 2.0;
    double prepare_max_velocity_ = 1.0;
    double prepare_reach_threshold_ = 0.02;

    std::string rl_model_path_;
    double rl_inference_frequency_ = 100.0;
    bool rl_publish_network_io_ = false;

    State state_ = State::kInit;
    double vx_ = 0.0;
    double yaw_rate_ = 0.0;
    double height_ = 0.0;
    std::vector<double> action_;
    std::vector<double> last_actions_;
    std::vector<double> prepare_pos_;
    bool prepare_reached_ = false;
    bool inference_ready_ = false;

    std::chrono::steady_clock::time_point last_update_time_{};
    std::chrono::steady_clock::time_point last_inference_time_{};
    std::chrono::steady_clock::time_point last_io_publish_time_{};
    bool last_inference_time_initialized_ = false;
    std::size_t last_reset_count_ = 0;

    OnnxRuntimeInference inference_;

    std::unique_ptr<rmcs_executor::Component::InputInterface<double>[]> joint_angle_input_;
    std::unique_ptr<rmcs_executor::Component::InputInterface<double>[]> joint_velocity_input_;
    std::unique_ptr<rmcs_executor::Component::OutputInterface<double>[]> joint_control_torque_output_;

    rmcs_executor::Component::InputInterface<Eigen::Quaterniond> imu_quaternion_;
    rmcs_executor::Component::InputInterface<Eigen::Vector3d> imu_angular_velocity_;

    rmcs_executor::Component::InputInterface<double> command_vx_;
    rmcs_executor::Component::InputInterface<double> command_yaw_rate_;
    rmcs_executor::Component::InputInterface<double> command_height_;
    rmcs_executor::Component::InputInterface<int> command_state_;
    rmcs_executor::Component::InputInterface<std::size_t> reset_count_;

    rmcs_executor::Component::OutputInterface<std::vector<double>> rl_observation_output_;
    rmcs_executor::Component::OutputInterface<std::vector<double>> rl_action_output_;
    rmcs_executor::Component::OutputInterface<int> rl_state_output_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr observation_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr action_publisher_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_publisher_;
};

} // namespace rmcs::rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs::rl::RlController, rmcs_executor::Component)
