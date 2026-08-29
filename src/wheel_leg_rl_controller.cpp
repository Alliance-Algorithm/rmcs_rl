#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>

#include "onnxruntime_inference.hpp"

namespace rmcs::rl {

/// 轮腿步兵 RL 控制器（推理段在 mini PC / RMCS 上位机）
///
/// 策略合同与 legged_gym 训练环境 `infantry_v4` 逐项对齐：
///   - DOF 序 [rf0, rf1, r_wheel, lf0, lf1, l_wheel]（训练 URDF 关节序）
///   - 观测 28 维：cmd3 | height_cmd(×5) | ang_vel3(×0.5) | gravity3 | joint_pos6(×1, 轮置零)
///     | joint_vel6(×0.1) | last_actions6
///   - 动作 6 维：腿位置目标 = leg_action_scale×a + default_dof_pos；轮速度目标 = wheel_velocity_scale×a
///   - PD：腿位置 PD（Kp/Kd），轮速度 PD（Kp/Kd），输出力矩
///   - 推理频率默认 50Hz（训练控制频率 sim dt 0.005 × decimation 4），PD 按执行频率每周期跑
///
/// FSM：INIT(0) / IDLE(1) / PREPARE(2) / RL(3)，经 /wheel_leg/command/state 切换；
/// 任意异常 failSafe 强制回 IDLE；RMCS reset（/wheel_leg/reset_count 变化）清空全部状态。
class WheelLegRLController
    : public rmcs_executor::Component
    , public rclcpp::Node {

    enum JointIndex : std::size_t {
        kRf0 = 0,
        kRf1 = 1,
        kRWheel = 2,
        kLf0 = 3,
        kLf1 = 4,
        kLWheel = 5,
    };

    static constexpr std::size_t kDofCount = 6;
    static constexpr std::size_t kLegCount = 4;
    static constexpr std::array<std::size_t, kLegCount> kLegIndices{kRf0, kRf1, kLf0, kLf1};
    static constexpr std::array<std::size_t, 2> kWheelIndices{kRWheel, kLWheel};
    static constexpr std::size_t kObservationSize = 28;
    static constexpr std::size_t kActionSize = 6;

    static constexpr std::array<const char*, kDofCount> kJointNames{
        "rf0", "rf1", "r_wheel", "lf0", "lf1", "l_wheel"};

    enum class State : std::uint8_t { kInit = 0, kIdle = 1, kPrepare = 2, kRl = 3 };

public:
    explicit WheelLegRLController()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {

        // ---- 策略合同参数 ----
        default_dof_pos_ = param_or("default_dof_pos", std::vector<double>{-0.5, -0.35, 0.0, 0.5, 0.35, 0.0});
        dof_pos_limits_lower_ =
            param_or("dof_pos_limits_lower", std::vector<double>{-100.0, -0.9, -100.0, -100.0, -0.65, -100.0});
        dof_pos_limits_upper_ =
            param_or("dof_pos_limits_upper", std::vector<double>{100.0, 0.65, 100.0, 100.0, 0.9, 100.0});
        leg_action_scale_ = param_or("leg_action_scale", 0.5);
        wheel_velocity_scale_ = param_or("wheel_velocity_scale", 10.0);
        max_wheel_vel_ = param_or("max_wheel_vel", 100.0);
        max_wheel_torque_ = param_or("max_wheel_torque", 6.0);
        leg_kp_ = param_or("leg_kp", 200.0);
        leg_kd_ = param_or("leg_kd", 4.0);
        wheel_kp_ = param_or("wheel_kp", 20.0);
        wheel_kd_ = param_or("wheel_kd", 0.5);
        leg_torque_max_ = param_or("leg_torque_max", 35.0);

        // ---- 观测缩放（与训练 normalization.obs_scales 一致）----
        obs_ang_vel_scale_ = param_or("obs_ang_vel_scale", 0.5);
        obs_gravity_scale_ = param_or("obs_gravity_scale", 1.0);
        obs_dof_pos_scale_ = param_or("obs_dof_pos_scale", 1.0);
        obs_dof_vel_scale_ = param_or("obs_dof_vel_scale", 0.1);
        obs_height_scale_ = param_or("obs_height_scale", 5.0);
        clip_observations_ = param_or("clip_observations", 100.0);
        clip_actions_ = param_or("clip_actions", 100.0);

        // ---- 指令范围 ----
        motion_linear_x_min_ = param_or("motion_linear_x_min", -2.5);
        motion_linear_x_max_ = param_or("motion_linear_x_max", 2.5);
        motion_angular_z_min_ = param_or("motion_angular_z_min", -3.0);
        motion_angular_z_max_ = param_or("motion_angular_z_max", 3.0);
        command_height_min_ = param_or("command_height_min", 0.20);
        command_height_max_ = param_or("command_height_max", 0.42);
        default_command_height_ = param_or("default_command_height", 0.22);

        // ---- FSM / PREPARE ----
        auto_enter_rl_ = param_or("auto_enter_rl", false);
        prepare_dof_pos_ = param_or(
            "prepare_dof_pos", std::vector<double>{-0.5, -0.35, 0.5, 0.35}); // 默认站立位（腿序 rf0,rf1,lf0,lf1）
        prepare_kp_ = param_or("prepare_kp", 80.0);
        prepare_kd_ = param_or("prepare_kd", 2.0);
        prepare_max_velocity_ = param_or("prepare_max_velocity", 1.0);
        prepare_reach_threshold_ = param_or("prepare_reach_threshold", 0.02);

        // ---- 模型与推理 ----
        rl_model_path_ = param_or<std::string>("rl_model_path", "");
        rl_inference_frequency_ = param_or("rl_inference_frequency", 50.0);
        rl_publish_network_io_ = param_or("rl_publish_network_io", false);

        // ---- 注册接口（构造体内直接可见）----
        for (std::size_t i = 0; i < kDofCount; ++i) {
            const std::string base = std::string("/wheel_leg/") + kJointNames[i];
            register_input(base + "/angle", joint_angle_input_[i]);
            register_input(base + "/velocity", joint_velocity_input_[i]);
            register_output(base + "/control_torque", joint_control_torque_output_[i], 0.0);
        }

        register_input("/wheel_leg/imu/quaternion", imu_quaternion_);
        register_input("/wheel_leg/imu/angular_velocity", imu_angular_velocity_);

        register_input("/wheel_leg/command/vx", command_vx_, false);
        register_input("/wheel_leg/command/yaw_rate", command_yaw_rate_, false);
        register_input("/wheel_leg/command/height", command_height_, false);
        register_input("/wheel_leg/command/state", command_state_, false);
        register_input("/wheel_leg/reset_count", reset_count_, false);

        if (rl_publish_network_io_) {
            register_output(
                "/wheel_leg/rl/observation", rl_observation_output_, std::array<double, kObservationSize>{});
            register_output("/wheel_leg/rl/action", rl_action_output_, std::array<double, kActionSize>{});
        }
        register_output("/wheel_leg/rl/state", rl_state_output_, 0);

        // ---- 加载策略 ----
        inference_ready_ = false;
        if (!rl_model_path_.empty()) {
            inference_ready_ = inference_.load(OnnxRuntimeInference::Config{
                .model_path = rl_model_path_,
                .input_name = "obs",
                .output_name = "actions",
                .input_size = kObservationSize,
                .output_size = kActionSize,
            });
            if (inference_ready_) {
                RCLCPP_INFO(
                    get_logger(), "RL policy loaded: %s ([1,%zu] -> [1,%zu], %.1f Hz)",
                    rl_model_path_.c_str(), kObservationSize, kActionSize, rl_inference_frequency_);
            } else {
                RCLCPP_ERROR(
                    get_logger(), "Failed to load RL policy '%s'; RL state unavailable",
                    rl_model_path_.c_str());
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

        // RMCS reset：计数变化 → 停输出、清状态、回 IDLE
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
    }

private:
    // ---- 参数辅助 ----
    template <typename T>
    T param_or(const std::string& name, const T& default_value) {
        T value;
        if (get_parameter(name, value))
            return value;
        RCLCPP_WARN(get_logger(), "Parameter '%s' not set, using default", name.c_str());
        return default_value;
    }

    // ---- 命令 ----
    void read_commands_() {
        const double vx = command_vx_.ready() ? *command_vx_ : 0.0;
        const double yaw = command_yaw_rate_.ready() ? *command_yaw_rate_ : 0.0;
        const double height = command_height_.ready() ? *command_height_ : default_command_height_;
        vx_ = std::clamp(vx, motion_linear_x_min_, motion_linear_x_max_);
        yaw_rate_ = std::clamp(yaw, motion_angular_z_min_, motion_angular_z_max_);
        height_ = std::clamp(height, command_height_min_, command_height_max_);
    }

    // ---- FSM ----
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
                if (target == State::kRl && !(state_ == State::kPrepare && prepare_reached_)) {
                    RCLCPP_WARN_THROTTLE(
                        get_logger(), *get_clock(), 1000,
                        "Refusing RL: robot must be prepared first (PREPARE -> RL)");
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
            for (std::size_t k = 0; k < kLegCount; ++k) {
                prepare_pos_[k] = read_joint_angle_(kLegIndices[k]);
            }
            prepare_reached_ = false;
        }
        RCLCPP_INFO(get_logger(), "Entering state %d", static_cast<int>(target));
    }

    // ---- 观测 ----
    bool build_observation_(std::array<double, kObservationSize>& obs) {
        // 0-2 指令（缩放 1.0，已在 read_commands_ 限幅）
        obs[0] = vx_;
        obs[1] = 0.0; // 横向速度指令恒 0（训练 lin_vel_y = 0）
        obs[2] = yaw_rate_;
        // 3 目标高度
        obs[3] = height_ * obs_height_scale_;
        // 4-6 机体角速度（机体系，IMU 陀螺）
        const Eigen::Vector3d ang_vel = *imu_angular_velocity_;
        for (std::size_t i = 0; i < 3; ++i)
            obs[4 + i] = ang_vel[i] * obs_ang_vel_scale_;
        // 7-9 重力投影（世界 [0,0,-1] 旋转到机体系；quat 为世界→机体，wxyz）
        const Eigen::Vector3d gravity = *imu_quaternion_ * Eigen::Vector3d(0.0, 0.0, -1.0);
        for (std::size_t i = 0; i < 3; ++i)
            obs[7 + i] = gravity[i] * obs_gravity_scale_;
        // 10-15 关节位置偏差（轮位置观测置零，与训练 mute_wheel_pos_obs 一致）
        for (std::size_t i = 0; i < kDofCount; ++i) {
            const bool wheel = (i == kRWheel || i == kLWheel);
            obs[10 + i] = wheel ? 0.0 : (read_joint_angle_(i) - default_dof_pos_[i]) * obs_dof_pos_scale_;
        }
        // 16-21 关节速度
        for (std::size_t i = 0; i < kDofCount; ++i)
            obs[16 + i] = read_joint_velocity_(i) * obs_dof_vel_scale_;
        // 22-27 上一帧动作（原始动作，与训练 obs 一致）
        for (std::size_t i = 0; i < kActionSize; ++i)
            obs[22 + i] = last_actions_[i];
        // 裁剪 + 有限性检查
        for (double& value : obs)
            value = std::clamp(value, -clip_observations_, clip_observations_);
        return std::all_of(obs.begin(), obs.end(), [](double v) { return std::isfinite(v); });
    }

    // ---- RL 推理（锁频）----
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

    void rl_step_() {
        if (!inference_ready_) {
            fail_safe_("policy session is not ready");
            return;
        }
        std::array<double, kObservationSize> obs;
        if (!build_observation_(obs)) {
            fail_safe_("policy observation is not a finite 28D vector");
            return;
        }
        if (rl_publish_network_io_)
            (*rl_observation_output_) = obs;

        if (should_infer_()) {
            std::array<float, kObservationSize> obs_f;
            for (std::size_t i = 0; i < kObservationSize; ++i)
                obs_f[i] = static_cast<float>(obs[i]);
            std::array<float, kActionSize> act_f;
            if (!inference_.run(obs_f, act_f)) {
                fail_safe_("ONNX Runtime rejected the policy input");
                return;
            }
            for (std::size_t i = 0; i < kActionSize; ++i)
                action_[i] = std::clamp(static_cast<double>(act_f[i]), -clip_actions_, clip_actions_);
            last_actions_ = action_;
            if (rl_publish_network_io_)
                (*rl_action_output_) = action_;
        }

        apply_pd_();
    }

    // ---- PD（与训练 _compute_torques 一致：腿位置 PD + 轮速度 PD，输出力矩）----
    void apply_pd_() {
        std::array<double, kDofCount> torques{};

        for (const std::size_t leg : kLegIndices) {
            const double target = std::clamp(
                leg_action_scale_ * action_[leg] + default_dof_pos_[leg],
                dof_pos_limits_lower_[leg], dof_pos_limits_upper_[leg]);
            const double tau = leg_kp_ * (target - read_joint_angle_(leg))
                               - leg_kd_ * read_joint_velocity_(leg);
            torques[leg] = std::clamp(tau, -leg_torque_max_, leg_torque_max_);
        }
        for (const std::size_t wheel : kWheelIndices) {
            const double vel_target = std::clamp(
                wheel_velocity_scale_ * action_[wheel], -max_wheel_vel_, max_wheel_vel_);
            const double tau = wheel_kp_ * (vel_target - read_joint_velocity_(wheel))
                               - wheel_kd_ * read_joint_velocity_(wheel);
            torques[wheel] = std::clamp(tau, -max_wheel_torque_, max_wheel_torque_);
        }

        write_outputs_(torques);
    }

    // ---- PREPARE：插值到站立预备位 ----
    void prepare_step_(double dt) {
        std::array<double, kDofCount> torques{};
        bool reached = true;
        for (std::size_t k = 0; k < kLegCount; ++k) {
            const std::size_t idx = kLegIndices[k];
            const double target = prepare_dof_pos_[k];
            const double step = prepare_max_velocity_ * dt;
            const double diff = target - prepare_pos_[k];
            if (std::abs(diff) > step) {
                prepare_pos_[k] += std::copysign(step, diff);
                reached = false;
            } else {
                prepare_pos_[k] = target;
            }
            const double tau = prepare_kp_ * (prepare_pos_[k] - read_joint_angle_(idx))
                               - prepare_kd_ * read_joint_velocity_(idx);
            torques[idx] = std::clamp(tau, -leg_torque_max_, leg_torque_max_);
            // 插值目标已到但关节实际未跟上（外力/摩擦）也算未到达
            if (std::abs(prepare_pos_[k] - read_joint_angle_(idx)) > prepare_reach_threshold_)
                reached = false;
        }
        prepare_reached_ = reached;
        write_outputs_(torques);
    }

    // ---- 输出 ----
    void write_outputs_(const std::array<double, kDofCount>& torques) {
        for (std::size_t i = 0; i < kDofCount; ++i) {
            const double value = std::isfinite(torques[i]) ? torques[i] : 0.0;
            *joint_control_torque_output_[i] = value;
        }
    }

    void write_zero_outputs_() {
        write_outputs_(std::array<double, kDofCount>{});
    }

    // ---- 安全 ----
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
        last_actions_.fill(0.0);
        action_.fill(0.0);
        prepare_pos_.fill(0.0);
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

    // ---- 读取 ----
    double read_joint_angle_(std::size_t index) const {
        return joint_angle_input_[index].ready() ? *joint_angle_input_[index] : 0.0;
    }
    double read_joint_velocity_(std::size_t index) const {
        return joint_velocity_input_[index].ready() ? *joint_velocity_input_[index] : 0.0;
    }

    // ---- 参数 ----
    std::vector<double> default_dof_pos_;
    std::vector<double> dof_pos_limits_lower_;
    std::vector<double> dof_pos_limits_upper_;
    double leg_action_scale_ = 0.5;
    double wheel_velocity_scale_ = 10.0;
    double max_wheel_vel_ = 100.0;
    double max_wheel_torque_ = 6.0;
    double leg_kp_ = 200.0;
    double leg_kd_ = 4.0;
    double wheel_kp_ = 20.0;
    double wheel_kd_ = 0.5;
    double leg_torque_max_ = 35.0;

    double obs_ang_vel_scale_ = 0.5;
    double obs_gravity_scale_ = 1.0;
    double obs_dof_pos_scale_ = 1.0;
    double obs_dof_vel_scale_ = 0.1;
    double obs_height_scale_ = 5.0;
    double clip_observations_ = 100.0;
    double clip_actions_ = 100.0;

    double motion_linear_x_min_ = -2.5;
    double motion_linear_x_max_ = 2.5;
    double motion_angular_z_min_ = -3.0;
    double motion_angular_z_max_ = 3.0;
    double command_height_min_ = 0.20;
    double command_height_max_ = 0.42;
    double default_command_height_ = 0.22;

    bool auto_enter_rl_ = false;
    std::vector<double> prepare_dof_pos_{0.0, 0.0, 0.0, 0.0};
    double prepare_kp_ = 80.0;
    double prepare_kd_ = 2.0;
    double prepare_max_velocity_ = 1.0;
    double prepare_reach_threshold_ = 0.02;

    std::string rl_model_path_;
    double rl_inference_frequency_ = 50.0;
    bool rl_publish_network_io_ = false;

    // ---- 运行时状态 ----
    State state_ = State::kInit;
    double vx_ = 0.0;
    double yaw_rate_ = 0.0;
    double height_ = 0.22;
    std::array<double, kActionSize> action_{};
    std::array<double, kActionSize> last_actions_{};
    std::array<double, kLegCount> prepare_pos_{};
    bool prepare_reached_ = false;
    bool inference_ready_ = false;

    std::chrono::steady_clock::time_point last_update_time_{};
    std::chrono::steady_clock::time_point last_inference_time_{};
    bool last_inference_time_initialized_ = false;
    std::size_t last_reset_count_ = 0;

    OnnxRuntimeInference inference_;

    // ---- 接口 ----
    std::array<rmcs_executor::Component::InputInterface<double>, kDofCount> joint_angle_input_;
    std::array<rmcs_executor::Component::InputInterface<double>, kDofCount> joint_velocity_input_;
    std::array<rmcs_executor::Component::OutputInterface<double>, kDofCount>
        joint_control_torque_output_;

    rmcs_executor::Component::InputInterface<Eigen::Quaterniond> imu_quaternion_;
    rmcs_executor::Component::InputInterface<Eigen::Vector3d> imu_angular_velocity_;

    rmcs_executor::Component::InputInterface<double> command_vx_;
    rmcs_executor::Component::InputInterface<double> command_yaw_rate_;
    rmcs_executor::Component::InputInterface<double> command_height_;
    rmcs_executor::Component::InputInterface<int> command_state_;
    rmcs_executor::Component::InputInterface<std::size_t> reset_count_;

    rmcs_executor::Component::OutputInterface<std::array<double, kObservationSize>>
        rl_observation_output_;
    rmcs_executor::Component::OutputInterface<std::array<double, kActionSize>> rl_action_output_;
    rmcs_executor::Component::OutputInterface<int> rl_state_output_;
};

} // namespace rmcs::rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs::rl::WheelLegRLController, rmcs_executor::Component)
