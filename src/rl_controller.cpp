#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>

#include "onnxruntime_inference.hpp"

namespace rmcs::rl {

/// 通用 RL 策略部署控制器（机器人无关）
///
/// 策略合同（训练导出模型必须满足，tensor 名 "obs"/"actions"，float32）：
///   obs    = cmd3 | height_cmd(×obs_height_scale) | ang_vel3(×obs_ang_vel_scale)
///            | gravity3(×obs_gravity_scale)
///            | joint_pos(N)(×obs_dof_pos_scale，velocity_pd 关节位置观测置零)
///            | joint_vel(N)(×obs_dof_vel_scale) | last_actions(N)
///   actions = N 维，与 joint_names 序一一对应（N = 关节数）
///
/// 低层控制（与训练环境同构）：
///   position_pd_joints：pos_target = position_action_scale×a + default_dof_pos，位置 PD → 力矩
///   velocity_pd_joints：vel_target = velocity_action_scale×a，速度 PD → 力矩
/// 默认参数对应 legged_gym `infantry_v4`（6 关节轮腿）训练合同。
///
/// FSM：INIT(0) / IDLE(1) / PREPARE(2) / RL(3)，经 {joint_base_path}/command/state 切换；
/// 异常 failSafe 强制回 IDLE；RMCS reset（{joint_base_path}/reset_count 变化）清空全部状态。
class RlController
    : public rmcs_executor::Component
    , public rclcpp::Node {

    enum class State : std::uint8_t { kInit = 0, kIdle = 1, kPrepare = 2, kRl = 3 };

public:
    explicit RlController()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {

        // ---- 机器人描述（通用参数化）----
        joint_names_ = param_or(
            "joint_names",
            std::vector<std::string>{"rf0", "rf1", "r_wheel", "lf0", "lf1", "l_wheel"});
        joint_base_path_ = param_or<std::string>("joint_base_path", "/wheel_leg");
        position_pd_joints_ = param_or(
            "position_pd_joints", std::vector<std::int64_t>{0, 1, 3, 4});
        velocity_pd_joints_ = param_or(
            "velocity_pd_joints", std::vector<std::int64_t>{2, 5});

        const std::size_t dof = joint_names_.size();
        if (dof == 0 || dof > 32)
            throw std::invalid_argument("joint_names must contain 1..32 joints");
        for (const auto idx : position_pd_joints_)
            if (idx < 0 || static_cast<std::size_t>(idx) >= dof)
                throw std::invalid_argument("position_pd_joints out of range");
        for (const auto idx : velocity_pd_joints_)
            if (idx < 0 || static_cast<std::size_t>(idx) >= dof)
                throw std::invalid_argument("velocity_pd_joints out of range");

        dof_ = dof;
        const std::size_t default_obs_size = 10 + 3 * dof;
        const std::size_t default_action_size = dof;

        // ---- 策略合同参数 ----
        rl_obs_size_ = static_cast<std::size_t>(param_or("rl_obs_size", static_cast<std::int64_t>(default_obs_size)));
        rl_action_size_ = static_cast<std::size_t>(param_or("rl_action_size", static_cast<std::int64_t>(default_action_size)));
        default_dof_pos_ = param_or("default_dof_pos", std::vector<double>(dof, 0.0));
        dof_pos_limits_lower_ = param_or("dof_pos_limits_lower", std::vector<double>(dof, -100.0));
        dof_pos_limits_upper_ = param_or("dof_pos_limits_upper", std::vector<double>(dof, 100.0));
        position_action_scale_ = param_or("position_action_scale", 0.5);
        velocity_action_scale_ = param_or("velocity_action_scale", 10.0);
        max_velocity_ = param_or("max_velocity", 100.0);
        position_kp_ = param_or("position_kp", 200.0);
        position_kd_ = param_or("position_kd", 4.0);
        velocity_kp_ = param_or("velocity_kp", 20.0);
        velocity_kd_ = param_or("velocity_kd", 0.5);
        position_torque_max_ = param_or("position_torque_max", 35.0);
        velocity_torque_max_ = param_or("velocity_torque_max", 6.0);

        // ---- 观测缩放（训练 normalization 对齐）----
        obs_height_scale_ = param_or("obs_height_scale", 5.0);
        obs_ang_vel_scale_ = param_or("obs_ang_vel_scale", 0.5);
        obs_gravity_scale_ = param_or("obs_gravity_scale", 1.0);
        obs_dof_pos_scale_ = param_or("obs_dof_pos_scale", 1.0);
        obs_dof_vel_scale_ = param_or("obs_dof_vel_scale", 0.1);
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
            "prepare_dof_pos", std::vector<double>{-0.5, -0.35, 0.5, 0.35}); // 对应 position_pd_joints 序
        prepare_kp_ = param_or("prepare_kp", 80.0);
        prepare_kd_ = param_or("prepare_kd", 2.0);
        prepare_max_velocity_ = param_or("prepare_max_velocity", 1.0);
        prepare_reach_threshold_ = param_or("prepare_reach_threshold", 0.02);

        // ---- 模型与推理 ----
        rl_model_path_ = param_or<std::string>("rl_model_path", "");
        rl_inference_frequency_ = param_or("rl_inference_frequency", 50.0);
        rl_publish_network_io_ = param_or("rl_publish_network_io", false);

        // ---- 接口（按 joint_names 动态注册）----
        joint_angle_input_ = std::make_unique<rmcs_executor::Component::InputInterface<double>[]>(dof);
        joint_velocity_input_ = std::make_unique<rmcs_executor::Component::InputInterface<double>[]>(dof);
        joint_control_torque_output_ = std::make_unique<rmcs_executor::Component::OutputInterface<double>[]>(dof);
        for (std::size_t i = 0; i < dof; ++i) {
            const std::string base = joint_base_path_ + "/" + joint_names_[i];
            register_input(base + "/angle", joint_angle_input_[i]);
            register_input(base + "/velocity", joint_velocity_input_[i]);
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
        }
        register_output(joint_base_path_ + "/rl/state", rl_state_output_, 0);

        // ---- 运行时缓冲 ----
        action_.assign(rl_action_size_, 0.0);
        last_actions_.assign(rl_action_size_, 0.0);
        prepare_pos_.assign(position_pd_joints_.size(), 0.0);

        // ---- 加载策略 ----
        // rl_model_path 支持相对路径：相对本包 share 目录解析
        // （如 "models/policy.onnx" → <install>/share/rmcs_rl/models/policy.onnx），
        // 随 sync-remote 同步，开发/运行环境一致。
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

    /// 相对路径 → 本包 share 目录下；绝对路径原样返回；空返回空。
    static std::string resolve_model_path_(const std::string& path) {
        if (path.empty() || path.front() == '/')
            return path;
        try {
            return ament_index_cpp::get_package_share_directory("rmcs_rl") + "/" + path;
        } catch (const std::exception&) {
            return path;
        }
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
            for (std::size_t k = 0; k < position_pd_joints_.size(); ++k)
                prepare_pos_[k] = read_joint_angle_(static_cast<std::size_t>(position_pd_joints_[k]));
            prepare_reached_ = false;
        }
        RCLCPP_INFO(get_logger(), "Entering state %d", static_cast<int>(target));
    }

    // ---- 观测 ----
    bool is_velocity_pd_joint_(std::size_t joint_index) const {
        return std::find(
                   velocity_pd_joints_.begin(), velocity_pd_joints_.end(),
                   static_cast<std::int64_t>(joint_index))
            != velocity_pd_joints_.end();
    }

    bool build_observation_(std::vector<double>& obs) {
        obs.assign(rl_obs_size_, 0.0);
        std::size_t k = 0;
        // 0-2 指令（缩放 1.0，已限幅）
        obs[k++] = vx_;
        obs[k++] = 0.0; // 横向速度指令恒 0（训练 lin_vel_y = 0）
        obs[k++] = yaw_rate_;
        // 3 目标高度
        obs[k++] = height_ * obs_height_scale_;
        // 4-6 机体角速度（机体系，IMU 陀螺）
        const Eigen::Vector3d ang_vel = *imu_angular_velocity_;
        for (std::size_t i = 0; i < 3; ++i)
            obs[k++] = ang_vel[i] * obs_ang_vel_scale_;
        // 7-9 重力投影（世界 [0,0,-1] 旋转到机体系；quat 世界→机体，wxyz）
        const Eigen::Vector3d gravity = *imu_quaternion_ * Eigen::Vector3d(0.0, 0.0, -1.0);
        for (std::size_t i = 0; i < 3; ++i)
            obs[k++] = gravity[i] * obs_gravity_scale_;
        // 10.. 关节位置偏差（velocity_pd 关节位置观测置零，与训练 mute 一致）
        for (std::size_t i = 0; i < dof_; ++i) {
            obs[k++] = is_velocity_pd_joint_(i)
                ? 0.0
                : (read_joint_angle_(i) - default_dof_pos_[i]) * obs_dof_pos_scale_;
        }
        // 关节速度
        for (std::size_t i = 0; i < dof_; ++i)
            obs[k++] = read_joint_velocity_(i) * obs_dof_vel_scale_;
        // 上一帧动作（原始动作，与训练 obs 一致）
        for (std::size_t i = 0; i < rl_action_size_; ++i)
            obs[k++] = last_actions_[i];

        if (k != rl_obs_size_)
            return false; // 观测尺寸与模型合同不符

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
            if (rl_publish_network_io_)
                (*rl_action_output_) = action_;
        }

        apply_pd_();
    }

    // ---- PD（与训练 _compute_torques 一致：位置组 PD + 速度组 PD，输出力矩）----
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

    // ---- PREPARE：position_pd 关节插值到预备位 ----
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

    // ---- 输出 ----
    void write_outputs_(const std::vector<double>& torques) {
        for (std::size_t i = 0; i < dof_; ++i) {
            const double value = std::isfinite(torques[i]) ? torques[i] : 0.0;
            *joint_control_torque_output_[i] = value;
        }
    }

    void write_zero_outputs_() { write_outputs_(std::vector<double>(dof_, 0.0)); }

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

    // ---- 读取 ----
    double read_joint_angle_(std::size_t index) const {
        return joint_angle_input_[index].ready() ? *joint_angle_input_[index] : 0.0;
    }
    double read_joint_velocity_(std::size_t index) const {
        return joint_velocity_input_[index].ready() ? *joint_velocity_input_[index] : 0.0;
    }

    // ---- 参数 ----
    std::vector<std::string> joint_names_;
    std::string joint_base_path_ = "/wheel_leg";
    std::vector<std::int64_t> position_pd_joints_;
    std::vector<std::int64_t> velocity_pd_joints_;
    std::size_t dof_ = 0;

    std::vector<double> default_dof_pos_;
    std::vector<double> dof_pos_limits_lower_;
    std::vector<double> dof_pos_limits_upper_;
    double position_action_scale_ = 0.5;
    double velocity_action_scale_ = 10.0;
    double max_velocity_ = 100.0;
    double position_kp_ = 200.0;
    double position_kd_ = 4.0;
    double velocity_kp_ = 20.0;
    double velocity_kd_ = 0.5;
    double position_torque_max_ = 35.0;
    double velocity_torque_max_ = 6.0;

    std::size_t rl_obs_size_ = 28;
    std::size_t rl_action_size_ = 6;
    double obs_height_scale_ = 5.0;
    double obs_ang_vel_scale_ = 0.5;
    double obs_gravity_scale_ = 1.0;
    double obs_dof_pos_scale_ = 1.0;
    double obs_dof_vel_scale_ = 0.1;
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
    std::vector<double> action_;
    std::vector<double> last_actions_;
    std::vector<double> prepare_pos_;
    bool prepare_reached_ = false;
    bool inference_ready_ = false;

    std::chrono::steady_clock::time_point last_update_time_{};
    std::chrono::steady_clock::time_point last_inference_time_{};
    bool last_inference_time_initialized_ = false;
    std::size_t last_reset_count_ = 0;

    OnnxRuntimeInference inference_;

    // ---- 接口 ----
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
};

} // namespace rmcs::rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs::rl::RlController, rmcs_executor::Component)
