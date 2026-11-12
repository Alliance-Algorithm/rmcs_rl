#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <eigen3/Eigen/Geometry>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_msgs/chassis_mode.hpp>
#include <rmcs_msgs/keyboard.hpp>
#include <rmcs_msgs/switch.hpp>
#include <rmcs_utility/rclcpp/node_mixin.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/header.hpp>

namespace rmcs_rl {

class WheelLegMirrorComponent
    : public rmcs_executor::Component
    , public rclcpp::Node
    , public rmcs_utility::NodeMixin {

    using Clock = std::chrono::steady_clock;
    using JointVector = std::array<double, 6>;

public:
    WheelLegMirrorComponent()
        : Node{get_component_name(), node::options()} {
        enabled_ = node::param_or("enabled", true);
        const double rate = node::param_or("update_rate", 100.0);
        if (!std::isfinite(rate) || rate <= 0. || rate > 1000.)
            throw std::invalid_argument("Wheel-leg telemetry update_rate must be in (0, 1000]");
        period_ =
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>{1. / rate});
        if (!enabled_)
            return;

        register_input("/predefined/timestamp", timestamp_);
        register_input("/chassis/reset_count", reset_);
        for (std::size_t i = 0; i < motors_.size(); ++i) {
            const auto prefix = std::string{"/wheel_leg/"} + kMotorNames[i];
            register_input(prefix + "/angle", motors_[i].angle);
            register_input(prefix + "/velocity", motors_[i].velocity);
            register_input(prefix + "/feedback_steady_ns", motors_[i].feedback_ns);
        }
        for (std::size_t i = 0; i < gimbal_.size(); ++i) {
            const auto prefix = std::string{"/gimbal/"} + kGimbalAxes[i];
            register_input(prefix + "/angle", gimbal_[i].angle);
            register_input(prefix + "/velocity", gimbal_[i].velocity);
        }
        register_input("/gimbal/left_friction/velocity", left_friction_velocity_);
        register_input("/gimbal/right_friction/velocity", right_friction_velocity_);
        register_input("/gimbal/bullet_feeder/velocity", bullet_feeder_velocity_);
        register_input("/wheel_leg/imu/quaternion", orientation_);
        register_input("/wheel_leg/imu/angular_velocity", gyro_);
        register_input("/wheel_leg/imu/last_steady_ns", imu_ns_);
        register_input("/remote/joystick/left", joystick_left_, false);
        register_input("/remote/joystick/right", joystick_right_, false);
        register_input("/remote/switch/left", switch_left_, false);
        register_input("/remote/switch/right", switch_right_, false);
        register_input("/remote/rotary_knob", knob_, false);
        register_input("/remote/keyboard", keyboard_, false);
        register_input("/wheel_leg/dr16_fresh", remote_fresh_, false);
        register_input("/chassis/control_velocity", command_, false);
        register_input("/chassis/control_height", height_, false);
        register_input("/chassis/control_mode", mode_, false);
        register_input("/wheel_leg/rl/state", rl_state_, false);
        register_input("/wheel_leg/recovery/phase", recovery_phase_, false);
        register_input("/wheel_leg/recovery/fault", recovery_fault_, false);
        register_input("/gimbal/yaw/control_velocity", gimbal_yaw_velocity_, false);
        register_input("/gimbal/pitch/control_torque", gimbal_pitch_torque_, false);
        register_input("/wheel_leg/feedback_fresh", fresh_);
        register_input("/chassis/control_enable", control_enabled_, false);
        register_input("/wheel_leg/dm_control_ready", dm_control_ready_, false);
        register_input("/wheel_leg/state/valid", valid_);
        register_input("/wheel_leg/state/position", model_position_);
        register_input("/wheel_leg/state/gravity", gravity_);
        register_input("/wheel_leg/state/passive_knee", knee_);
        register_input("/wheel_leg/state/leg_length", length_);
        register_input("/wheel_leg/state/model_offsets", model_offsets_);
        register_input("/wheel_leg/state/calibration_sha256", calibration_sha256_);
        register_input(
            "/wheel_leg/state/calibration_model_spec_sha256", calibration_model_spec_sha256_);

        const auto qos = rclcpp::QoS{5}.best_effort().durability_volatile();
        joints_pub_ = create_publisher<sensor_msgs::msg::JointState>(kPrefix + "joint_states", qos);
        imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(kPrefix + "imu_body", qos);
        gravity_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>(
            kPrefix + "rl_projected_gravity", qos);
        gyro_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>(
            kPrefix + "rl_angular_velocity", qos);
        diagnostics_pub_ =
            create_publisher<diagnostic_msgs::msg::DiagnosticArray>(kPrefix + "diagnostics", qos);
    }

    void update() override {
        if (!enabled_ || !publish_due())
            return;

        std_msgs::msg::Header header;
        header.stamp = get_clock()->now();
        header.frame_id = "chassis_body";
        const auto joints = build_joint_state(header);
        const auto imu = build_imu(header);
        const auto gravity = build_vector(header, *gravity_);
        const auto gyro = build_vector(header, *gyro_);
        const auto diagnostics = build_diagnostics(header);

        joints_pub_->publish(joints);
        imu_pub_->publish(imu);
        gravity_pub_->publish(gravity);
        gyro_pub_->publish(gyro);
        diagnostics_pub_->publish(diagnostics);
    }

private:
    bool publish_due() {
        const auto now = *timestamp_;
        if (*reset_ != last_reset_) {
            last_publish_.reset();
            last_reset_ = *reset_;
        }
        if (last_publish_ && now >= *last_publish_) {
            const auto elapsed = now - *last_publish_;
            if (elapsed < period_)
                return false;
            last_publish_ = now - elapsed % period_;
        } else {
            last_publish_ = now;
        }
        return true;
    }

    sensor_msgs::msg::JointState build_joint_state(const std_msgs::msg::Header& header) const {
        sensor_msgs::msg::JointState joints;
        joints.header = header;
        for (std::size_t i = 0; i < motors_.size(); ++i) {
            joints.name.emplace_back(kMotorNames[i]);
            joints.position.push_back(*motors_[i].angle);
            joints.velocity.push_back(*motors_[i].velocity);
        }
        for (std::size_t i = 0; i < gimbal_.size(); ++i) {
            joints.name.emplace_back(kGimbalNames[i]);
            joints.position.push_back(*gimbal_[i].angle);
            joints.velocity.push_back(*gimbal_[i].velocity);
        }
        return joints;
    }

    sensor_msgs::msg::Imu build_imu(const std_msgs::msg::Header& header) const {
        sensor_msgs::msg::Imu imu;
        imu.header = header;
        const auto& q = *orientation_;
        imu.orientation.w = q.w();
        imu.orientation.x = q.x();
        imu.orientation.y = q.y();
        imu.orientation.z = q.z();
        imu.angular_velocity.x = gyro_->x();
        imu.angular_velocity.y = gyro_->y();
        imu.angular_velocity.z = gyro_->z();
        imu.linear_acceleration_covariance[0] = -1.;
        if (!q.coeffs().allFinite() || std::abs(q.squaredNorm() - 1.) > .01)
            imu.orientation_covariance[0] = -1.;
        return imu;
    }

    static geometry_msgs::msg::Vector3Stamped
        build_vector(const std_msgs::msg::Header& header, const Eigen::Vector3d& value) {
        geometry_msgs::msg::Vector3Stamped message;
        message.header = header;
        message.vector.x = value.x();
        message.vector.y = value.y();
        message.vector.z = value.z();
        return message;
    }

    diagnostic_msgs::msg::DiagnosticArray
        build_diagnostics(const std_msgs::msg::Header& header) const {
        diagnostic_msgs::msg::DiagnosticArray diagnostics;
        diagnostics.header = header;
        auto& status = diagnostics.status.emplace_back();
        status.name = "wheel_leg/telemetry";
        status.hardware_id = "wheel_leg";
        status.level = *fresh_ && *valid_ ? diagnostic_msgs::msg::DiagnosticStatus::OK
                                          : diagnostic_msgs::msg::DiagnosticStatus::WARN;
        status.message = *fresh_ && *valid_ ? "valid" : "feedback or estimator invalid";
        const auto add = [&status](const std::string& key, const std::string& value) {
            auto& entry = status.values.emplace_back();
            entry.key = key;
            entry.value = value;
        };
        const auto flag = [](bool value) { return value ? "true" : "false"; };

        add("schema", "wheel_leg_telemetry_v1");
        add("coordinate", "rmcs_hardware_output_shaft_v1");
        add("calibration_sha256", *calibration_sha256_);
        add("calibration_model_spec_sha256", *calibration_model_spec_sha256_);
        add("feedback_fresh", flag(*fresh_));
        add("state_valid", flag(*valid_));
        add("control_enabled", control_enabled_.ready() ? flag(*control_enabled_) : "unknown");
        add("dm_control_ready", dm_control_ready_.ready() ? flag(*dm_control_ready_) : "unknown");
        add("reset_count", std::to_string(*reset_));

        const bool command_ready = command_.ready() && height_.ready() && mode_.ready()
                                && control_enabled_.ready() && remote_fresh_.ready();
        add("operator/ready", flag(command_ready));
        if (command_ready) {
            add("operator/remote_fresh", flag(*remote_fresh_));
            add("operator/vx", number(command_->vector.x()));
            add("operator/vy", number(command_->vector.y()));
            add("operator/wz", number(command_->vector.z()));
            add("operator/height_m", number(*height_));
            add("operator/mode", std::to_string(static_cast<int>(*mode_)));
        }
        if (joystick_left_.ready() && joystick_right_.ready() && switch_left_.ready()
            && switch_right_.ready() && knob_.ready() && keyboard_.ready()) {
            add("remote/left_x", number(joystick_left_->x()));
            add("remote/left_y", number(joystick_left_->y()));
            add("remote/right_x", number(joystick_right_->x()));
            add("remote/right_y", number(joystick_right_->y()));
            add("remote/switch_left", std::to_string(static_cast<int>(*switch_left_)));
            add("remote/switch_right", std::to_string(static_cast<int>(*switch_right_)));
            add("remote/knob", number(*knob_));
            add("remote/keyboard", std::to_string(std::bit_cast<std::uint16_t>(*keyboard_)));
        }
        if (rl_state_.ready())
            add("rl/state", std::to_string(*rl_state_));
        if (recovery_phase_.ready())
            add("recovery/phase", std::to_string(*recovery_phase_));
        if (recovery_fault_.ready())
            add("recovery/fault", std::to_string(*recovery_fault_));
        if (gimbal_yaw_velocity_.ready())
            add("gimbal/yaw_control_velocity", number(*gimbal_yaw_velocity_));
        if (gimbal_pitch_torque_.ready())
            add("gimbal/pitch_control_torque", number(*gimbal_pitch_torque_));

        const auto steady_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
                .count();
        const auto age = [steady_ns](std::uint64_t sample_ns) {
            return sample_ns == 0 || sample_ns > static_cast<std::uint64_t>(steady_ns)
                     ? -1.
                     : static_cast<double>(steady_ns - sample_ns) * 1e-9;
        };
        add("imu_age_s", number(age(*imu_ns_)));
        for (std::size_t i = 0; i < motors_.size(); ++i) {
            const auto name = std::string{kMotorNames[i]};
            add("motor_age_s/" + name, number(age(*motors_[i].feedback_ns)));
            add("model_position_rad/" + name, number((*model_position_)[i]));
            if (i < 4)
                add("model_offset_rad/" + name, number((*model_offsets_)[i]));
        }
        add("gimbal_state_ready", "true");
        add("gimbal/yaw_angle_rad", number(*gimbal_[0].angle));
        add("gimbal/yaw_velocity_rad_s", number(*gimbal_[0].velocity));
        add("gimbal/pitch_angle_rad", number(*gimbal_[1].angle));
        add("gimbal/pitch_velocity_rad_s", number(*gimbal_[1].velocity));
        add("gimbal/left_friction_velocity_rad_s", number(*left_friction_velocity_));
        add("gimbal/right_friction_velocity_rad_s", number(*right_friction_velocity_));
        add("gimbal/bullet_feeder_velocity_rad_s", number(*bullet_feeder_velocity_));
        for (std::size_t side = 0; side < 2; ++side) {
            const auto prefix = std::string{side == 0 ? "left/" : "right/"};
            add(prefix + "passive_knee_rad", number((*knee_)[side]));
            add(prefix + "leg_length_m", number((*length_)[side]));
        }
        return diagnostics;
    }

    static std::string number(double value) { return std::format("{}", value); }

    inline static const std::string kPrefix = "/wheel_leg/telemetry/";
    inline static constexpr std::array<const char*, 6> kMotorNames{
        "left_hip_joint",   "left_knee_joint", "right_hip_joint",
        "right_knee_joint", "left_wheel",      "right_wheel"};
    inline static constexpr std::array<const char*, 2> kGimbalNames{
        "gimbal_yaw_joint", "gimbal_pitch_joint"};
    inline static constexpr std::array<const char*, 2> kGimbalAxes{"yaw", "pitch"};
    struct MotorInputs {
        InputInterface<double> angle, velocity;
        InputInterface<std::uint64_t> feedback_ns;
    };

    struct GimbalInputs {
        InputInterface<double> angle, velocity;
    };

    InputInterface<Clock::time_point> timestamp_;
    InputInterface<std::size_t> reset_;
    std::array<MotorInputs, 6> motors_;
    std::array<GimbalInputs, 2> gimbal_;
    InputInterface<double> left_friction_velocity_, right_friction_velocity_,
        bullet_feeder_velocity_;
    InputInterface<Eigen::Quaterniond> orientation_;
    InputInterface<Eigen::Vector3d> gyro_, gravity_;
    InputInterface<std::uint64_t> imu_ns_;
    InputInterface<Eigen::Vector2d> joystick_left_, joystick_right_;
    InputInterface<rmcs_msgs::Switch> switch_left_, switch_right_;
    InputInterface<rmcs_msgs::Keyboard> keyboard_;
    InputInterface<double> knob_, height_, gimbal_yaw_velocity_, gimbal_pitch_torque_;
    InputInterface<rmcs_description::BaseLink::DirectionVector> command_;
    InputInterface<rmcs_msgs::ChassisMode> mode_;
    InputInterface<bool> remote_fresh_;
    InputInterface<int> rl_state_, recovery_phase_, recovery_fault_;
    InputInterface<bool> fresh_, valid_;
    InputInterface<bool> control_enabled_, dm_control_ready_;
    InputInterface<JointVector> model_position_;
    InputInterface<std::array<double, 2>> knee_, length_;
    InputInterface<std::array<double, 4>> model_offsets_;
    InputInterface<std::string> calibration_sha256_, calibration_model_spec_sha256_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joints_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr gravity_pub_, gyro_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    bool enabled_ = true;
    Clock::duration period_{};
    std::optional<Clock::time_point> last_publish_;
    std::size_t last_reset_ = 0;
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_rl::WheelLegMirrorComponent, rmcs_executor::Component)
