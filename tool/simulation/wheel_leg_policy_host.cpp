// Local, pipe-driven simulation host. Only these four plugins can be loaded;
// no executor YAML component list, hardware class, CAN or control publisher.
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_msgs/chassis_mode.hpp>

namespace rmcs_executor {
// Binding adapter for the local simulation host, using the same declarations
// as the executor. It deliberately has no hardware/component-list loader.
class Executor {
public:
    static void append(
        std::vector<std::shared_ptr<Component>>& list,
        const std::shared_ptr<Component>& component) {
        list.push_back(component);
        for (const auto& partner : component->partner_component_list_)
            append(list, partner);
    }
    static void pair(const std::vector<std::shared_ptr<Component>>& list) {
        Component::OutputInfoMap info;
        for (const auto& component : list)
            for (const auto& output : component->output_list_)
                if (!info.emplace(output.name, Component::OutputInfo{output.type, output.kind})
                         .second)
                    throw std::runtime_error("duplicate simulation output: " + output.name);
        for (const auto& component : list)
            component->before_pairing(info);
        for (const auto& component : list) {
            for (auto& input : component->input_list_) {
                bool found = false;
                for (const auto& source : list)
                    for (auto& output : source->output_list_)
                        if (input.name == output.name) {
                            if (input.type != output.type || input.kind != output.kind)
                                throw std::runtime_error("simulation binding type: " + input.name);
                            input.bind(input.binding, output.binding);
                            found = true;
                        }
                if (!found && input.required)
                    throw std::runtime_error("missing simulation input: " + input.name);
            }
        }
        for (const auto& component : list)
            component->before_updating();
    }
    template <class T>
    static const T&
        read(const std::vector<std::shared_ptr<Component>>& list, const std::string& name) {
        for (const auto& component : list)
            for (const auto& output : component->output_list_)
                if (output.name == name && output.type == typeid(T))
                    return *static_cast<const T*>(output.binding);
        throw std::runtime_error("missing simulation result: " + name);
    }
};
} // namespace rmcs_executor

class SimulationFeedback final : public rmcs_executor::Component {
public:
    SimulationFeedback() {
        register_output("/predefined/update_rate", rate_, 1000.);
        register_output("/predefined/update_count", tick_, std::size_t{0});
        register_output("/predefined/timestamp", timestamp_, std::chrono::steady_clock::now());
        register_output("/wheel_leg/feedback_available", available_, true);
        register_output("/wheel_leg/dm_control_ready", ready_, true);
        register_output("/wheel_leg/simulation", simulation_, true);
        register_output("/wheel_leg/imu/quaternion", orientation_, Eigen::Quaterniond::Identity());
        register_output("/wheel_leg/imu/angular_velocity", gyro_, Eigen::Vector3d::Zero());
        register_output("/chassis/control_enable", enabled_, false);
        register_output("/chassis/reset_count", reset_, std::size_t{0});
        register_output("/chassis/control_mode", mode_, rmcs_msgs::ChassisMode::AUTO);
        register_output("/chassis/control_height", height_, .305);
        register_output("/chassis/control_velocity", command_);
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto base = std::string{"/wheel_leg/"} + names[i];
            register_output(base + "/angle", q_[i], 0.);
            register_output(base + "/velocity", dq_[i], 0.);
            register_output(base + "/max_torque", caps_[i], i < 4 ? 40. : 4.5);
        }
    }
    void accept(const nlohmann::json& value) {
        const auto q = value.at("positions").get<std::array<double, 6>>();
        const auto dq = value.at("velocities").get<std::array<double, 6>>();
        const auto orientation = value.at("quaternion_wxyz").get<std::array<double, 4>>();
        const auto gyro = value.at("gyro_body").get<std::array<double, 3>>();
        const auto cmd = value.at("command").get<std::array<double, 3>>();
        for (int i = 0; i < 6; ++i) {
            *q_[i] = q[i];
            *dq_[i] = dq[i];
        }
        *orientation_ =
            Eigen::Quaterniond{orientation[0], orientation[1], orientation[2], orientation[3]};
        *gyro_ = Eigen::Vector3d{gyro[0], gyro[1], gyro[2]};
        command_->vector = Eigen::Vector3d{cmd[0], cmd[1], cmd[2]};
        *height_ = value.at("height").get<double>();
        *enabled_ = value.at("enabled").get<bool>();
        *reset_ = value.at("reset_count").get<std::size_t>();
        *mode_ = static_cast<rmcs_msgs::ChassisMode>(value.at("mode").get<int>());
    }
    void update() override {
        ++*tick_;
        *timestamp_ = std::chrono::steady_clock::now();
    }
    inline static constexpr std::array<const char*, 6> names{"left_hip_joint",  "left_knee_joint",
                                                             "right_hip_joint", "right_knee_joint",
                                                             "left_wheel",      "right_wheel"};

private:
    OutputInterface<double> rate_, height_;
    OutputInterface<std::size_t> tick_, reset_;
    OutputInterface<std::chrono::steady_clock::time_point> timestamp_;
    OutputInterface<bool> available_, ready_, simulation_, enabled_;
    OutputInterface<Eigen::Quaterniond> orientation_;
    OutputInterface<Eigen::Vector3d> gyro_;
    OutputInterface<rmcs_msgs::ChassisMode> mode_;
    OutputInterface<rmcs_description::BaseLink::DirectionVector> command_;
    std::array<OutputInterface<double>, 6> q_, dq_, caps_;
};

int main(int argc, char** argv) {
    using rmcs_executor::Component;
    using rmcs_executor::Executor;
    try {
        rclcpp::init(argc, argv);
        pluginlib::ClassLoader<Component> loader{"rmcs_executor", "rmcs_executor::Component"};
        Component::initializing_component_name = "simulation_feedback";
        auto feedback = std::make_shared<SimulationFeedback>();
        std::vector<std::shared_ptr<Component>> components{feedback};
        const auto load = [&](const std::string& name, const std::string& type) {
            Component::initializing_component_name = name;
            auto result = loader.createSharedInstance(type);
            Executor::append(components, result);
            return result;
        };
        auto estimator = load(
            "wheel_leg_state_estimator", "rmcs_core::controller::chassis::WheelLegStateEstimator");
        auto controller =
            load("wheel_leg_rl_controller", "rmcs_core::controller::chassis::WheelLegRlController");
        auto observation =
            load("wheel_leg_observation", "rmcs_core::controller::chassis::WheelLegObservation");
        auto bridge = load("rl_bridge", "rmcs_rl::RlBridge");
        Executor::pair(components);
        std::cout
            << nlohmann::
                   json{{"ready", true}, {"hardware_created", false}, {"executor_hz", 1000}, {"pd_hz", 200}, {"policy_hz", 50}}
                       .dump()
            << std::endl;
        for (std::string line; std::getline(std::cin, line);) {
            const auto input = nlohmann::json::parse(line);
            feedback->accept(input);
            const int ticks = input.value("ticks", 5);
            if (ticks < 1 || ticks > 20)
                throw std::invalid_argument("simulation ticks must be in [1,20]");
            for (int i = 0; i < ticks; ++i) {
                feedback->update();
                estimator->update();
                // Action partner runs before the controller, observation/submit after.
                for (const auto& component : components)
                    if (component != feedback && component != estimator && component != controller
                        && component != observation && component != bridge)
                        component->update();
                controller->update();
                observation->update();
                bridge->update();
            }
            std::array<double, 6> effort{};
            for (int i = 0; i < 6; ++i)
                effort[i] = Executor::read<double>(
                    components,
                    std::string{"/wheel_leg/"} + SimulationFeedback::names[i] + "/control_torque");
            std::cout
                << nlohmann::json{{"effort_api", effort}, {"state", Executor::read<int>(components, "/wheel_leg/rl/state")}, {"phase", Executor::read<int>(components, "/wheel_leg/recovery/phase")}, {"fault", Executor::read<int>(components, "/wheel_leg/recovery/fault")}, {"fault_latched", Executor::read<bool>(components, "/wheel_leg/rl/fault_latched")}, {"action_seq", Executor::read<std::size_t>(components, "/wheel_leg/rl/action_seq")}, {"applied_action", Executor::read<std::array<double, 6>>(components, "/wheel_leg/rl/applied_action")}, {"enable", Executor::read<bool>(components, "/wheel_leg/enable_request")}}
                       .dump()
                << std::endl;
        }
        rclcpp::shutdown();
    } catch (const std::exception& error) {
        std::cerr << "local simulation host: " << error.what() << '\n';
        return 1;
    }
}
