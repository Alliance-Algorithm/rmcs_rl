#include "rl_bridge/joint_config.hpp"

#include <stdexcept>

#include "rl_bridge/parameters.hpp"
#include "rl_bridge/utility.hpp"

namespace rmcs_rl {

JointConfig load_joint_config(rclcpp::Node& node) {
    JointConfig config;
    config.names = string_array_or(node, "joint_names");
    config.base_path = string_or(node, "joint_base_path", "");
    if (!config.names.empty() && config.base_path.empty())
        throw std::invalid_argument("RlBridge: joint_base_path is required when joint_names is set");
    for (std::size_t i = 0; i < config.names.size(); ++i) {
        if (config.names[i].empty())
            throw std::invalid_argument("RlBridge: joint_names contains an empty name");
        if (!config.index.emplace(config.names[i], i).second)
            throw std::invalid_argument(
                "RlBridge: duplicate joint name '" + config.names[i] + "'");
    }

    const std::string default_angle = string_or(node, "joint_angle_suffix", "/angle");
    const std::string default_velocity = string_or(node, "joint_velocity_suffix", "/velocity");
    const std::string default_torque = string_or(node, "joint_torque_suffix", "/torque");
    for (const auto& joint : config.names) {
        config.angle_suffix[joint] = default_angle;
        config.velocity_suffix[joint] = default_velocity;
        config.torque_suffix[joint] = default_torque;
    }

    constexpr const char* prefix = "joint_interface_overrides.";
    for (const auto& name : node.list_parameters({"joint_interface_overrides"}, 10).names) {
        if (name.rfind(prefix, 0) != 0)
            continue;
        const std::string remainder = name.substr(std::string{prefix}.size());
        const auto separator = remainder.find_last_of('.');
        if (separator == std::string::npos)
            throw std::invalid_argument(
                "RlBridge: joint_interface_overrides entry '" + name
                + "' must be <joint>.<angle|velocity|torque>");
        const std::string joint = remainder.substr(0, separator);
        const std::string field = remainder.substr(separator + 1);
        if (config.index.count(joint) == 0)
            throw std::invalid_argument(
                "RlBridge: joint_interface_overrides references unknown joint '" + joint + "'");
        const std::string value = string_or(node, name, "");
        if (field == "angle")
            config.angle_suffix[joint] = value;
        else if (field == "velocity")
            config.velocity_suffix[joint] = value;
        else if (field == "torque")
            config.torque_suffix[joint] = value;
        else
            throw std::invalid_argument(
                "RlBridge: joint_interface_overrides field '" + field
                + "' is not angle/velocity/torque");
    }
    return config;
}

std::string joint_path(const JointConfig& config, const std::string& joint, char field) {
    const auto suffix = field == 'a' ? config.angle_suffix.at(joint)
                      : field == 'v' ? config.velocity_suffix.at(joint)
                                     : config.torque_suffix.at(joint);
    return config.base_path + "/" + joint + suffix;
}

std::optional<double>
default_joint_pos(rclcpp::Node& node, const JointConfig& config, std::size_t joint) {
    const auto name = "default_joint_pos." + config.names[joint];
    const auto value = number_parameter(node, name);
    if (!value.has_value())
        return std::nullopt;
    if (!is_finite(*value))
        throw std::invalid_argument("RlBridge: parameter '" + name + "' is not finite");
    return value;
}

} // namespace rmcs_rl
