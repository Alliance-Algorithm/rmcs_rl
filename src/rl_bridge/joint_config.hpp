#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/node.hpp>

namespace rmcs_rl {

struct JointConfig {
    std::vector<std::string> names;
    std::string base_path;
    std::unordered_map<std::string, std::size_t> index;
    std::unordered_map<std::string, std::string> angle_suffix;
    std::unordered_map<std::string, std::string> velocity_suffix;
    std::unordered_map<std::string, std::string> torque_suffix;
};

JointConfig load_joint_config(rclcpp::Node& node);
std::string joint_path(const JointConfig& config, const std::string& joint, char field);
std::optional<double>
default_joint_pos(rclcpp::Node& node, const JointConfig& config, std::size_t joint);

} // namespace rmcs_rl
