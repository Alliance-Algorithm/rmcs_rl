#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/node.hpp>

namespace rmcs_rl {

std::optional<double> number_parameter(rclcpp::Node& node, const std::string& name);
double number_or(rclcpp::Node& node, const std::string& name, double fallback);
std::uint64_t parse_u64(const std::string& text, const std::string& name);
std::optional<std::int64_t> integer_parameter(rclcpp::Node& node, const std::string& name);
bool bool_or(rclcpp::Node& node, const std::string& name, bool fallback);
std::string string_or(rclcpp::Node& node, const std::string& name, const std::string& fallback);
std::vector<std::string> string_array_or(rclcpp::Node& node, const std::string& name);

} // namespace rmcs_rl
