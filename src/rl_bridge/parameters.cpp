#include <rmcs_rl/rl_bridge/parameters.hpp>

#include <cmath>
#include <stdexcept>

namespace rmcs_rl {

std::optional<double> number_parameter(rclcpp::Node& node, const std::string& name) {
    rclcpp::Parameter parameter;
    try {
        if (!node.get_parameter(name, parameter))
            return std::nullopt;
    } catch (const std::exception&) {
        return std::nullopt;
    }
    switch (parameter.get_type()) {
    case rclcpp::ParameterType::PARAMETER_DOUBLE: return parameter.as_double();
    case rclcpp::ParameterType::PARAMETER_INTEGER: return static_cast<double>(parameter.as_int());
    case rclcpp::ParameterType::PARAMETER_NOT_SET: return std::nullopt;
    default: throw std::invalid_argument("RlBridge: parameter '" + name + "' must be a number");
    }
}

double number_or(rclcpp::Node& node, const std::string& name, double fallback) {
    const auto value = number_parameter(node, name);
    if (!value.has_value())
        return fallback;
    if (!std::isfinite(*value))
        throw std::invalid_argument("RlBridge: parameter '" + name + "' is not finite");
    return *value;
}

std::uint64_t parse_u64(const std::string& text, const std::string& name) {
    try {
        std::size_t consumed = 0;
        const bool hex = text.rfind("0x", 0) == 0 || text.rfind("0X", 0) == 0;
        const auto value = std::stoull(hex ? text.substr(2) : text, &consumed, hex ? 16 : 10);
        if (consumed != (hex ? text.size() - 2 : text.size()))
            throw std::invalid_argument("trailing characters");
        return static_cast<std::uint64_t>(value);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "RlBridge: parameter '" + name
            + "' must be a decimal or 0x-prefixed 64-bit id, got '" + text + "'");
    }
}

std::optional<std::int64_t> integer_parameter(rclcpp::Node& node, const std::string& name) {
    const auto value = number_parameter(node, name);
    if (!value.has_value())
        return std::nullopt;
    const auto rounded = std::llround(*value);
    if (std::abs(*value - static_cast<double>(rounded)) > 1e-9)
        throw std::invalid_argument("RlBridge: parameter '" + name + "' must be an integer");
    return rounded;
}

bool bool_or(rclcpp::Node& node, const std::string& name, bool fallback) {
    rclcpp::Parameter parameter;
    try {
        if (!node.get_parameter(name, parameter))
            return fallback;
    } catch (const std::exception&) {
        return fallback;
    }
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
        return fallback;
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL)
        throw std::invalid_argument("RlBridge: parameter '" + name + "' must be a boolean");
    return parameter.as_bool();
}

std::string string_or(rclcpp::Node& node, const std::string& name, const std::string& fallback) {
    rclcpp::Parameter parameter;
    try {
        if (!node.get_parameter(name, parameter))
            return fallback;
    } catch (const std::exception&) {
        return fallback;
    }
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
        return fallback;
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING)
        throw std::invalid_argument("RlBridge: parameter '" + name + "' must be a string");
    return parameter.as_string();
}

std::vector<std::string> string_array_or(rclcpp::Node& node, const std::string& name) {
    std::vector<std::string> value;
    try {
        if (!node.get_parameter(name, value))
            return {};
    } catch (const std::exception&) {
        throw std::invalid_argument("RlBridge: parameter '" + name + "' must be a list of strings");
    }
    return value;
}

} // namespace rmcs_rl
