#include "configuration.hpp"

#include <cmath>
#include <stdexcept>
#include <string_view>
#include <variant>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <rmcs_rl/parameters.hpp>

namespace rmcs_rl {

PolicyConfig load_policy_config(
    rclcpp::Node& node, std::size_t observation_size, std::size_t action_size,
    std::size_t sequence_length, std::size_t feature_size) {
    PolicyConfig config;
    config.path = string_or(node, "rl_model_path", string_or(node, "model_path", ""));
    if (config.path.empty())
        throw std::invalid_argument("required YAML parameter 'rl_model_path' is empty");
    if (config.path.front() != '/')
        config.path = ament_index_cpp::get_package_share_directory("rmcs_rl") + "/" + config.path;

    config.input_name = string_or(node, "input_name", "");
    config.output_name = string_or(node, "output_name", "");
    config.model_type = string_or(node, "model_type", "auto");
    config.expected_model_spec_sha256 =
        string_or(node, "expected_model_spec_sha256", "");
    if (config.model_type == "wheel_leg_v7" && config.expected_model_spec_sha256.empty())
        config.expected_model_spec_sha256 =
            "0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997";
    config.expected_leg_winding_version = string_or(node, "leg_winding_version", "");
    config.observation_size = observation_size;
    config.action_size = action_size;
    config.sequence_length = sequence_length;
    config.feature_size = feature_size;
    config.normalization_from_metadata = bool_or(node, "normalization_from_metadata", true);
    if (const double clip = number_or(node, "obs_clip", -1.0); clip >= 0.0)
        config.obs_clip = clip;
    if (const double clip = number_or(node, "action_clip", -1.0); clip >= 0.0)
        config.action_clip = clip;

    for (const auto& parameter_name : node.list_parameters({"extra_inputs"}, 1).names) {
        static constexpr std::string_view kPrefix = "extra_inputs.";
        if (!parameter_name.starts_with(kPrefix))
            continue;
        const auto extra_name = parameter_name.substr(kPrefix.size());
        if (extra_name.empty())
            throw std::invalid_argument(
                "extra_inputs parameter must use the form extra_inputs.<input_name>");
        const auto parameter = node.get_parameter(parameter_name);
        PolicyExtraInput values;
        switch (parameter.get_type()) {
        case rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY:
            values = parameter.as_double_array();
            break;
        case rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY:
            values = parameter.as_integer_array();
            break;
        case rclcpp::ParameterType::PARAMETER_BOOL_ARRAY: values = parameter.as_bool_array(); break;
        default:
            throw std::invalid_argument(
                "extra_inputs." + std::string{extra_name}
                + " must be a float, integer, or bool list");
        }
        if (std::visit([](const auto& items) { return items.empty(); }, values))
            throw std::invalid_argument("extra_inputs." + std::string{extra_name} + " is empty");
        if (const auto* floats = std::get_if<std::vector<double>>(&values))
            for (const double value : *floats)
                if (!std::isfinite(value))
                    throw std::invalid_argument(
                        "extra_inputs." + std::string{extra_name} + " contains a non-finite value");
        config.extra_input_values.emplace(std::string{extra_name}, std::move(values));
    }
    return config;
}

} // namespace rmcs_rl
