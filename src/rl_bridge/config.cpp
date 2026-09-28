#include <rmcs_rl/rl_bridge/config.hpp>

#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <rmcs_rl/parameters.hpp>
#include <rmcs_rl/rl_bridge/term_parser.hpp>
#include <rmcs_rl/rl_bridge/utility.hpp>

namespace rmcs_rl {

BridgeConfig load_bridge_config(rclcpp::Node& node) {
    BridgeConfig config;
    config.rl_base = string_or(node, "rl_base", "/rl");
    config.joint_config = load_joint_config(node);
    const auto action_specs = string_array_or(node, "action_terms");
    if (action_specs.empty())
        throw std::invalid_argument("RlBridge: required parameter 'action_terms' is missing");
    std::unordered_set<std::string> output_paths;
    std::set<std::size_t> action_indices;
    for (const auto& spec : action_specs) {
        auto term = parse_action_term(spec);
        if (!output_paths.emplace(term.output).second)
            throw std::invalid_argument(
                "RlBridge: two action terms map to the same output "
                "interface '"
                + term.output + "'");
        if (!action_indices.emplace(term.index).second)
            throw std::invalid_argument(
                "RlBridge: duplicate action index " + std::to_string(term.index)
                + " (each slot must appear exactly once)");
        config.action_terms.push_back(std::move(term));
    }
    for (std::size_t i = 0; i < config.action_terms.size(); ++i)
        if (config.action_terms[i].index != i)
            throw std::invalid_argument(
                "RlBridge: action indices must form the complete "
                "permutation 0.."
                + std::to_string(config.action_terms.size() - 1)
                + "; got index=" + std::to_string(config.action_terms[i].index) + " at position "
                + std::to_string(i));
    config.action_size = config.action_terms.size();
    if (const auto declared = integer_parameter(node, "rl_action_size");
        declared.has_value() && static_cast<std::size_t>(*declared) != config.action_size)
        throw std::invalid_argument(
            "RlBridge: rl_action_size=" + std::to_string(*declared) + " but action_terms has "
            + std::to_string(config.action_size) + " entries");
    const auto observation_specs = string_array_or(node, "observation_terms");
    if (observation_specs.empty())
        throw std::invalid_argument(
            "RlBridge: required parameter 'observation_terms' is "
            "missing");
    const TermParseContext context{node, config.joint_config, config.action_size};
    for (const auto& spec : observation_specs)
        config.obs_terms.push_back(parse_obs_term(spec, context));

    assign_observation_indices(config.obs_terms);
    config.obs_frame_size = [&] {
        std::size_t cursor = 0;
        for (const auto& term : config.obs_terms)
            cursor += term.dim;
        return cursor;
    }();
    const auto history_length = integer_parameter(node, "history_length").value_or(1);
    if (history_length < 1 || history_length > 64)
        throw std::invalid_argument("RlBridge: history_length must be in [1, 64]");
    config.history_length = static_cast<std::size_t>(history_length);
    config.obs_size = config.obs_frame_size * config.history_length;
    if (const auto declared = integer_parameter(node, "rl_obs_size");
        declared.has_value() && static_cast<std::size_t>(*declared) != config.obs_size)
        throw std::invalid_argument(
            "RlBridge: rl_obs_size=" + std::to_string(*declared)
            + " but observation frame/history contract is " + std::to_string(config.obs_frame_size)
            + "x" + std::to_string(config.history_length) + "=" + std::to_string(config.obs_size));
    config.policy_rate = number_or(node, "policy_rate", 50.0);
    if (!(config.policy_rate > 0.0) || !is_finite(config.policy_rate))
        throw std::invalid_argument("RlBridge: policy_rate must be finite and > 0");
    config.pub_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / config.policy_rate));
    config.max_action_age = number_or(node, "max_action_age", 2.0 / config.policy_rate);
    if (!(config.max_action_age > 0.0) || !is_finite(config.max_action_age))
        throw std::invalid_argument("RlBridge: max_action_age must be finite and > 0");
    config.expected_model_id =
        parse_u64(string_or(node, "expected_model_id", "0"), "expected_model_id");

    const std::string invalid = string_or(node, "invalid_value", "nan");
    if (invalid == "nan")
        config.invalid_mode = InvalidMode::kNaN;
    else if (invalid == "zero")
        config.invalid_mode = InvalidMode::kZero;
    else if (invalid == "hold")
        config.invalid_mode = InvalidMode::kHold;
    else
        throw std::invalid_argument(
            "RlBridge: invalid_value must be nan|zero|hold "
            "(quote it in YAML: invalid_value: \"nan\" — an unquoted "
            "nan is parsed as a float), got '"
            + string_or(node, "invalid_value", "<non-string>") + "'");

    config.enable_path = string_or(node, "enable_interface", config.rl_base + "/enable");
    config.enable_default = bool_or(node, "enable_default", false);
    config.reset_path = string_or(node, "reset_interface", "");
    return config;
}

} // namespace rmcs_rl
