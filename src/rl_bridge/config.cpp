#include "rl_bridge/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace rmcs_rl {
namespace {

class Parameters {
public:
    explicit Parameters(rclcpp::Node& node, std::string prefix = {})
        : node_(node)
        , prefix_(std::move(prefix)) {}

    Parameters child(std::string_view name) const { return Parameters{node_, path(name)}; }

    std::string path(std::string_view key) const {
        return prefix_.empty() ? std::string{key} : std::format("{}.{}", prefix_, key);
    }

    std::optional<double> number(std::string_view key) const {
        const auto parameter = find(key);
        if (!parameter)
            return std::nullopt;
        switch (parameter->get_type()) {
        case rclcpp::ParameterType::PARAMETER_DOUBLE: return parameter->as_double();
        case rclcpp::ParameterType::PARAMETER_INTEGER:
            return static_cast<double>(parameter->as_int());
        default: throw invalid(key, "must be a number");
        }
    }

    double number_or(std::string_view key, double fallback) const {
        const auto value = number(key);
        if (!value)
            return fallback;
        if (!std::isfinite(*value))
            throw invalid(key, "is not finite");
        return *value;
    }

    std::optional<std::int64_t> integer(std::string_view key) const {
        const auto value = number(key);
        if (!value)
            return std::nullopt;
        const auto rounded = std::llround(*value);
        if (std::abs(*value - static_cast<double>(rounded)) > 1e-9)
            throw invalid(key, "must be an integer");
        return rounded;
    }

    bool boolean_or(std::string_view key, bool fallback) const {
        const auto parameter = find(key);
        if (!parameter)
            return fallback;
        if (parameter->get_type() != rclcpp::ParameterType::PARAMETER_BOOL)
            throw invalid(key, "must be a boolean");
        return parameter->as_bool();
    }

    std::string string_or(std::string_view key, std::string fallback) const {
        const auto parameter = find(key);
        if (!parameter)
            return fallback;
        if (parameter->get_type() != rclcpp::ParameterType::PARAMETER_STRING)
            throw invalid(key, "must be a string");
        return parameter->as_string();
    }

    std::string required_string(std::string_view key) const {
        auto value = string_or(key, "");
        if (value.empty())
            throw std::invalid_argument(path(key) + " is required");
        return value;
    }

    std::vector<std::string> strings(std::string_view key) const {
        std::vector<std::string> value;
        try {
            if (!node_.get_parameter(path(key), value))
                return {};
        } catch (const std::exception&) {
            throw invalid(key, "must be a list of strings");
        }
        return value;
    }

    std::vector<double> numbers(std::string_view key) const {
        rclcpp::Parameter parameter;
        if (!node_.get_parameter(path(key), parameter))
            return {};
        std::vector<double> result;
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
            result = parameter.as_double_array();
        } else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
            for (const auto item : parameter.as_integer_array())
                result.push_back(static_cast<double>(item));
        } else {
            throw std::invalid_argument(path(key) + " must be a numeric array");
        }
        if (result.empty()
            || !std::ranges::all_of(result, [](double v) { return std::isfinite(v); }))
            throw std::invalid_argument(path(key) + " must be a nonempty finite numeric array");
        return result;
    }

    std::optional<std::vector<std::int64_t>> integers(std::string_view key) const {
        rclcpp::Parameter parameter;
        if (!node_.get_parameter(path(key), parameter))
            return std::nullopt;
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY
            || parameter.as_integer_array().empty())
            throw std::invalid_argument(path(key) + " must be a nonempty integer array");
        return parameter.as_integer_array();
    }

    std::optional<Range> range(std::string_view key) const {
        const auto values = numbers(key);
        if (values.empty())
            return std::nullopt;
        if (values.size() != 2 || values[0] > values[1])
            throw std::invalid_argument(path(key) + " must be [lower, upper]");
        return Range{.lower = values[0], .upper = values[1]};
    }

    std::map<std::string, std::string> string_map() const {
        std::map<std::string, std::string> result;
        for (const auto& name : node_.list_parameters({prefix_}, 1).names)
            result.emplace(name.substr(prefix_.size() + 1), Parameters{node_}.string_or(name, ""));
        return result;
    }

    void check_keys(std::initializer_list<std::string_view> allowed) const {
        for (const auto& name : node_.list_parameters({prefix_}, 0).names)
            if (std::ranges::find(allowed, std::string_view{name}.substr(prefix_.size() + 1))
                == allowed.end())
                throw std::invalid_argument("unknown policy parameter: " + name);
    }

    // Names listed in "<prefix>.order"; every other block under the prefix must be listed.
    std::vector<std::string> ordered_names() const {
        const auto names = strings("order");
        if (names.empty())
            throw std::invalid_argument(path("order") + " must not be empty");
        std::unordered_set<std::string> seen;
        for (const auto& name : names)
            if (name.empty()
                || name.find_first_not_of(
                       "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                       != std::string::npos
                || !seen.emplace(name).second)
                throw std::invalid_argument(
                    path("order") + " contains invalid or duplicate name: " + name);
        for (const auto& key : node_.list_parameters({prefix_}, 0).names) {
            const auto suffix = key.substr(prefix_.size() + 1);
            const auto name = suffix.substr(0, suffix.find('.'));
            if (name != "order" && !seen.contains(name))
                throw std::invalid_argument(key + " is not referenced by " + path("order"));
        }
        return names;
    }

private:
    std::optional<rclcpp::Parameter> find(std::string_view key) const {
        rclcpp::Parameter parameter;
        try {
            if (!node_.get_parameter(path(key), parameter))
                return std::nullopt;
        } catch (const std::exception&) {
            return std::nullopt;
        }
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
            return std::nullopt;
        return parameter;
    }

    std::invalid_argument invalid(std::string_view key, std::string_view reason) const {
        return std::invalid_argument(std::format("rmcs_rl: parameter '{}' {}", path(key), reason));
    }

    rclcpp::Node& node_;
    std::string prefix_;
};

void check_declared_size(
    const Parameters& parameters, std::string_view key, std::size_t actual,
    std::string_view description) {
    const auto declared = parameters.integer(key);
    if (declared && static_cast<std::size_t>(*declared) != actual)
        throw std::invalid_argument(
            std::format("RlBridge: {}={} but {}", key, *declared, description));
}

ActionTerm load_action_term(const Parameters& parameters, const std::string& name) {
    parameters.check_keys({"output", "scale", "offset", "clip"});
    return ActionTerm{
        .name = name,
        .output = parameters.required_string("output"),
        .scale = parameters.number_or("scale", 1.0),
        .offset = parameters.number_or("offset", 0.0),
        .clip = parameters.range("clip"),
    };
}

ObservationTerm load_observation_term(
    const Parameters& parameters, const std::string& name, std::size_t action_size) {
    parameters.check_keys(
        {"source", "sources", "type", "size", "select", "value", "scale", "offset", "clip",
         "default", "precision", "source_clip"});

    ObservationTerm term;
    term.name = name;
    const auto type = parameters.string_or("type", "scalar");
    if (type == "scalars") {
        term.type = ObservationType::kScalars;
        term.sources = parameters.strings("sources");
        if (term.sources.empty()
            || std::ranges::any_of(term.sources, [](const auto& source) { return source.empty(); }))
            throw std::invalid_argument(
                parameters.path("sources") + " requires nonempty interface paths");
        term.size = term.sources.size();
    } else if (type == "constant") {
        term.type = ObservationType::kConstant;
        term.constants = parameters.numbers("value");
        if (term.constants.empty())
            throw std::invalid_argument(parameters.path("value") + " is required");
        term.size = term.constants.size();
    } else if (type == "previous_action") {
        term.type = ObservationType::kPreviousAction;
        term.size = action_size;
    } else {
        term.source = parameters.required_string("source");
        if (type == "scalar") {
            term.type = ObservationType::kScalar;
        } else if (type == "vector3") {
            term.type = ObservationType::kVector3;
            term.size = 3;
        } else if (type == "projected_gravity") {
            term.type = ObservationType::kProjectedGravity;
            term.size = 3;
        } else if (type == "array") {
            term.type = ObservationType::kArray;
            const auto size = parameters.integer("size").value_or(0);
            if (size < 1 || size > 4096)
                throw std::invalid_argument(parameters.path("size") + " must be in [1, 4096]");
            term.size = static_cast<std::size_t>(size);
        } else {
            throw std::invalid_argument(parameters.path("type") + " is not supported: " + type);
        }
    }
    term.source_size = term.size;

    if (const auto selection = parameters.integers("select")) {
        for (const auto index : *selection) {
            if (index < 0 || static_cast<std::size_t>(index) >= term.source_size)
                throw std::invalid_argument(
                    parameters.path("select") + " index is outside source size");
            term.selection.push_back(static_cast<std::size_t>(index));
        }
        term.size = term.selection.size();
    }

    if (const auto value = parameters.number("default")) {
        if (term.type != ObservationType::kScalar || !std::isfinite(*value))
            throw std::invalid_argument(
                parameters.path("default") + " requires a finite scalar source");
        term.default_value = *value;
    }

    term.scale = parameters.number_or("scale", 1.0);
    term.offset = parameters.number_or("offset", 0.0);
    term.clip = parameters.range("clip");
    const auto precision = parameters.string_or("precision", "float64");
    if (precision != "float32" && precision != "float64")
        throw std::invalid_argument(parameters.path("precision") + " must be float32 or float64");
    term.float32 = precision == "float32";
    term.source_clip = parameters.range("source_clip");
    return term;
}

PolicyConfig load_policy_config(const Parameters& parameters, const BridgeConfig& bridge) {
    auto path = parameters.string_or("rl_model_path", "");
    if (path.empty())
        throw std::invalid_argument("required YAML parameter 'rl_model_path' is empty");
    if (path.front() != '/')
        path = ament_index_cpp::get_package_share_directory("rmcs_rl") + "/" + path;

    const double obs_clip = parameters.number_or("obs_clip", -1.0);
    const double action_clip = parameters.number_or("action_clip", -1.0);
    return PolicyConfig{
        .path = std::move(path),
        .model_type = parameters.string_or("model_type", "auto"),
        .sequence_length = bridge.history_length,
        .feature_size = bridge.frame_size,
        .observation_size = bridge.observation_size,
        .action_size = bridge.action_terms.size(),
        .expected_policy_rate = bridge.policy_rate,
        .expected_model_spec_sha256 = parameters.string_or("expected_model_spec_sha256", ""),
        .expected_observation_contract_sha256 =
            parameters.string_or("expected_observation_contract_sha256", ""),
        .expected_observation_version = parameters.string_or("expected_observation_version", ""),
        .expected_obs_normalization = parameters.string_or("expected_obs_normalization", ""),
        .expected_metadata = parameters.child("expected_metadata").string_map(),
        .normalization_from_metadata = parameters.boolean_or("normalization_from_metadata", true),
        .obs_clip = obs_clip >= 0.0 ? std::optional{obs_clip} : std::nullopt,
        .action_clip = action_clip >= 0.0 ? std::optional{action_clip} : std::nullopt,
    };
}

InvalidAction load_invalid_action(const Parameters& parameters) {
    const auto value = parameters.string_or("invalid_value", "nan");
    if (value == "nan")
        return InvalidAction::kNaN;
    if (value == "zero")
        return InvalidAction::kZero;
    if (value == "hold")
        return InvalidAction::kHold;
    throw std::invalid_argument(
        "RlBridge: invalid_value must be \"nan\", \"zero\" or \"hold\", got '" + value + "'");
}

double positive_or(const Parameters& parameters, std::string_view key, double fallback) {
    const double value = parameters.number_or(key, fallback);
    if (!(value > 0.0))
        throw std::invalid_argument(std::format("RlBridge: {} must be finite and > 0", key));
    return value;
}

} // namespace

BridgeConfig load_bridge_config(rclcpp::Node& node) {
    const Parameters parameters{node};
    BridgeConfig config;
    config.rl_base = parameters.string_or("rl_base", "/rl");

    const auto actions = parameters.child("actions");
    std::unordered_set<std::string> outputs;
    for (const auto& name : actions.ordered_names()) {
        auto term = load_action_term(actions.child(name), name);
        if (!outputs.emplace(term.output).second)
            throw std::invalid_argument(
                "RlBridge: two action terms map to the same output interface '" + term.output
                + "'");
        config.action_terms.push_back(std::move(term));
    }
    check_declared_size(
        parameters, "rl_action_size", config.action_terms.size(),
        std::format("actions.order has {} entries", config.action_terms.size()));

    const auto observations = parameters.child("observations");
    for (const auto& name : observations.ordered_names()) {
        auto term =
            load_observation_term(observations.child(name), name, config.action_terms.size());
        term.index = config.frame_size;
        config.frame_size += term.size;
        config.observation_terms.push_back(std::move(term));
    }
    check_declared_size(
        parameters, "rl_feature_size", config.frame_size,
        std::format("observations declare {}", config.frame_size));

    const auto history_length = parameters.integer("history_length").value_or(1);
    if (history_length < 1 || history_length > 64)
        throw std::invalid_argument("RlBridge: history_length must be in [1, 64]");
    config.history_length = static_cast<std::size_t>(history_length);
    config.observation_size = config.frame_size * config.history_length;
    check_declared_size(
        parameters, "rl_obs_size", config.observation_size,
        std::format(
            "observation frame/history contract is {}x{}={}", config.frame_size,
            config.history_length, config.observation_size));

    config.policy_rate = positive_or(parameters, "policy_rate", 50.0);
    config.policy_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / config.policy_rate));
    config.max_action_age = positive_or(parameters, "max_action_age", 2.0 / config.policy_rate);
    config.inference_timeout =
        positive_or(parameters, "inference_timeout", 2.0 / config.policy_rate);
    config.invalid_action = load_invalid_action(parameters);

    config.enable_interface = parameters.string_or("enable_interface", config.rl_base + "/enable");
    config.enable_default = parameters.boolean_or("enable_default", false);
    config.reset_interface = parameters.string_or("reset_interface", "");
    config.frame_sequence_interface =
        parameters.string_or("observation_frame_sequence_interface", "");

    config.policy = load_policy_config(parameters, config);
    return config;
}

} // namespace rmcs_rl
