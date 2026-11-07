#include <rmcs_rl/rl_bridge/structured_config.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <rmcs_rl/parameters.hpp>

namespace rmcs_rl {
namespace {

std::vector<double> numbers(rclcpp::Node& node, const std::string& key) {
    rclcpp::Parameter value;
    if (!node.get_parameter(key, value))
        return {};
    std::vector<double> result;
    if (value.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY)
        result = value.as_double_array();
    else if (value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
        for (const auto item : value.as_integer_array())
            result.push_back(static_cast<double>(item));
    } else
        throw std::invalid_argument(key + " must be a numeric array");
    if (result.empty() || !std::ranges::all_of(result, [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument(key + " must be a nonempty finite numeric array");
    return result;
}

std::vector<std::string> order(rclcpp::Node& node, const std::string& prefix) {
    auto names = string_array_or(node, prefix + ".order");
    if (names.empty())
        throw std::invalid_argument(prefix + ".order must not be empty");
    std::unordered_set<std::string> seen;
    for (const auto& name : names)
        if (name.empty()
            || name.find_first_not_of(
                   "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                   != std::string::npos
            || !seen.emplace(name).second)
            throw std::invalid_argument(
                prefix + ".order contains invalid or duplicate name: " + name);
    for (const auto& key : node.list_parameters({prefix}, 0).names) {
        const auto suffix = key.substr(prefix.size() + 1);
        const auto name = suffix.substr(0, suffix.find('.'));
        if (name != "order" && !seen.contains(name))
            throw std::invalid_argument(key + " is not referenced by " + prefix + ".order");
    }
    return names;
}

void check_keys(
    rclcpp::Node& node, const std::string& prefix, const std::set<std::string>& allowed) {
    for (const auto& key : node.list_parameters({prefix}, 0).names)
        if (!allowed.contains(key.substr(prefix.size() + 1)))
            throw std::invalid_argument("unknown policy parameter: " + key);
}

template <class Term>
void transform(rclcpp::Node& node, const std::string& prefix, Term& term) {
    term.scale = number_or(node, prefix + ".scale", 1.0);
    term.offset = number_or(node, prefix + ".offset", 0.0);
    const auto clip = numbers(node, prefix + ".clip");
    if (!clip.empty()) {
        if (clip.size() != 2 || clip[0] > clip[1])
            throw std::invalid_argument(prefix + ".clip must be [lower, upper]");
        term.has_clip = true;
        term.clip_min = clip[0];
        term.clip_max = clip[1];
    }
}

} // namespace

std::vector<ActionTerm> load_action_terms(rclcpp::Node& node) {
    std::vector<ActionTerm> terms;
    for (const auto& name : order(node, "actions")) {
        const auto prefix = "actions." + name;
        check_keys(node, prefix, {"output", "scale", "offset", "clip"});
        ActionTerm term;
        term.id = name;
        term.index = terms.size();
        term.output = string_or(node, prefix + ".output", "");
        if (term.output.empty())
            throw std::invalid_argument(prefix + ".output is required");
        transform(node, prefix, term);
        terms.push_back(std::move(term));
    }
    return terms;
}

std::vector<ObsTerm> load_observation_terms(rclcpp::Node& node) {
    std::vector<ObsTerm> terms;
    for (const auto& name : order(node, "observations")) {
        const auto prefix = "observations." + name;
        check_keys(
            node, prefix,
            {"source", "sources", "type", "size", "select", "value", "scale", "offset", "clip",
             "default", "precision", "source_clip"});
        ObsTerm term;
        term.id = name;
        const auto type = string_or(node, prefix + ".type", "scalar");
        if (type == "scalars") {
            term.kind = TermKind::kSources;
            term.sources = string_array_or(node, prefix + ".sources");
            if (term.sources.empty()
                || std::ranges::any_of(term.sources, [](const auto& s) { return s.empty(); }))
                throw std::invalid_argument(prefix + ".sources requires nonempty interface paths");
            term.dim = term.sources.size();
        } else if (type == "constant") {
            term.kind = TermKind::kConstant;
            term.constants = numbers(node, prefix + ".value");
            if (term.constants.empty())
                throw std::invalid_argument(prefix + ".value is required");
            term.dim = term.constants.size();
        } else if (type == "previous_action") {
            term.kind = TermKind::kLastAction;
            const auto actions = string_array_or(node, "actions.order");
            for (std::size_t i = 0; i < actions.size(); ++i)
                term.action_indices.push_back(i);
            term.dim = actions.size();
            if (term.dim == 0)
                throw std::invalid_argument(prefix + " requires actions.order");
        } else {
            term.path = string_or(node, prefix + ".source", "");
            if (term.path.empty())
                throw std::invalid_argument(prefix + ".source is required");
            if (type == "scalar")
                term.take = Take::kScalar;
            else if (type == "vector3") {
                term.take = Take::kVector;
                term.dim = 3;
            } else if (type == "array") {
                term.take = Take::kArray;
                const auto size = integer_parameter(node, prefix + ".size").value_or(0);
                if (size < 1 || size > 4096)
                    throw std::invalid_argument(prefix + ".size must be in [1, 4096]");
                term.dim = static_cast<std::size_t>(size);
            } else if (type == "projected_gravity") {
                term.take = Take::kGravity;
                term.dim = 3;
            } else
                throw std::invalid_argument(prefix + ".type is not supported: " + type);
        }
        term.source_size = term.dim;
        rclcpp::Parameter selection;
        if (node.get_parameter(prefix + ".select", selection)) {
            if (selection.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY
                || selection.as_integer_array().empty())
                throw std::invalid_argument(prefix + ".select must be a nonempty integer array");
            for (const auto index : selection.as_integer_array()) {
                if (index < 0 || static_cast<std::size_t>(index) >= term.source_size)
                    throw std::invalid_argument(prefix + ".select index is outside source size");
                term.selection.push_back(static_cast<std::size_t>(index));
            }
            term.dim = term.selection.size();
        }
        if (const auto value = number_parameter(node, prefix + ".default")) {
            if (term.kind != TermKind::kPath || term.take != Take::kScalar
                || !std::isfinite(*value))
                throw std::invalid_argument(prefix + ".default requires a finite scalar source");
            term.has_default = true;
            term.default_value = *value;
        }
        transform(node, prefix, term);
        const auto precision = string_or(node, prefix + ".precision", "float64");
        if (precision != "float32" && precision != "float64")
            throw std::invalid_argument(prefix + ".precision must be float32 or float64");
        term.float32 = precision == "float32";
        const auto source_clip = numbers(node, prefix + ".source_clip");
        if (!source_clip.empty()) {
            if (source_clip.size() != 2 || source_clip[0] > source_clip[1])
                throw std::invalid_argument(prefix + ".source_clip must be [lower, upper]");
            term.has_source_clip = true;
            term.source_clip_min = source_clip[0];
            term.source_clip_max = source_clip[1];
        }
        terms.push_back(std::move(term));
    }
    return terms;
}

} // namespace rmcs_rl
