#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/node.hpp>

namespace rmcs_rl {

struct Range {
    double lower, upper;
};

enum class ObservationType { kScalar, kVector3, kArray, kScalars, kPreviousAction };

struct ObservationTerm {
    std::string name;
    ObservationType type = ObservationType::kScalar;
    std::vector<std::string> sources;
    std::size_t source_size = 1;
    std::vector<std::size_t> selection;
    std::size_t index = 0, size = 1;
    bool float32 = false;
    std::optional<Range> source_clip;
    double scale = 1.;
    std::optional<Range> clip;
};

struct ActionTerm {
    std::string name, output;
    double scale = 1.;
    std::optional<Range> clip;
};

inline std::optional<Range> load_range(rclcpp::Node& node, const std::string& name) {
    if (!node.has_parameter(name))
        return std::nullopt;
    const auto values = node.get_parameter(name).as_double_array();
    return Range{values.at(0), values.at(1)};
}

inline std::vector<ActionTerm> load_action_terms(rclcpp::Node& node) {
    std::vector<ActionTerm> terms;
    const auto names = node.get_parameter("actions.order").as_string_array();
    for (const auto& name : names) {
        const auto prefix = "actions." + name + ".";
        terms.push_back(
            {.name = name,
             .output = node.get_parameter(prefix + "output").as_string(),
             .scale = node.get_parameter_or<double>(prefix + "scale", 1.),
             .clip = load_range(node, prefix + "clip")});
    }
    return terms;
}

inline std::vector<ObservationTerm>
    load_observation_terms(rclcpp::Node& node, std::size_t action_size) {
    static const std::map<std::string, ObservationType> kTypes{
        {"scalar", ObservationType::kScalar},
        {"vector3", ObservationType::kVector3},
        {"array", ObservationType::kArray},
        {"scalars", ObservationType::kScalars},
        {"previous_action", ObservationType::kPreviousAction}};
    std::vector<ObservationTerm> terms;
    const auto names = node.get_parameter("observations.order").as_string_array();
    for (const auto& name : names) {
        const auto prefix = "observations." + name + ".";
        ObservationTerm term;
        term.name = name;
        term.type = kTypes.at(node.get_parameter_or<std::string>(prefix + "type", "scalar"));
        if (term.type == ObservationType::kPreviousAction) {
            term.source_size = action_size;
        } else if (term.type == ObservationType::kScalars) {
            term.sources = node.get_parameter(prefix + "sources").as_string_array();
            term.source_size = term.sources.size();
        } else {
            term.sources = {node.get_parameter(prefix + "source").as_string()};
            if (term.type == ObservationType::kVector3)
                term.source_size = 3;
            else if (term.type == ObservationType::kArray)
                term.source_size =
                    static_cast<std::size_t>(node.get_parameter(prefix + "size").as_int());
        }
        if (node.has_parameter(prefix + "select")) {
            const auto selection = node.get_parameter(prefix + "select").as_integer_array();
            for (const auto index : selection)
                term.selection.push_back(static_cast<std::size_t>(index));
        }
        term.size = term.selection.empty() ? term.source_size : term.selection.size();
        term.float32 =
            node.get_parameter_or<std::string>(prefix + "precision", "float64") == "float32";
        term.source_clip = load_range(node, prefix + "source_clip");
        term.scale = node.get_parameter_or<double>(prefix + "scale", 1.);
        term.clip = load_range(node, prefix + "clip");
        terms.push_back(std::move(term));
    }
    return terms;
}

inline std::map<std::string, std::string> load_expected_metadata(rclcpp::Node& node) {
    std::map<std::string, std::string> values;
    for (const auto& name : node.list_parameters({"expected_metadata"}, 1).names)
        values.emplace(
            name.substr(std::string{"expected_metadata."}.size()),
            node.get_parameter(name).as_string());
    return values;
}

} // namespace rmcs_rl
