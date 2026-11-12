#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/node.hpp>

#include "policy/policy_config.hpp"

namespace rmcs_rl {

struct Range {
    double lower;
    double upper;
};

enum class ObservationType {
    kScalar,
    kVector3,
    kArray,
    kProjectedGravity,
    kScalars,
    kConstant,
    kPreviousAction,
};

struct ObservationTerm {
    std::string name;
    ObservationType type = ObservationType::kScalar;
    std::string source;
    std::vector<std::string> sources;
    std::vector<double> constants;
    std::optional<double> default_value;

    std::size_t source_size = 1;
    std::vector<std::size_t> selection;
    std::size_t index = 0;
    std::size_t size = 1;

    bool float32 = false;
    std::optional<Range> source_clip;
    double scale = 1.0;
    double offset = 0.0;
    std::optional<Range> clip;
};

struct ActionTerm {
    std::string name;
    std::string output;
    double scale = 1.0;
    double offset = 0.0;
    std::optional<Range> clip;
};

enum class InvalidAction { kNaN, kZero, kHold };

struct BridgeConfig {
    std::string rl_base;
    std::vector<ObservationTerm> observation_terms;
    std::vector<ActionTerm> action_terms;

    std::size_t frame_size = 0;
    std::size_t history_length = 1;
    std::size_t observation_size = 0;

    double policy_rate = 50.0;
    std::chrono::nanoseconds policy_period{};
    double max_action_age = 0.0;
    double inference_timeout = 0.0;
    InvalidAction invalid_action = InvalidAction::kNaN;

    std::string enable_interface;
    bool enable_default = false;
    std::string reset_interface;
    std::string frame_sequence_interface;

    PolicyConfig policy;
};

BridgeConfig load_bridge_config(rclcpp::Node& node);

} // namespace rmcs_rl
