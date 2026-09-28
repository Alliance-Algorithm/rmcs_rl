#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <rclcpp/node.hpp>
#include <rmcs_rl/rl_bridge/joint_config.hpp>
#include <rmcs_rl/rl_bridge/types.hpp>

namespace rmcs_rl {

struct BridgeConfig {
    JointConfig joint_config;
    std::vector<ObsTerm> obs_terms;
    std::vector<ActionTerm> action_terms;
    std::size_t obs_size = 0;
    std::size_t obs_frame_size = 0;
    std::size_t history_length = 1;
    std::size_t action_size = 0;
    std::string rl_base;
    double policy_rate = 50.0;
    double max_action_age = 0.04;
    std::chrono::nanoseconds pub_period{};
    std::uint64_t expected_model_id = 0;
    InvalidMode invalid_mode = InvalidMode::kNaN;
    std::string enable_path;
    bool enable_default = false;
    std::string reset_path;
};

BridgeConfig load_bridge_config(rclcpp::Node& node);

} // namespace rmcs_rl
