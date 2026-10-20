#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/node.hpp>
#include <rmcs_rl/policy_config.hpp>
#include <rmcs_rl/rl_bridge/joint_config.hpp>
#include <rmcs_rl/rl_bridge/types.hpp>

namespace rmcs_rl {

class PolicyWorker;

struct PolicyIoConfig {
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
    std::chrono::nanoseconds history_period{};
};

struct BridgeConfig : PolicyIoConfig {
    InvalidMode invalid_mode = InvalidMode::kNaN;
    std::string enable_path;
    bool enable_default = false;
    std::string reset_path;
    std::string flat_hold_path;
    std::string context_path;
    double inference_timeout = 0.0;
    std::shared_ptr<PolicyWorker> policy_worker;
};

PolicyIoConfig load_policy_io_config(rclcpp::Node& node);
BridgeConfig load_bridge_config(rclcpp::Node& node);

} // namespace rmcs_rl
