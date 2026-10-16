#pragma once

#include <cstddef>

#include <rclcpp/node.hpp>

#include "policy_model.hpp"

namespace rmcs_rl {

PolicyConfig load_policy_config(
    rclcpp::Node& node, std::size_t observation_size, std::size_t action_size,
    std::size_t sequence_length, std::size_t feature_size);

} // namespace rmcs_rl
