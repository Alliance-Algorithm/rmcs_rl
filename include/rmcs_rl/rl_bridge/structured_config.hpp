#pragma once

#include <vector>

#include <rclcpp/node.hpp>

#include <rmcs_rl/rl_bridge/types.hpp>

namespace rmcs_rl {

std::vector<ObsTerm> load_observation_terms(rclcpp::Node& node);
std::vector<ActionTerm> load_action_terms(rclcpp::Node& node);

} // namespace rmcs_rl
