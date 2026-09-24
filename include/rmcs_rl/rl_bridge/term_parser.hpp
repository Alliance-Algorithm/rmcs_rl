#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <rclcpp/node.hpp>

#include <rmcs_rl/rl_bridge/joint_config.hpp>
#include <rmcs_rl/rl_bridge/types.hpp>

namespace rmcs_rl {

struct TermParseContext {
    rclcpp::Node& node;
    const JointConfig& joints;
    std::size_t action_size;
};

ObsTerm parse_obs_term(const std::string& spec, const TermParseContext& context);
ActionTerm parse_action_term(const std::string& spec);
void assign_observation_indices(std::vector<ObsTerm>& terms);

} // namespace rmcs_rl
