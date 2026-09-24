#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rl_bridge/types.hpp"

namespace rmcs_rl {

bool build_observation(
    const std::vector<ObsTerm>& terms, const std::vector<Slot>& slots,
    const std::vector<double>& last_actions, std::size_t obs_size, std::vector<double>& obs);

std::string obs_layout_signature(const std::vector<ObsTerm>& terms, std::size_t history_length);
std::string actions_layout_signature(const std::vector<ActionTerm>& terms);

} // namespace rmcs_rl
