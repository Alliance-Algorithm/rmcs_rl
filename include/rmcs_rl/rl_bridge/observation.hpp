#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <rmcs_rl/rl_bridge/types.hpp>

namespace rmcs_rl {

bool build_observation(
    const std::vector<ObsTerm>& terms, const std::vector<Slot>& slots,
    const std::vector<double>& last_actions, std::size_t obs_size, std::vector<double>& obs);

} // namespace rmcs_rl
