#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <vector>

#include <rmcs_executor/component.hpp>

#include <rmcs_rl/rl_bridge/types.hpp>

namespace rmcs_rl {

const char* binding_name(Binding binding);
bool binding_matches_type(Binding binding, const std::type_info& type);

std::size_t acquire_slot(
    rmcs_executor::Component& component, const std::string& path,
    const std::vector<Binding>& candidates, bool required,
    const rmcs_executor::Component::OutputInfoMap& output_map, std::vector<Slot>& slots,
    const char* context);

bool read_double(const Slot& slot, double& value);
bool read_unsigned(const Slot& slot, std::uint64_t& value);

} // namespace rmcs_rl
