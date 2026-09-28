#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rmcs_executor/component.hpp>
#include <rmcs_rl/msg/action.hpp>

#include <rmcs_rl/rl_bridge/types.hpp>

namespace rmcs_rl {

class ActionChannel {
public:
    void resize(std::size_t action_size);
    void store(const rmcs_rl::msg::Action& message);
    bool try_read(ActionSnapshot& snapshot);

private:
    ActionSnapshot incoming_;
    std::mutex mutex_;
    bool received_ = false;
};

void write_actions(
    bool valid, const ActionSnapshot& snapshot, const std::vector<ActionTerm>& terms,
    InvalidMode invalid_mode, std::vector<double>& written,
    std::vector<std::unique_ptr<rmcs_executor::Component::OutputInterface<double>>>& outputs);

void reset_action_state(std::vector<double>& last_actions, std::vector<double>& written);

std::string invalid_reason(
    bool enabled, bool contract_ok, bool has_snapshot, bool fresh, bool seq_ok, bool finite);

} // namespace rmcs_rl
