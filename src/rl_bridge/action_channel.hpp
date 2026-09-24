#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rmcs_executor/component.hpp>
#include <rmcs_rl/msg/action.hpp>

#include "rl_bridge/types.hpp"

namespace rmcs_rl {

class ActionChannel {
public:
    void resize(std::size_t action_size) { incoming_.action.assign(action_size, 0.0); }

    void store(const rmcs_rl::msg::Action& message) {
        const auto received = std::chrono::steady_clock::now();

        const std::uint64_t sequence = action_sequence_.load(std::memory_order_relaxed);
        action_sequence_.store(sequence + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);

        incoming_.obs_seq = message.obs_seq;
        incoming_.layout_hash = message.layout_hash;
        incoming_.model_id = message.model_id;
        incoming_.received = received;
        std::copy(message.action.begin(), message.action.end(), incoming_.action.begin());

        std::atomic_thread_fence(std::memory_order_release);
        action_sequence_.store(sequence + 2, std::memory_order_relaxed);
    }

    bool try_read(ActionSnapshot& snapshot) {
        for (int attempt = 0; attempt < 4; ++attempt) {
            const std::uint64_t before = action_sequence_.load(std::memory_order_relaxed);
            if (before == 0 || (before & 1U) != 0)
                return false;
            std::atomic_thread_fence(std::memory_order_acquire);
            snapshot = incoming_;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (action_sequence_.load(std::memory_order_relaxed) == before)
                return true;
        }
        return false;
    }

private:
    ActionSnapshot incoming_;
    alignas(64) std::atomic<std::uint64_t> action_sequence_{0};
};

void write_actions(
    bool valid, const ActionSnapshot& snapshot, const std::vector<ActionTerm>& terms,
    InvalidMode invalid_mode, std::vector<double>& written,
    std::vector<std::unique_ptr<rmcs_executor::Component::OutputInterface<double>>>& outputs);

void reset_action_state(std::vector<double>& last_actions, std::vector<double>& written);

std::string invalid_reason(
    bool enabled, bool contract_ok, bool has_snapshot, bool fresh, bool seq_ok, bool finite);

} // namespace rmcs_rl
