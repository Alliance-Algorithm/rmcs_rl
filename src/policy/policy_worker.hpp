#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "policy/policy_model.hpp"

namespace rmcs_rl {

class PolicyWorker {
public:
    using Clock = std::chrono::steady_clock;
    struct Result {
        std::uint64_t sequence = 0, generation = 0;
        Clock::time_point submitted_at{};
        bool success = false;
        std::string error;
        std::vector<double> action;
    };

    PolicyWorker(
        const std::string& path, std::size_t history_length, std::size_t frame_size,
        std::size_t action_size, const std::map<std::string, std::string>& expected_metadata);

    bool healthy() const { return healthy_.load(std::memory_order::acquire); }
    const std::map<std::string, std::string>& metadata() const { return model_.metadata(); }
    void submit(
        std::span<const double> observation, std::uint64_t sequence, std::uint64_t generation,
        Clock::time_point submitted_at);
    bool try_take(Result& result);
    void reset();

private:
    void run(const std::stop_token& stop);

    PolicyModel model_;
    std::mutex mutex_;
    std::condition_variable_any ready_;
    std::vector<double> pending_observation_, observation_, action_;
    Result pending_{}, result_{};
    bool has_pending_ = false, has_result_ = false;
    std::atomic<bool> healthy_{true};
    std::jthread thread_;
};

} // namespace rmcs_rl
