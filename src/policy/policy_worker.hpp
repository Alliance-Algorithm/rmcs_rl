#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <expected>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "policy/policy_config.hpp"
#include "policy/policy_model.hpp"

namespace rmcs_rl {

class PolicyWorker {
public:
    using Clock = std::chrono::steady_clock;

    struct Result {
        std::uint64_t sequence = 0;
        std::uint64_t generation = 0;
        Clock::time_point submitted_at{};
        Clock::time_point queued_at{};
        Clock::time_point inference_started_at{};
        Clock::time_point inference_completed_at{};
        Clock::time_point ready_at{};
        bool success = false;
        std::string error;
        std::vector<double> action;
    };

    explicit PolicyWorker(const PolicyConfig& config);

    PolicyWorker(const PolicyWorker&) = delete;
    PolicyWorker& operator=(const PolicyWorker&) = delete;
    PolicyWorker(PolicyWorker&&) = delete;
    PolicyWorker& operator=(PolicyWorker&&) = delete;

    ~PolicyWorker() = default;

    const PolicyModel::Info& info() const { return model_.info(); }
    bool healthy() const { return healthy_.load(std::memory_order::acquire); }

    std::uint64_t submit_attempts() const { return submit_attempts_.load(); }
    std::uint64_t submit_rejected() const { return submit_rejected_.load(); }
    std::uint64_t pending_overwritten() const { return pending_overwritten_.load(); }
    std::uint64_t result_overwritten() const { return result_overwritten_.load(); }

    bool submit(
        std::span<const double> observation, std::uint64_t sequence, std::uint64_t generation,
        Clock::time_point submitted_at);
    bool try_take(Result& result);

    // Drops pending and in-flight work and clears a latched inference failure.
    void reset();

private:
    struct Request {
        std::uint64_t sequence = 0;
        std::uint64_t generation = 0;
        Clock::time_point submitted_at{};
        Clock::time_point queued_at{};
    };

    void run(const std::stop_token& stop);
    void store_result(
        const Request& request, Clock::time_point started, Clock::time_point completed,
        std::expected<void, std::string> outcome);

    PolicyModel model_;

    std::mutex input_mutex_;
    std::condition_variable_any input_ready_;
    std::vector<double> pending_observation_;
    Request pending_request_;
    std::uint64_t reset_epoch_ = 0;
    bool pending_ = false;

    std::mutex result_mutex_;
    Result result_;
    bool result_ready_ = false;

    std::atomic<bool> healthy_{true};
    std::atomic<std::uint64_t> submit_attempts_{0};
    std::atomic<std::uint64_t> submit_rejected_{0};
    std::atomic<std::uint64_t> pending_overwritten_{0};
    std::atomic<std::uint64_t> result_overwritten_{0};

    std::vector<double> input_buffer_;
    std::vector<double> action_buffer_;

    // Declared last: stops and joins before the state above is destroyed.
    std::jthread thread_;
};

} // namespace rmcs_rl
