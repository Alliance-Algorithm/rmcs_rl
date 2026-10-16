#pragma once

#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "policy_model.hpp"

namespace rmcs_rl {

class PolicyWorker {
public:
    using Clock = std::chrono::steady_clock;

    struct Result {
        std::uint64_t sequence = 0;
        std::uint64_t generation = 0;
        Clock::time_point submitted_at{};
        std::uint64_t context = 0;
        Clock::time_point completed_at{};
        double inference_microseconds = 0.0;
        bool success = false;
        std::string error;
        std::vector<double> action;
    };

    explicit PolicyWorker(PolicyConfig config);
    ~PolicyWorker();

    PolicyWorker(const PolicyWorker&) = delete;
    PolicyWorker& operator=(const PolicyWorker&) = delete;

    [[nodiscard]] const PolicyModel::Info& info() const { return model_.info(); }
    [[nodiscard]] bool healthy() const;
    [[nodiscard]] std::string error() const;

    template <std::floating_point T>
    bool submit(
        std::span<const T> observation, std::uint64_t sequence, std::uint64_t generation,
        Clock::time_point submitted_at, std::uint64_t context = 0) {
        if (observation.size() != pending_observation_.size())
            return false;
        const std::unique_lock lock{input_mutex_, std::try_to_lock};
        if (!lock.owns_lock() || stopping_.load(std::memory_order_acquire)
            || !healthy_.load(std::memory_order_acquire))
            return false;
        std::transform(
            observation.begin(), observation.end(), pending_observation_.begin(),
            [](T value) { return static_cast<double>(value); });
        pending_sequence_ = sequence;
        pending_generation_ = generation;
        pending_submitted_at_ = submitted_at;
        pending_context_ = context;
        pending_ = true;
        input_ready_.notify_one();
        return true;
    }
    bool try_take(Result& result);

private:
    void run_();

    PolicyModel model_;
    std::vector<double> pending_observation_;
    std::vector<double> pending_action_;
    std::vector<double> input_buffer_;
    std::vector<double> action_buffer_;
    std::uint64_t pending_sequence_ = 0;
    std::uint64_t pending_generation_ = 0;
    Clock::time_point pending_submitted_at_{};
    std::uint64_t pending_context_ = 0;
    bool pending_ = false;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> healthy_{true};
    std::string error_;
    mutable std::mutex input_mutex_;
    mutable std::mutex error_mutex_;
    std::mutex result_mutex_;
    std::condition_variable input_ready_;
    Result result_;
    bool result_ready_ = false;
    std::jthread worker_;
};

} // namespace rmcs_rl
