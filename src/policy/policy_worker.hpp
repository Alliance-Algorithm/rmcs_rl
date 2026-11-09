#pragma once

#include <algorithm>
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
        Clock::time_point queued_at{}, inference_started_at{}, inference_completed_at{}, ready_at{};
        std::uint64_t context = 0;
        std::uint64_t source_steady_ns = 0;
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
    std::uint64_t submit_attempts() const {
        return submit_attempts_.load(std::memory_order_relaxed);
    }
    std::uint64_t submit_rejected() const {
        return submit_rejected_.load(std::memory_order_relaxed);
    }
    std::uint64_t pending_overwritten() const {
        return pending_overwritten_.load(std::memory_order_relaxed);
    }
    std::uint64_t result_overwritten() const {
        return result_overwritten_.load(std::memory_order_relaxed);
    }

    template <std::floating_point T>
    bool submit(
        std::span<const T> observation, std::uint64_t sequence, std::uint64_t generation,
        Clock::time_point submitted_at, std::uint64_t context = 0,
        std::uint64_t source_steady_ns = 0) {
        ++submit_attempts_;
        if (observation.size() != pending_observation_.size()) {
            ++submit_rejected_;
            return false;
        }
        const std::unique_lock lock{input_mutex_, std::try_to_lock};
        if (!lock.owns_lock() || stopping_.load(std::memory_order_acquire)
            || !healthy_.load(std::memory_order_acquire)) {
            ++submit_rejected_;
            return false;
        }
        if (pending_)
            ++pending_overwritten_;
        std::transform(
            observation.begin(), observation.end(), pending_observation_.begin(),
            [](T value) { return static_cast<double>(value); });
        pending_sequence_ = sequence;
        pending_generation_ = generation;
        pending_submitted_at_ = submitted_at;
        pending_context_ = context;
        pending_source_ns_ = source_steady_ns;
        pending_queued_at_ = Clock::now();
        pending_ = true;
        input_ready_.notify_one();
        return true;
    }
    bool try_take(Result& result);

private:
    void run_();

    PolicyModel model_;
    std::vector<double> pending_observation_;
    std::vector<double> input_buffer_;
    std::vector<double> action_buffer_;
    std::uint64_t pending_sequence_ = 0;
    std::uint64_t pending_generation_ = 0;
    Clock::time_point pending_submitted_at_{};
    Clock::time_point pending_queued_at_{};
    std::uint64_t pending_context_ = 0;
    std::uint64_t pending_source_ns_ = 0;
    bool pending_ = false;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> healthy_{true};
    std::atomic<std::uint64_t> submit_attempts_{0}, submit_rejected_{0};
    std::atomic<std::uint64_t> pending_overwritten_{0}, result_overwritten_{0};
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
