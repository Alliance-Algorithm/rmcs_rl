#include "policy_worker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rmcs_rl {

PolicyWorker::PolicyWorker(PolicyConfig config)
    : model_(config)
    , pending_observation_(model_.info().obs_size)
    , input_buffer_(model_.info().obs_size)
    , action_buffer_(model_.info().action_size)
    , result_{} {
    result_.action.reserve(model_.info().action_size);
    worker_ = std::jthread([this] { run_(); });
}

PolicyWorker::~PolicyWorker() {
    stopping_.store(true, std::memory_order_release);
    input_ready_.notify_one();
    if (worker_.joinable())
        worker_.join();
}

bool PolicyWorker::healthy() const { return healthy_.load(std::memory_order_acquire); }

std::string PolicyWorker::error() const {
    const std::lock_guard lock{error_mutex_};
    return error_;
}

bool PolicyWorker::try_take(Result& result) {
    const std::unique_lock lock{result_mutex_, std::try_to_lock};
    if (!lock.owns_lock() || !result_ready_)
        return false;
    std::swap(result.sequence, result_.sequence);
    std::swap(result.generation, result_.generation);
    std::swap(result.submitted_at, result_.submitted_at);
    std::swap(result.queued_at, result_.queued_at);
    std::swap(result.inference_started_at, result_.inference_started_at);
    std::swap(result.inference_completed_at, result_.inference_completed_at);
    std::swap(result.ready_at, result_.ready_at);
    std::swap(result.context, result_.context);
    std::swap(result.source_steady_ns, result_.source_steady_ns);
    std::swap(result.inference_microseconds, result_.inference_microseconds);
    std::swap(result.success, result_.success);
    result.error.swap(result_.error);
    result.action.clear();
    result.action.insert(result.action.end(), result_.action.begin(), result_.action.end());
    result_ready_ = false;
    return true;
}

void PolicyWorker::run_() {
    while (true) {
        std::uint64_t sequence;
        std::uint64_t generation;
        std::uint64_t context;
        std::uint64_t source_ns;
        Clock::time_point submitted_at;
        Clock::time_point queued_at;
        {
            std::unique_lock lock{input_mutex_};
            input_ready_.wait(
                lock, [this] { return pending_ || stopping_.load(std::memory_order_acquire); });
            if (stopping_.load(std::memory_order_acquire))
                return;
            std::copy(
                pending_observation_.begin(), pending_observation_.end(), input_buffer_.begin());
            sequence = pending_sequence_;
            generation = pending_generation_;
            submitted_at = pending_submitted_at_;
            queued_at = pending_queued_at_;
            context = pending_context_;
            source_ns = pending_source_ns_;
            pending_ = false;
        }

        std::string error;
        const auto started = Clock::now();
        const bool success = model_.run(input_buffer_, action_buffer_, error);
        const auto completed = Clock::now();

        {
            if (!success) {
                healthy_.store(false, std::memory_order_release);
                const std::lock_guard error_lock{error_mutex_};
                error_ = error;
            }
        }
        {
            std::lock_guard lock{result_mutex_};
            if (result_ready_)
                ++result_overwritten_;
            result_.sequence = sequence;
            result_.generation = generation;
            result_.submitted_at = submitted_at;
            result_.queued_at = queued_at;
            result_.inference_started_at = started;
            result_.inference_completed_at = completed;
            result_.context = context;
            result_.source_steady_ns = source_ns;
            result_.inference_microseconds =
                std::chrono::duration<double, std::micro>{completed - started}.count();
            result_.success = success;
            result_.error = std::move(error);
            if (success) {
                result_.action.clear();
                result_.action.insert(
                    result_.action.end(), action_buffer_.begin(), action_buffer_.end());
            }
            result_.ready_at = Clock::now();
            result_ready_ = true;
        }
        if (!success)
            return;
    }
}

} // namespace rmcs_rl
