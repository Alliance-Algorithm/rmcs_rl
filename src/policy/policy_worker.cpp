#include "policy_worker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rmcs_rl {

PolicyWorker::PolicyWorker(PolicyConfig config)
    : model_(config)
    , pending_observation_(model_.info().obs_size)
    , pending_action_(model_.info().action_size)
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
    std::swap(result.completed_at, result_.completed_at);
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
        Clock::time_point submitted_at;
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
            context = pending_context_;
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
            result_.sequence = sequence;
            result_.generation = generation;
            result_.submitted_at = submitted_at;
            result_.context = context;
            result_.completed_at = completed;
            result_.inference_microseconds =
                std::chrono::duration<double, std::micro>{completed - started}.count();
            result_.success = success;
            result_.error = std::move(error);
            if (success) {
                result_.action.clear();
                result_.action.insert(
                    result_.action.end(), action_buffer_.begin(), action_buffer_.end());
            }
            result_ready_ = true;
        }
        if (!success)
            return;
    }
}

} // namespace rmcs_rl
