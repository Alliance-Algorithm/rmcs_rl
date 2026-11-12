#include "policy/policy_worker.hpp"

#include <algorithm>
#include <utility>

namespace rmcs_rl {

PolicyWorker::PolicyWorker(const PolicyConfig& config)
    : model_(config)
    , pending_observation_(model_.info().observation_size)
    , input_buffer_(model_.info().observation_size)
    , action_buffer_(model_.info().action_size) {
    result_.action.reserve(model_.info().action_size);
    thread_ = std::jthread{[this](const std::stop_token& stop) { run(stop); }};
}

bool PolicyWorker::submit(
    std::span<const double> observation, std::uint64_t sequence, std::uint64_t generation,
    Clock::time_point submitted_at) {
    ++submit_attempts_;
    if (observation.size() != pending_observation_.size()) [[unlikely]] {
        ++submit_rejected_;
        return false;
    }

    const std::unique_lock lock{input_mutex_, std::try_to_lock};
    if (!lock.owns_lock() || !healthy_.load(std::memory_order::acquire)) {
        ++submit_rejected_;
        return false;
    }
    if (pending_)
        ++pending_overwritten_;
    std::ranges::copy(observation, pending_observation_.begin());
    pending_request_ = {
        .sequence = sequence,
        .generation = generation,
        .submitted_at = submitted_at,
        .queued_at = Clock::now(),
    };
    pending_ = true;
    input_ready_.notify_one();
    return true;
}

bool PolicyWorker::try_take(Result& result) {
    const std::unique_lock lock{result_mutex_, std::try_to_lock};
    if (!lock.owns_lock() || !result_ready_)
        return false;
    std::swap(result, result_);
    result_ready_ = false;
    return true;
}

void PolicyWorker::reset() {
    {
        const std::lock_guard lock{input_mutex_};
        ++reset_epoch_;
        pending_ = false;
        healthy_.store(true, std::memory_order::release);
    }
    input_ready_.notify_one();
}

void PolicyWorker::run(const std::stop_token& stop) {
    while (true) {
        Request request;
        std::uint64_t reset_epoch;
        {
            std::unique_lock lock{input_mutex_};
            if (!input_ready_.wait(lock, stop, [this] { return pending_; })
                || stop.stop_requested())
                return;
            std::ranges::copy(pending_observation_, input_buffer_.begin());
            request = pending_request_;
            reset_epoch = reset_epoch_;
            pending_ = false;
        }

        const auto started = Clock::now();
        auto outcome = model_.run(input_buffer_, action_buffer_);
        const auto completed = Clock::now();

        const bool success = outcome.has_value();
        {
            const std::lock_guard lock{input_mutex_};
            if (reset_epoch != reset_epoch_)
                continue;
            if (!success)
                healthy_.store(false, std::memory_order::release);
        }
        store_result(request, started, completed, std::move(outcome));

        if (!success) {
            std::unique_lock lock{input_mutex_};
            if (!input_ready_.wait(
                    lock, stop, [this] { return healthy_.load(std::memory_order::acquire); }))
                return;
        }
    }
}

void PolicyWorker::store_result(
    const Request& request, Clock::time_point started, Clock::time_point completed,
    std::expected<void, std::string> outcome) {
    const std::lock_guard lock{result_mutex_};
    if (result_ready_)
        ++result_overwritten_;
    result_.sequence = request.sequence;
    result_.generation = request.generation;
    result_.submitted_at = request.submitted_at;
    result_.queued_at = request.queued_at;
    result_.inference_started_at = started;
    result_.inference_completed_at = completed;
    result_.success = outcome.has_value();
    if (result_.success) {
        result_.error.clear();
        result_.action.assign(action_buffer_.begin(), action_buffer_.end());
    } else {
        result_.error = std::move(outcome.error());
        result_.action.clear();
    }
    result_.ready_at = Clock::now();
    result_ready_ = true;
}

} // namespace rmcs_rl
