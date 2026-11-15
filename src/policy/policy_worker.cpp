#include "policy/policy_worker.hpp"

#include <algorithm>
#include <exception>
#include <utility>

namespace rmcs_rl {

PolicyWorker::PolicyWorker(
    const std::string& path, std::size_t history_length, std::size_t frame_size,
    std::size_t action_size, const std::map<std::string, std::string>& expected_metadata)
    : model_(path, history_length, frame_size, action_size, expected_metadata)
    , pending_observation_(history_length * frame_size)
    , observation_(history_length * frame_size)
    , action_(action_size) {
    result_.action.resize(action_size);
    thread_ = std::jthread{[this](const std::stop_token& stop) { run(stop); }};
}

void PolicyWorker::submit(
    std::span<const double> observation, std::uint64_t sequence, std::uint64_t generation) {
    {
        const std::lock_guard lock{mutex_};
        std::ranges::copy(observation, pending_observation_.begin());
        pending_.sequence = sequence;
        pending_.generation = generation;
        pending_.submitted_at = Clock::now();
        has_pending_ = true;
    }
    ready_.notify_one();
}

bool PolicyWorker::try_take(Result& result) {
    const std::unique_lock lock{mutex_, std::try_to_lock};
    if (!lock.owns_lock() || !has_result_)
        return false;
    std::swap(result, result_);
    has_result_ = false;
    return true;
}

void PolicyWorker::reset() {
    {
        const std::lock_guard lock{mutex_};
        pending_.sequence = 0;
        has_pending_ = has_result_ = false;
        healthy_.store(true, std::memory_order::release);
    }
    ready_.notify_one();
}

void PolicyWorker::run(const std::stop_token& stop) {
    while (true) {
        Result request;
        {
            std::unique_lock lock{mutex_};
            if (!ready_.wait(lock, stop, [this] { return has_pending_ && healthy(); }))
                return;
            std::ranges::copy(pending_observation_, observation_.begin());
            request.sequence = pending_.sequence;
            request.generation = pending_.generation;
            request.submitted_at = pending_.submitted_at;
            has_pending_ = false;
        }
        try {
            model_.run(observation_, action_);
            request.success = true;
        } catch (const std::exception& error) {
            request.error = error.what();
        }
        const std::lock_guard lock{mutex_};
        // A reset invalidates the pending identity, including in-flight failures.
        if (pending_.sequence == 0 || request.generation != pending_.generation)
            continue;
        if (!request.success)
            healthy_.store(false, std::memory_order::release);
        result_.sequence = request.sequence;
        result_.generation = request.generation;
        result_.submitted_at = request.submitted_at;
        result_.success = request.success;
        result_.error = std::move(request.error);
        result_.action.assign(action_.begin(), action_.end());
        has_result_ = true;
    }
}

} // namespace rmcs_rl
