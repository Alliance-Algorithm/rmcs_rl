#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <deque>
#include <span>
#include <stdexcept>
#include <vector>

namespace rmcs_rl {

// Prior-action features retain their training delay when inference is faster.
class ActionFeedbackHistory {
public:
    using Clock = std::chrono::steady_clock;
    explicit ActionFeedbackHistory(std::size_t size) : values_(size, 0.0) {}

    void store(Clock::time_point observation_time, std::span<const double> action) {
        if (action.size() != values_.size())
            throw std::invalid_argument("action feedback size mismatch");
        if (!samples_.empty() && observation_time <= samples_.back().time)
            return;
        samples_.push_back({observation_time, std::vector<double>{action.begin(), action.end()}});
    }

    const std::vector<double>& sample(Clock::time_point now, std::chrono::nanoseconds delay) {
        const auto target = now - delay;
        while (samples_.size() > 1 && samples_[1].time <= target)
            samples_.pop_front();
        if (!samples_.empty() && samples_.front().time <= target)
            values_ = samples_.front().action;
        return values_;
    }

    void reset() {
        samples_.clear();
        std::fill(values_.begin(), values_.end(), 0.0);
    }

private:
    struct Sample {
        Clock::time_point time;
        std::vector<double> action;
    };
    std::deque<Sample> samples_;
    std::vector<double> values_;
};

} // namespace rmcs_rl
