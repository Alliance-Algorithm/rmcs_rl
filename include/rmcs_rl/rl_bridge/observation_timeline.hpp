#pragma once

#include <chrono>
#include <cstdint>
#include <limits>

namespace rmcs_rl {

class ObservationTimeline {
public:
    using Clock = std::chrono::steady_clock;

    bool due(Clock::time_point now, std::chrono::nanoseconds period) const {
        return !started_ || now - current_time_ >= period;
    }

    std::uint64_t publish(Clock::time_point now) {
        previous_valid_ = started_;
        previous_time_ = current_time_;
        current_time_ = now;
        started_ = true;
        return ++sequence_;
    }

    double age(std::uint64_t sequence, Clock::time_point now) const {
        if (started_ && sequence == sequence_)
            return std::chrono::duration<double>(now - current_time_).count();
        if (previous_valid_ && sequence == sequence_ - 1)
            return std::chrono::duration<double>(now - previous_time_).count();
        return std::numeric_limits<double>::quiet_NaN();
    }

    void reset() {
        // Keep IDs monotonic so a delayed action cannot match a new frame after reset.
        started_ = previous_valid_ = false;
        current_time_ = previous_time_ = {};
    }

    [[nodiscard]] std::uint64_t sequence() const { return sequence_; }

private:
    Clock::time_point current_time_{};
    Clock::time_point previous_time_{};
    std::uint64_t sequence_ = 0;
    bool started_ = false;
    bool previous_valid_ = false;
};

} // namespace rmcs_rl
