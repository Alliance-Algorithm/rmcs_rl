#pragma once

#include <chrono>

namespace rmcs_rl {

class ObservationTimeline {
public:
    using Clock = std::chrono::steady_clock;

    explicit ObservationTimeline(std::chrono::nanoseconds period)
        : period_(period) {}

    [[nodiscard]] bool due(Clock::time_point now) const {
        return !started_ || now - scheduled_ >= period_;
    }

    void advance(Clock::time_point now) {
        scheduled_ = !started_ || period_.count() <= 0
                       ? now
                       : scheduled_ + period_ * ((now - scheduled_) / period_);
        started_ = true;
    }

    void reset() {
        started_ = false;
        scheduled_ = {};
    }

private:
    std::chrono::nanoseconds period_;
    Clock::time_point scheduled_{};
    bool started_ = false;
};

} // namespace rmcs_rl
