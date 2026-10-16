#pragma once

#include <cstddef>
#include <chrono>
#include <deque>
#include <span>
#include <vector>

namespace rmcs_rl {

// Frames are oldest first; startup and reset fill the history with the first valid frame.
class ObservationHistory {
public:
    using Clock = std::chrono::steady_clock;
    ObservationHistory(std::size_t frame_size, std::size_t history_length,
        std::chrono::nanoseconds sample_period = {});
    const std::vector<double>& append(std::span<const double> frame);
    const std::vector<double>& append(std::span<const double> frame, Clock::time_point now);
    void reset();

private:
    std::size_t frame_size_;
    std::vector<double> values_;
    struct Sample {
        Clock::time_point time;
        std::vector<double> frame;
    };
    std::chrono::nanoseconds sample_period_{};
    std::deque<Sample> samples_;
    bool initialized_ = false;
};

} // namespace rmcs_rl
