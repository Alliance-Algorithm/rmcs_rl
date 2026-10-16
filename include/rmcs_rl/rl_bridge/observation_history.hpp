#pragma once

#include <chrono>
#include <cstddef>
#include <span>
#include <vector>

namespace rmcs_rl {

// Frames are oldest first; startup and reset fill the history with the first valid frame.
class ObservationHistory {
public:
    using Clock = std::chrono::steady_clock;
    ObservationHistory(
        std::size_t frame_size, std::size_t history_length,
        std::chrono::nanoseconds sample_period = {});
    const std::vector<double>& append(std::span<const double> frame);
    const std::vector<double>& append(std::span<const double> frame, Clock::time_point now);
    void reset();

private:
    std::size_t frame_size_;
    std::vector<double> values_;
    std::vector<Clock::time_point> sample_times_;
    std::vector<double> sample_values_;
    std::chrono::nanoseconds sample_period_{};
    std::size_t sample_capacity_ = 0;
    std::size_t sample_count_ = 0;
    std::size_t next_sample_ = 0;
    bool initialized_ = false;
};

} // namespace rmcs_rl
