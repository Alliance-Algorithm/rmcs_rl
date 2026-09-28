#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace rmcs_rl {

// Frames are oldest first; startup and reset fill the history with the first valid frame.
class ObservationHistory {
public:
    ObservationHistory(std::size_t frame_size, std::size_t history_length);
    const std::vector<double>& append(std::span<const double> frame);
    void reset();

private:
    std::size_t frame_size_;
    std::vector<double> values_;
    bool initialized_ = false;
};

} // namespace rmcs_rl
