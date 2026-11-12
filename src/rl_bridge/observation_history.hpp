#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace rmcs_rl {

// Frames are oldest first; the first frame after a reset fills the whole history.
class ObservationHistory {
public:
    ObservationHistory(std::size_t frame_size, std::size_t history_length)
        : frame_size_(frame_size)
        , values_(frame_size * history_length) {
        if (frame_size == 0 || history_length == 0)
            throw std::invalid_argument(
                "observation frame size and history length must be positive");
    }

    std::span<const double> append(std::span<const double> frame) {
        if (frame.size() != frame_size_)
            throw std::invalid_argument("observation frame does not match configured frame size");
        if (!initialized_) {
            for (std::size_t offset = 0; offset < values_.size(); offset += frame_size_)
                std::ranges::copy(frame, values_.begin() + offset);
            initialized_ = true;
        } else {
            std::shift_left(values_.begin(), values_.end(), frame_size_);
            std::ranges::copy(frame, values_.end() - frame_size_);
        }
        return values_;
    }

    void reset() {
        initialized_ = false;
        std::ranges::fill(values_, 0.0);
    }

private:
    std::size_t frame_size_;
    std::vector<double> values_;
    bool initialized_ = false;
};

} // namespace rmcs_rl
