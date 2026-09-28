#include <rmcs_rl/rl_bridge/observation_history.hpp>

#include <algorithm>
#include <stdexcept>

namespace rmcs_rl {

ObservationHistory::ObservationHistory(std::size_t frame_size, std::size_t history_length)
    : frame_size_(frame_size)
    , values_(frame_size * history_length) {
    if (frame_size == 0 || history_length == 0)
        throw std::invalid_argument("observation frame size and history length must be positive");
}

const std::vector<double>& ObservationHistory::append(std::span<const double> frame) {
    if (frame.size() != frame_size_)
        throw std::invalid_argument("observation frame does not match configured frame size");
    if (!initialized_) {
        for (std::size_t offset = 0; offset < values_.size(); offset += frame_size_)
            std::copy(frame.begin(), frame.end(), values_.begin() + offset);
        initialized_ = true;
    } else {
        std::move(values_.begin() + frame_size_, values_.end(), values_.begin());
        std::copy(frame.begin(), frame.end(), values_.end() - frame_size_);
    }
    return values_;
}

void ObservationHistory::reset() {
    initialized_ = false;
    std::fill(values_.begin(), values_.end(), 0.0);
}

} // namespace rmcs_rl
