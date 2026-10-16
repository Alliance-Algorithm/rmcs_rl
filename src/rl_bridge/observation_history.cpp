#include <rmcs_rl/rl_bridge/observation_history.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace rmcs_rl {

ObservationHistory::ObservationHistory(
    std::size_t frame_size, std::size_t history_length, std::chrono::nanoseconds sample_period)
    : frame_size_(frame_size)
    , values_(frame_size * history_length)
    , sample_times_(history_length)
    , sample_values_(frame_size * history_length)
    , sample_period_(sample_period) {
    if (frame_size == 0 || history_length == 0)
        throw std::invalid_argument("observation frame size and history length must be positive");
    if (sample_period.count() < 0)
        throw std::invalid_argument("observation history period cannot be negative");
    sample_capacity_ = history_length;
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

const std::vector<double>&
    ObservationHistory::append(std::span<const double> frame, Clock::time_point now) {
    if (sample_period_.count() == 0)
        return append(frame);
    if (frame.size() != frame_size_)
        throw std::invalid_argument("observation frame does not match configured frame size");
    if (sample_count_ > 0) {
        const std::size_t newest = (next_sample_ + sample_capacity_ - 1) % sample_capacity_;
        const auto last_sample_time = sample_times_[newest];
        if (now <= last_sample_time)
            throw std::invalid_argument("observation history timestamps must increase");
        if (now - last_sample_time < sample_period_)
            return values_;
        if (now - last_sample_time > 2 * sample_period_)
            reset();
    }
    const std::size_t slot = next_sample_;
    sample_times_[slot] = now;
    std::copy(frame.begin(), frame.end(), sample_values_.begin() + slot * frame_size_);
    next_sample_ = (next_sample_ + 1) % sample_capacity_;
    sample_count_ = std::min(sample_count_ + 1, sample_capacity_);
    const std::size_t oldest_slot =
        (next_sample_ + sample_capacity_ - sample_count_) % sample_capacity_;
    const auto oldest_time =
        now - sample_period_ * static_cast<std::int64_t>(values_.size() / frame_size_ - 1);
    std::size_t sample_index = 0;
    for (std::size_t i = 0; i < values_.size() / frame_size_; ++i) {
        const auto target = oldest_time + sample_period_ * static_cast<std::int64_t>(i);
        while (sample_index + 1 < sample_count_) {
            const auto candidate = (oldest_slot + sample_index + 1) % sample_capacity_;
            if (sample_times_[candidate] > target)
                break;
            ++sample_index;
        }
        const auto source = (oldest_slot + sample_index) % sample_capacity_;
        std::copy_n(
            sample_values_.begin() + source * frame_size_, frame_size_,
            values_.begin() + i * frame_size_);
    }
    initialized_ = true;
    return values_;
}

void ObservationHistory::reset() {
    initialized_ = false;
    sample_count_ = 0;
    next_sample_ = 0;
    std::fill(values_.begin(), values_.end(), 0.0);
}

} // namespace rmcs_rl
