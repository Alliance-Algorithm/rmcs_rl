#include <rmcs_rl/rl_bridge/observation_history.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace rmcs_rl {

ObservationHistory::ObservationHistory(std::size_t frame_size, std::size_t history_length,
    std::chrono::nanoseconds sample_period)
    : frame_size_(frame_size)
    , values_(frame_size * history_length)
    , sample_period_(sample_period) {
    if (frame_size == 0 || history_length == 0)
        throw std::invalid_argument("observation frame size and history length must be positive");
    if (sample_period.count() < 0)
        throw std::invalid_argument("observation history period cannot be negative");
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

const std::vector<double>& ObservationHistory::append(std::span<const double> frame, Clock::time_point now) {
    if (sample_period_.count() == 0)
        return append(frame);
    if (frame.size() != frame_size_)
        throw std::invalid_argument("observation frame does not match configured frame size");
    if (!samples_.empty() && now <= samples_.back().time)
        throw std::invalid_argument("observation history timestamps must increase");
    if (!samples_.empty() && now - samples_.back().time > 2 * sample_period_)
        reset();
    samples_.push_back({now, std::vector<double>{frame.begin(), frame.end()}});
    const std::size_t length = values_.size() / frame_size_;
    const auto oldest_time = now - sample_period_ * static_cast<std::int64_t>(length - 1);
    while (samples_.size() > 1 && samples_[1].time <= oldest_time)
        samples_.pop_front();
    std::size_t sample = 0;
    for (std::size_t i = 0; i < length; ++i) {
        const auto target = now - sample_period_ * static_cast<std::int64_t>(length - i - 1);
        while (sample + 1 < samples_.size() && samples_[sample + 1].time <= target)
            ++sample;
        std::copy(samples_[sample].frame.begin(), samples_[sample].frame.end(),
            values_.begin() + i * frame_size_);
    }
    return values_;
}

void ObservationHistory::reset() {
    initialized_ = false;
    samples_.clear();
    std::fill(values_.begin(), values_.end(), 0.0);
}

} // namespace rmcs_rl
