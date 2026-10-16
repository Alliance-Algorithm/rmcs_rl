#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <rmcs_executor/component.hpp>
#include <rmcs_rl/rl_bridge/types.hpp>

namespace rmcs_rl {

class ObservationTimeline {
public:
    using Clock = std::chrono::steady_clock;
    bool due(Clock::time_point now, std::chrono::nanoseconds period) const {
        return !started_ || now - scheduled_ >= period;
    }
    std::uint64_t publish(Clock::time_point now, std::chrono::nanoseconds period) {
        scheduled_ = !started_ || period.count() <= 0
                       ? now
                       : scheduled_ + period * ((now - scheduled_) / period);
        started_ = true;
        return ++sequence_;
    }
    void reset() {
        started_ = false;
        scheduled_ = {};
    }
    [[nodiscard]] std::uint64_t sequence() const { return sequence_; }

private:
    Clock::time_point scheduled_{};
    std::uint64_t sequence_ = 0;
    bool started_ = false;
};

class ObservationHistory {
public:
    using Clock = std::chrono::steady_clock;
    ObservationHistory(
        std::size_t frame_size, std::size_t history_length, std::chrono::nanoseconds sample_period)
        : frame_size_(frame_size)
        , values_(frame_size * history_length)
        , sample_times_(history_length)
        , sample_values_(frame_size * history_length)
        , sample_period_(sample_period)
        , capacity_(history_length) {
        if (frame_size == 0 || history_length == 0 || sample_period.count() < 0)
            throw std::invalid_argument("invalid observation history configuration");
    }

    const std::vector<double>& append(std::span<const double> frame, Clock::time_point now) {
        if (frame.size() != frame_size_)
            throw std::invalid_argument("observation frame does not match configured frame size");
        if (sample_period_.count() == 0)
            return append_fast_(frame);
        if (sample_count_ != 0) {
            const auto newest = (next_sample_ + capacity_ - 1) % capacity_;
            if (now <= sample_times_[newest])
                return values_;
            if (now - sample_times_[newest] < sample_period_)
                return values_;
            if (now - sample_times_[newest] > 2 * sample_period_)
                reset();
        }
        const auto slot = next_sample_;
        sample_times_[slot] = now;
        std::copy(frame.begin(), frame.end(), sample_values_.begin() + slot * frame_size_);
        next_sample_ = (next_sample_ + 1) % capacity_;
        sample_count_ = std::min(sample_count_ + 1, capacity_);
        const auto oldest = (next_sample_ + capacity_ - sample_count_) % capacity_;
        const auto history_length = values_.size() / frame_size_;
        const auto oldest_time =
            now - sample_period_ * static_cast<std::int64_t>(history_length - 1);
        std::size_t sample = 0;
        for (std::size_t i = 0; i < history_length; ++i) {
            const auto target = oldest_time + sample_period_ * static_cast<std::int64_t>(i);
            while (sample + 1 < sample_count_
                   && sample_times_[(oldest + sample + 1) % capacity_] <= target)
                ++sample;
            const auto source = (oldest + sample) % capacity_;
            std::copy_n(
                sample_values_.begin() + source * frame_size_, frame_size_,
                values_.begin() + i * frame_size_);
        }
        return values_;
    }

    void reset() {
        initialized_ = false;
        sample_count_ = next_sample_ = 0;
        std::fill(values_.begin(), values_.end(), 0.0);
    }

private:
    const std::vector<double>& append_fast_(std::span<const double> frame) {
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

    std::size_t frame_size_;
    std::vector<double> values_;
    std::vector<Clock::time_point> sample_times_;
    std::vector<double> sample_values_;
    std::chrono::nanoseconds sample_period_;
    std::size_t capacity_;
    std::size_t sample_count_ = 0;
    std::size_t next_sample_ = 0;
    bool initialized_ = false;
};

class ActionFeedbackHistory {
public:
    using Clock = std::chrono::steady_clock;
    ActionFeedbackHistory(std::size_t size, std::size_t capacity)
        : values_(size)
        , times_(capacity)
        , actions_(size * capacity)
        , capacity_(capacity) {}

    void store(Clock::time_point time, std::span<const double> action) {
        if (action.size() != values_.size())
            throw std::invalid_argument("action feedback size mismatch");
        if (count_ != 0) {
            const auto previous = (next_ + capacity_ - 1) % capacity_;
            if (time <= times_[previous])
                return;
        }
        times_[next_] = time;
        std::copy(action.begin(), action.end(), actions_.begin() + next_ * values_.size());
        next_ = (next_ + 1) % capacity_;
        count_ = std::min(count_ + 1, capacity_);
    }

    const std::vector<double>& sample(Clock::time_point now, std::chrono::nanoseconds delay) {
        if (count_ == 0)
            return values_;
        const auto target = now - delay;
        const auto oldest = (next_ + capacity_ - count_) % capacity_;
        std::size_t selected = 0;
        for (std::size_t i = 1; i < count_; ++i) {
            const auto candidate = (oldest + i) % capacity_;
            if (times_[candidate] > target)
                break;
            selected = i;
        }
        const auto source = (oldest + selected) % capacity_;
        if (times_[source] <= target)
            std::copy_n(
                actions_.begin() + source * values_.size(), values_.size(), values_.begin());
        return values_;
    }

    void reset() {
        count_ = next_ = 0;
        std::fill(values_.begin(), values_.end(), 0.0);
    }

private:
    std::vector<double> values_;
    std::vector<Clock::time_point> times_;
    std::vector<double> actions_;
    std::size_t capacity_;
    std::size_t count_ = 0;
    std::size_t next_ = 0;
};

inline void reset_action_state(std::vector<double>& last_actions, std::vector<double>& written) {
    std::fill(last_actions.begin(), last_actions.end(), 0.0);
    std::fill(written.begin(), written.end(), 0.0);
}

inline void write_actions(
    bool valid, std::span<const double> action, const std::vector<ActionTerm>& terms,
    InvalidMode invalid_mode, std::vector<double>& written,
    std::vector<std::unique_ptr<rmcs_executor::Component::OutputInterface<double>>>& outputs) {
    for (std::size_t i = 0; i < terms.size(); ++i) {
        double value;
        if (valid) {
            value = action[i] * terms[i].scale;
            if (terms[i].has_clip)
                value = std::clamp(value, terms[i].clip_min, terms[i].clip_max);
            written[i] = value;
        } else if (invalid_mode == InvalidMode::kNaN) {
            value = std::numeric_limits<double>::quiet_NaN();
        } else if (invalid_mode == InvalidMode::kZero) {
            value = 0.0;
        } else {
            value = written[i];
        }
        **outputs[i] = value;
    }
}

inline std::string invalid_reason(
    bool enabled, bool healthy, bool has_action, bool fresh, bool sequence_ok, bool finite) {
    if (!healthy)
        return "inference worker failed";
    if (!enabled)
        return "disabled by enable interface";
    if (!has_action)
        return "no inference result available";
    if (!finite)
        return "action contains non-finite values";
    if (!sequence_ok)
        return "action is from an invalid or previous reset generation";
    if (!fresh)
        return "inference result is stale";
    return "unknown";
}

using OutputAction =
    std::vector<std::unique_ptr<rmcs_executor::Component::OutputInterface<double>>>;

} // namespace rmcs_rl
