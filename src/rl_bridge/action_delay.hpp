#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rmcs_rl {

// ccdeb79 ActionFeedbackHistory semantics, keyed by executor ticks instead of steady_clock.
class ActionDelay {
public:
    ActionDelay(std::size_t size, std::size_t capacity)
        : held_(size)
        , ticks_(capacity)
        , sequences_(capacity)
        , actions_(size * capacity) {}

    void mark_submission(std::uint64_t sequence, std::size_t tick) {
        submissions_[sequence % kSubmissionSlots] = {sequence, tick};
    }

    void store(std::uint64_t sequence, std::span<const double> action) {
        const auto& submission = submissions_[sequence % kSubmissionSlots];
        if (sequence == 0 || submission.sequence != sequence)
            return;
        if (count_ && submission.tick <= ticks_[(next_ + ticks_.size() - 1) % ticks_.size()])
            return;
        ticks_[next_] = submission.tick;
        sequences_[next_] = sequence;
        std::ranges::copy(action, actions_.begin() + next_ * held_.size());
        next_ = (next_ + 1) % ticks_.size();
        count_ = std::min(count_ + 1, ticks_.size());
    }

    std::span<const double> select(std::size_t tick) {
        for (std::size_t i = 0; i < count_; ++i) {
            const auto slot = (next_ + ticks_.size() - count_ + i) % ticks_.size();
            if (ticks_[slot] > tick)
                break;
            std::copy_n(actions_.begin() + slot * held_.size(), held_.size(), held_.begin());
        }
        return held_;
    }

    void reset() {
        submissions_.fill({});
        std::ranges::fill(held_, 0.);
        std::ranges::fill(ticks_, 0);
        std::ranges::fill(sequences_, 0);
        std::ranges::fill(actions_, 0.);
        count_ = next_ = 0;
    }

private:
    static constexpr std::size_t kSubmissionSlots = 64;
    struct Submission {
        std::uint64_t sequence = 0;
        std::size_t tick = 0;
    };
    std::array<Submission, kSubmissionSlots> submissions_{};
    std::vector<double> held_;
    std::vector<std::size_t> ticks_;
    std::vector<std::uint64_t> sequences_;
    std::vector<double> actions_;
    std::size_t count_ = 0, next_ = 0;
};

} // namespace rmcs_rl
