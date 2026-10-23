#pragma once

#include <cstddef>

namespace rmcs_rl {

// The producer owns frame cadence. Re-reading a frame cannot refresh its age.
class ObservationFrameGate {
public:
    enum class Decision { kWait, kSubmit, kReset };

    Decision observe(std::size_t sequence) {
        if (blocked_)
            return Decision::kWait;
        if (sequence == 0) {
            const bool had_frame = last_sequence_ != 0;
            last_sequence_ = 0;
            return had_frame ? Decision::kReset : Decision::kWait;
        }
        if (sequence == last_sequence_)
            return Decision::kWait;
        if (sequence < last_sequence_) {
            blocked_ = true;
            return Decision::kReset;
        }
        last_sequence_ = sequence;
        return Decision::kSubmit;
    }

    void reset() {
        last_sequence_ = 0;
        blocked_ = false;
    }

private:
    std::size_t last_sequence_ = 0;
    bool blocked_ = false;
};

} // namespace rmcs_rl
