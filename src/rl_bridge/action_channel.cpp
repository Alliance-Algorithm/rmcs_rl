#include <rmcs_rl/rl_bridge/action_channel.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace rmcs_rl {

void ActionChannel::resize(std::size_t action_size) {
    const std::lock_guard lock{mutex_};
    incoming_.action.assign(action_size, 0.0);
    received_ = false;
    has_read_.store(false, std::memory_order_release);
}

void ActionChannel::store(const rmcs_rl::msg::Action& message) {
    const std::lock_guard lock{mutex_};
    if (message.action.size() != incoming_.action.size())
        throw std::invalid_argument("action channel size mismatch");
    incoming_.obs_seq = message.obs_seq;
    incoming_.layout_hash = message.layout_hash;
    incoming_.model_id = message.model_id;
    std::copy(message.action.begin(), message.action.end(), incoming_.action.begin());
    received_ = true;
}

bool ActionChannel::try_read(ActionSnapshot& snapshot) {
    // The executor must never wait for the ROS callback. A seqlock cannot protect
    // concurrent reads/writes of a non-atomic vector under the C++ memory model.
    const std::unique_lock lock{mutex_, std::try_to_lock};
    if (!lock.owns_lock())
        // Keep the caller's previous snapshot. The bridge still checks its age,
        // sequence and contract before allowing it to drive outputs.
        return has_read_.load(std::memory_order_acquire);
    if (!received_)
        return false;
    snapshot = incoming_;
    has_read_.store(true, std::memory_order_release);
    return true;
}

void write_actions(
    bool valid, const ActionSnapshot& snapshot, const std::vector<ActionTerm>& terms,
    InvalidMode invalid_mode, std::vector<double>& written,
    std::vector<std::unique_ptr<rmcs_executor::Component::OutputInterface<double>>>& outputs) {
    for (std::size_t i = 0; i < terms.size(); ++i) {
        const auto& term = terms[i];
        double value = 0.0;
        if (valid) {
            value = snapshot.action[i] * term.scale;
            if (term.has_clip)
                value = std::clamp(value, term.clip_min, term.clip_max);
            written[i] = value;
        } else {
            switch (invalid_mode) {
            case InvalidMode::kNaN: value = std::numeric_limits<double>::quiet_NaN(); break;
            case InvalidMode::kZero: value = 0.0; break;
            case InvalidMode::kHold: value = written[i]; break;
            }
        }
        **outputs[i] = value;
    }
}

void reset_action_state(std::vector<double>& last_actions, std::vector<double>& written) {
    std::fill(last_actions.begin(), last_actions.end(), 0.0);
    std::fill(written.begin(), written.end(), 0.0);
}

std::string invalid_reason(
    bool enabled, bool contract_ok, bool has_snapshot, bool fresh, bool seq_ok, bool finite) {
    if (!contract_ok)
        return "contract/model mismatch (latched; restart required)";
    if (!enabled)
        return "disabled by enable interface";
    if (!has_snapshot)
        return "no action received yet";
    if (!finite)
        return "action contains non-finite values";
    if (!seq_ok)
        return "action answers an obs frame older than the previous one";
    if (!fresh)
        return "action is stale (age > max_action_age)";
    return "unknown";
}

} // namespace rmcs_rl
