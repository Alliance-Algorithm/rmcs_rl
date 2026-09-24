#include <rmcs_rl/rl_bridge/action_channel.hpp>

#include <algorithm>
#include <limits>

namespace rmcs_rl {

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
            case InvalidMode::kNaN:
                value = std::numeric_limits<double>::quiet_NaN();
                break;
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
        return "contract/model mismatch (latched; restart or re-enable)";
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
