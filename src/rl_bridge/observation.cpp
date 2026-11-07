#include <rmcs_rl/rl_bridge/observation.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <rmcs_rl/rl_bridge/interface_binding.hpp>
#include <rmcs_rl/rl_bridge/utility.hpp>

namespace rmcs_rl {

bool build_observation(
    const std::vector<ObsTerm>& terms, const std::vector<Slot>& slots,
    const std::vector<double>& last_actions, std::size_t obs_size, std::vector<double>& obs) {
    obs.assign(obs_size, 0.0);
    for (const auto& term : terms) {
        if (term.index > obs_size || term.dim > obs_size - term.index)
            return false;
        const auto push = [&obs, &term](
                              std::size_t index, double value, double scale, bool has_clip,
                              double clip_min, double clip_max, bool& ok) {
            if (!term.selection.empty()
                && std::ranges::find(term.selection, index - term.index) == term.selection.end())
                return;
            const auto quantize = [&term](double v) {
                return term.float32 ? static_cast<double>(static_cast<float>(v)) : v;
            };
            double result = quantize(value);
            if (!is_finite(value) || !is_finite(result)) {
                ok = false;
                return;
            }
            if (term.has_source_clip)
                result = std::clamp(
                    result, quantize(term.source_clip_min), quantize(term.source_clip_max));
            result = quantize(result * quantize(scale));
            result = quantize(result + quantize(term.offset));
            if (has_clip)
                result = std::clamp(result, clip_min, clip_max);
            if (!is_finite(value) || !is_finite(result))
                ok = false;
            if (term.selection.empty())
                obs[index] = result;
            else
                for (std::size_t i = 0; i < term.selection.size(); ++i)
                    if (term.selection[i] == index - term.index)
                        obs[term.index + i] = result;
        };
        bool ok = true;
        switch (term.kind) {
        case TermKind::kPath: {
            if (term.slot == kNoSlot) {
                push(
                    term.index, term.default_value, term.scale, term.has_clip, term.clip_min,
                    term.clip_max, ok);
                break;
            }
            if (term.slot >= slots.size())
                return false;
            const auto& entry = slots[term.slot];
            if (term.take == Take::kScalar) {
                double value = 0.0;
                if (!read_double(entry, value))
                    return false;
                push(
                    term.index, value, term.scale, term.has_clip, term.clip_min, term.clip_max, ok);
            } else if (term.take == Take::kComponent) {
                double value = 0.0;
                if (entry.binding == Binding::kVector3) {
                    if (!entry.vector3_value->ready())
                        return false;
                    value = (**entry.vector3_value)[term.component];
                } else {
                    if (!entry.direction_vector_value->ready())
                        return false;
                    value = (**entry.direction_vector_value).vector[term.component];
                }
                push(
                    term.index, value, term.scale, term.has_clip, term.clip_min, term.clip_max, ok);
            } else if (term.take == Take::kVector) {
                if (entry.binding == Binding::kVector3) {
                    if (!entry.vector3_value->ready())
                        return false;
                    const auto& value = **entry.vector3_value;
                    for (std::size_t i = 0; i < 3; ++i)
                        push(
                            term.index + i, value[static_cast<Eigen::Index>(i)], term.scale,
                            term.has_clip, term.clip_min, term.clip_max, ok);
                } else {
                    if (!entry.direction_vector_value->ready())
                        return false;
                    const auto& value = (**entry.direction_vector_value).vector;
                    for (std::size_t i = 0; i < 3; ++i)
                        push(
                            term.index + i, value[static_cast<Eigen::Index>(i)], term.scale,
                            term.has_clip, term.clip_min, term.clip_max, ok);
                }
            } else if (term.take == Take::kArray) {
                if (entry.binding != Binding::kDoubleArray || !entry.array_value
                    || !entry.array_value->ready())
                    return false;
                const auto& values = **entry.array_value;
                const auto size = term.source_size == 0 ? term.dim : term.source_size;
                if (values.size() != size)
                    return false;
                for (std::size_t i = 0; i < size; ++i)
                    push(
                        term.index + i, values[i], term.scale, term.has_clip, term.clip_min,
                        term.clip_max, ok);
            } else {
                if (!entry.quaternion_value->ready())
                    return false;
                const Eigen::Vector3d gravity =
                    **entry.quaternion_value * Eigen::Vector3d(0.0, 0.0, -1.0);
                for (std::size_t i = 0; i < 3; ++i)
                    push(
                        term.index + i, gravity[static_cast<Eigen::Index>(i)], term.scale,
                        term.has_clip, term.clip_min, term.clip_max, ok);
            }
            break;
        }
        case TermKind::kJointPos: {
            for (std::size_t j = 0; j < term.joints.size(); ++j) {
                double value = 0.0;
                if (!term.zero) {
                    if (!read_double(slots[term.joint_slots[j]], value))
                        return false;
                    if (term.relative) {
                        const double delta = value - term.joint_defaults[j];
                        value = std::atan2(std::sin(delta), std::cos(delta));
                    }
                }
                push(
                    term.index + j, value, term.scale, term.has_clip, term.clip_min, term.clip_max,
                    ok);
            }
            break;
        }
        case TermKind::kJointVel:
        case TermKind::kSources:
        case TermKind::kJointTorque: {
            for (std::size_t j = 0; j < term.joint_slots.size(); ++j) {
                double value = 0.0;
                if (!read_double(slots[term.joint_slots[j]], value))
                    return false;
                push(
                    term.index + j, value, term.scale, term.has_clip, term.clip_min, term.clip_max,
                    ok);
            }
            break;
        }
        case TermKind::kLastAction: {
            for (std::size_t j = 0; j < term.action_indices.size(); ++j)
                push(
                    term.index + j, last_actions[term.action_indices[j]], term.scale, term.has_clip,
                    term.clip_min, term.clip_max, ok);
            break;
        }
        case TermKind::kConstant: {
            for (std::size_t j = 0; j < term.constants.size(); ++j)
                push(
                    term.index + j, term.constants[j], term.scale, term.has_clip, term.clip_min,
                    term.clip_max, ok);
            break;
        }
        }
        if (!ok)
            return false;
    }
    return true;
}

} // namespace rmcs_rl
