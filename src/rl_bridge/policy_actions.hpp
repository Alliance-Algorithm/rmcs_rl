#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include <rmcs_executor/component.hpp>

#include "rl_bridge/config.hpp"

namespace rmcs_rl {

class PolicyActions {
public:
    PolicyActions(
        rmcs_executor::Component& component, const std::vector<ActionTerm>& terms,
        InvalidAction invalid_action)
        : invalid_action_(invalid_action) {
        constexpr double infinity = std::numeric_limits<double>::infinity();
        for (const auto& term : terms) {
            auto& channel = *channels_.emplace_back(std::make_unique<Channel>(term));
            component.register_output(term.output, channel.value, 0.0);
            component.register_output(term.output + "/raw", channel.raw, 0.0);
            component.register_output(term.output + "/scale", channel.scale, term.scale);
            component.register_output(term.output + "/offset", channel.offset, term.offset);
            component.register_output(
                term.output + "/clip_min", channel.clip_min,
                term.clip ? term.clip->lower : -infinity);
            component.register_output(
                term.output + "/clip_max", channel.clip_max,
                term.clip ? term.clip->upper : infinity);
        }
    }

    PolicyActions(const PolicyActions&) = delete;
    PolicyActions& operator=(const PolicyActions&) = delete;
    PolicyActions(PolicyActions&&) = delete;
    PolicyActions& operator=(PolicyActions&&) = delete;

    ~PolicyActions() = default;

    void publish(std::span<const double> raw) {
        for (std::size_t i = 0; i < channels_.size(); ++i) {
            auto& channel = *channels_[i];
            channel.written = map(channel.term, raw[i]);
            *channel.value = channel.written;
            *channel.raw = raw[i];
        }
    }

    void publish_invalid() {
        for (const auto& channel : channels_) {
            *channel->value = invalid_value(*channel);
            *channel->raw = nan_;
        }
    }

    // Converts mapped actions back to normalized policy units for previous_action terms.
    void normalize(std::span<const double> raw, std::span<double> normalized) const {
        for (std::size_t i = 0; i < channels_.size(); ++i) {
            const auto& term = channels_[i]->term;
            normalized[i] =
                term.scale == 0.0 ? 0.0 : (map(term, raw[i]) - term.offset) / term.scale;
        }
    }

    void reset() {
        for (const auto& channel : channels_)
            channel->written = 0.0;
    }

private:
    struct Channel {
        explicit Channel(const ActionTerm& term)
            : term(term) {}

        ActionTerm term;
        double written = 0.0;

        rmcs_executor::Component::OutputInterface<double> value, raw;
        rmcs_executor::Component::OutputInterface<double> scale, offset, clip_min, clip_max;
    };

    static double map(const ActionTerm& term, double raw) {
        const double value = raw * term.scale + term.offset;
        return term.clip ? std::clamp(value, term.clip->lower, term.clip->upper) : value;
    }

    double invalid_value(const Channel& channel) const {
        switch (invalid_action_) {
        case InvalidAction::kNaN: return nan_;
        case InvalidAction::kZero: return 0.0;
        case InvalidAction::kHold: return channel.written;
        }
        return nan_;
    }

    static constexpr double nan_ = std::numeric_limits<double>::quiet_NaN();

    InvalidAction invalid_action_;
    std::vector<std::unique_ptr<Channel>> channels_;
};

} // namespace rmcs_rl
