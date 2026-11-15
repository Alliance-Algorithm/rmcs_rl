#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <rmcs_executor/component.hpp>

#include "rl_bridge/config.hpp"

namespace rmcs_rl {

class PolicyActions {
public:
    PolicyActions(rmcs_executor::Component& component, std::vector<ActionTerm> terms) {
        for (auto& term : terms) {
            auto& channel = *channels_.emplace_back(std::make_unique<Channel>(std::move(term)));
            component.register_output(channel.term.output, channel.value, kNaN);
            component.register_output(channel.term.output + "/raw", channel.raw, kNaN);
        }
    }

    std::size_t size() const { return channels_.size(); }

    void publish(std::span<const double> raw) {
        for (std::size_t i = 0; i < channels_.size(); ++i) {
            *channels_[i]->value = map(channels_[i]->term, raw[i]);
            *channels_[i]->raw = raw[i];
        }
    }

    void publish_invalid() {
        for (auto& channel : channels_)
            *channel->value = *channel->raw = kNaN;
    }

    void normalize(std::span<const double> raw, std::span<double> normalized) const {
        for (std::size_t i = 0; i < channels_.size(); ++i) {
            const auto& term = channels_[i]->term;
            normalized[i] = term.scale == 0. ? 0. : map(term, raw[i]) / term.scale;
        }
    }

private:
    struct Channel {
        explicit Channel(ActionTerm term)
            : term(std::move(term)) {}
        ActionTerm term;
        rmcs_executor::Component::OutputInterface<double> value, raw;
    };

    static double map(const ActionTerm& term, double raw) {
        const double value = raw * term.scale;
        return term.clip ? std::clamp(value, term.clip->lower, term.clip->upper) : value;
    }

    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::unique_ptr<Channel>> channels_;
};

} // namespace rmcs_rl
