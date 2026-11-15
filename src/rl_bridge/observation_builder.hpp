#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <eigen3/Eigen/Core>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>

#include "rl_bridge/config.hpp"

namespace rmcs_rl {

class ObservationBuilder {
    using Component = rmcs_executor::Component;
    using DirectionVector = rmcs_description::BaseLink::DirectionVector;
    template <typename T>
    using Interface = Component::InputInterface<T>;
    using Input = std::variant<
        Interface<double>, Interface<Eigen::Vector3d>, Interface<DirectionVector>,
        Interface<std::vector<double>>>;

public:
    explicit ObservationBuilder(std::vector<ObservationTerm> terms)
        : terms_(std::move(terms))
        , term_inputs_(terms_.size()) {
        std::size_t source_size = 0;
        for (auto& term : terms_) {
            term.index = frame_size_;
            frame_size_ += term.size;
            source_size = std::max(source_size, term.source_size);
        }
        source_.resize(source_size);
    }

    std::size_t frame_size() const { return frame_size_; }

    void bind(Component& component, const Component::OutputInfoMap& output_map) {
        std::map<std::string, Input*> registered;
        auto input = [&]<typename T>(const std::string& path) {
            if (const auto found = registered.find(path); found != registered.end())
                return found->second;
            auto& value =
                inputs_.emplace_back(std::make_unique<Input>(std::in_place_type<Interface<T>>));
            component.register_input(path, std::get<Interface<T>>(*value));
            registered.emplace(path, value.get());
            return value.get();
        };
        for (std::size_t i = 0; i < terms_.size(); ++i) {
            const auto& term = terms_[i];
            for (const auto& path : term.sources) {
                Input* source;
                if (term.type == ObservationType::kVector3) {
                    if (output_map.at(path).type.get() == typeid(Eigen::Vector3d))
                        source = input.template operator()<Eigen::Vector3d>(path);
                    else
                        source = input.template operator()<DirectionVector>(path);
                } else if (term.type == ObservationType::kArray) {
                    source = input.template operator()<std::vector<double>>(path);
                } else {
                    source = input.template operator()<double>(path);
                }
                term_inputs_[i].push_back(source);
            }
        }
    }

    bool build(std::span<const double> previous_action, std::span<double> frame) {
        for (std::size_t i = 0; i < terms_.size(); ++i) {
            const auto& term = terms_[i];
            const std::span<double> source{source_.data(), term.source_size};
            if (!read_source(term, term_inputs_[i], previous_action, source))
                return false;
            for (std::size_t j = 0; j < term.size; ++j)
                if (!preprocess(
                        term, source[term.selection.empty() ? j : term.selection[j]],
                        frame[term.index + j]))
                    return false;
        }
        return true;
    }

private:
    bool read_source(
        const ObservationTerm& term, std::span<Input* const> inputs,
        std::span<const double> previous_action, std::span<double> source) const {
        if (term.type == ObservationType::kPreviousAction) {
            if (previous_action.size() != source.size())
                return false;
            std::ranges::copy(previous_action, source.begin());
            return true;
        }
        if (term.type == ObservationType::kScalars) {
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                const auto& input = std::get<Interface<double>>(*inputs[i]);
                if (!input.ready())
                    return false;
                source[i] = *input;
            }
            return true;
        }
        return std::visit(
            [source](const auto& input) {
                if (!input.ready())
                    return false;
                using Value = std::remove_cvref_t<decltype(*input)>;
                if constexpr (std::is_same_v<Value, double>) {
                    source[0] = *input;
                } else if constexpr (std::is_same_v<Value, DirectionVector>) {
                    std::copy_n(input->vector.data(), 3, source.begin());
                } else {
                    if (static_cast<std::size_t>(input->size()) != source.size())
                        return false;
                    std::ranges::copy(*input, source.begin());
                }
                return true;
            },
            *inputs.front());
    }

    static bool preprocess(const ObservationTerm& term, double value, double& output) {
        const auto quantize = [&term](double v) {
            return term.float32 ? static_cast<double>(static_cast<float>(v)) : v;
        };
        double result = quantize(value);
        if (!std::isfinite(value) || !std::isfinite(result))
            return false;
        if (term.source_clip)
            result = std::clamp(
                result, quantize(term.source_clip->lower), quantize(term.source_clip->upper));
        result = quantize(result * quantize(term.scale));
        if (term.clip)
            result = std::clamp(result, term.clip->lower, term.clip->upper);
        if (!std::isfinite(result))
            return false;
        output = result;
        return true;
    }

    std::vector<ObservationTerm> terms_;
    std::vector<std::unique_ptr<Input>> inputs_;
    std::vector<std::vector<Input*>> term_inputs_;
    std::vector<double> source_;
    std::size_t frame_size_ = 0;
};

} // namespace rmcs_rl
