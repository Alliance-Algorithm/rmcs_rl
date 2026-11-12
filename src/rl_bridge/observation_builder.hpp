#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>

#include "rl_bridge/config.hpp"

namespace rmcs_rl {

class ObservationBuilder {
    using Component = rmcs_executor::Component;
    using DirectionVector = rmcs_description::BaseLink::DirectionVector;

    template <typename T>
    using Input = Component::InputInterface<T>;

public:
    explicit ObservationBuilder(const std::vector<ObservationTerm>& terms)
        : terms_(terms)
        , term_slots_(terms.size()) {
        std::size_t source_size = 0;
        for (const auto& term : terms)
            source_size = std::max(source_size, term.source_size);
        source_.resize(source_size);
    }

    void bind(Component& component, const Component::OutputInfoMap& output_map) {
        for (std::size_t i = 0; i < terms_.size(); ++i) {
            const auto& term = terms_[i];
            auto& slots = term_slots_[i];
            switch (term.type) {
            case ObservationType::kScalar: {
                const auto slot = acquire<double, bool, int, std::size_t>(
                    component, output_map, term.source, !term.default_value);
                if (slot != kNoSlot)
                    slots.push_back(slot);
                break;
            }
            case ObservationType::kVector3:
                slots.push_back(
                    acquire<Eigen::Vector3d, DirectionVector>(
                        component, output_map, term.source, true));
                break;
            case ObservationType::kProjectedGravity:
                slots.push_back(
                    acquire<Eigen::Quaterniond>(component, output_map, term.source, true));
                break;
            case ObservationType::kArray:
                slots.push_back(
                    acquire<std::vector<double>>(component, output_map, term.source, true));
                break;
            case ObservationType::kScalars:
                for (const auto& source : term.sources)
                    slots.push_back(acquire<double>(component, output_map, source, true));
                break;
            case ObservationType::kConstant:
            case ObservationType::kPreviousAction: break;
            }
        }
    }

    [[nodiscard]] const char* source_type(std::size_t term) const {
        const auto& slots = term_slots_[term];
        return slots.empty() ? "default" : slots_[slots.front()].type;
    }

    // Fills the frame in term order; the error names the first term that is unready or non-finite.
    std::expected<void, std::string_view>
        build(std::span<const double> previous_action, std::span<double> frame) {
        for (std::size_t i = 0; i < terms_.size(); ++i) {
            const auto& term = terms_[i];
            const std::span<double> source{source_.data(), term.source_size};
            if (!read_source(term, term_slots_[i], previous_action, source))
                return std::unexpected<std::string_view>{term.name};
            for (std::size_t j = 0; j < term.size; ++j) {
                const double value = source[term.selection.empty() ? j : term.selection[j]];
                if (!preprocess(term, value, frame[term.index + j]))
                    return std::unexpected<std::string_view>{term.name};
            }
        }
        return {};
    }

private:
    struct Slot {
        std::string path;
        const char* type;
        std::variant<
            std::unique_ptr<Input<double>>, std::unique_ptr<Input<bool>>,
            std::unique_ptr<Input<int>>, std::unique_ptr<Input<std::size_t>>,
            std::unique_ptr<Input<Eigen::Vector3d>>, std::unique_ptr<Input<DirectionVector>>,
            std::unique_ptr<Input<Eigen::Quaterniond>>, std::unique_ptr<Input<std::vector<double>>>>
            input;
    };

    static constexpr std::size_t kNoSlot = std::numeric_limits<std::size_t>::max();

    static const char* type_name(const std::type_info& type) {
        if (type == typeid(double))
            return "double";
        if (type == typeid(bool))
            return "bool";
        if (type == typeid(int))
            return "int";
        if (type == typeid(std::size_t))
            return "std::size_t";
        if (type == typeid(Eigen::Vector3d))
            return "Eigen::Vector3d";
        if (type == typeid(DirectionVector))
            return "BaseLink::DirectionVector";
        if (type == typeid(Eigen::Quaterniond))
            return "Eigen::Quaterniond";
        if (type == typeid(std::vector<double>))
            return "std::vector<double>";
        return type.name();
    }

    template <typename... Types>
    std::size_t acquire(
        Component& component, const Component::OutputInfoMap& output_map, const std::string& path,
        bool required) {
        const auto output = output_map.find(path);
        if (output == output_map.end()) {
            if (!required)
                return kNoSlot;
            throw std::runtime_error(
                "RlBridge: required observation interface \"" + path
                + "\" is not produced by any component; check the path or add default");
        }
        if (output->second.kind != rmcs_executor::InterfaceKind::Normal)
            throw std::runtime_error(
                "RlBridge: observation interface \"" + path + "\" must be a Normal interface");

        const std::type_info& type = output->second.type.get();
        if (!((type == typeid(Types)) || ...)) {
            std::string expected;
            ((expected += (expected.empty() ? "" : ", ") + std::string{type_name(typeid(Types))}),
             ...);
            throw std::runtime_error(
                "RlBridge: observation interface \"" + path + "\" has type " + type_name(type)
                + " but the term accepts { " + expected + " }");
        }

        for (std::size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].path == path)
                return i;
        static_cast<void>((emplace_slot<Types>(component, path, type, required) || ...));
        return slots_.size() - 1;
    }

    template <typename T>
    bool emplace_slot(
        Component& component, const std::string& path, const std::type_info& type, bool required) {
        if (type != typeid(T))
            return false;
        auto input = std::make_unique<Input<T>>();
        component.register_input(path, *input, required);
        slots_.push_back(Slot{path, type_name(type), std::move(input)});
        return true;
    }

    template <typename T>
    const T* get(std::size_t slot) const {
        const auto* input = std::get_if<std::unique_ptr<Input<T>>>(&slots_[slot].input);
        if (!input || !(*input)->ready())
            return nullptr;
        return &***input;
    }

    std::optional<double> read_scalar(std::size_t slot) const {
        return std::visit(
            [](const auto& input) -> std::optional<double> {
                using Value = std::remove_cvref_t<decltype(**input)>;
                if constexpr (std::is_arithmetic_v<Value>) {
                    if (input->ready())
                        return static_cast<double>(**input);
                }
                return std::nullopt;
            },
            slots_[slot].input);
    }

    bool read_source(
        const ObservationTerm& term, const std::vector<std::size_t>& slots,
        std::span<const double> previous_action, std::span<double> source) const {
        switch (term.type) {
        case ObservationType::kScalar: {
            if (slots.empty()) {
                source[0] = *term.default_value;
                return true;
            }
            const auto value = read_scalar(slots[0]);
            if (!value)
                return false;
            source[0] = *value;
            return true;
        }
        case ObservationType::kVector3: {
            const auto* vector = get<Eigen::Vector3d>(slots[0]);
            if (const auto* direction = get<DirectionVector>(slots[0]))
                vector = &direction->vector;
            if (!vector)
                return false;
            std::copy_n(vector->data(), 3, source.begin());
            return true;
        }
        case ObservationType::kProjectedGravity: {
            const auto* orientation = get<Eigen::Quaterniond>(slots[0]);
            if (!orientation)
                return false;
            const Eigen::Vector3d gravity = *orientation * Eigen::Vector3d(0.0, 0.0, -1.0);
            std::copy_n(gravity.data(), 3, source.begin());
            return true;
        }
        case ObservationType::kArray: {
            const auto* values = get<std::vector<double>>(slots[0]);
            if (!values || values->size() != source.size())
                return false;
            std::ranges::copy(*values, source.begin());
            return true;
        }
        case ObservationType::kScalars:
            for (std::size_t i = 0; i < slots.size(); ++i) {
                const auto value = read_scalar(slots[i]);
                if (!value)
                    return false;
                source[i] = *value;
            }
            return true;
        case ObservationType::kConstant:
            std::ranges::copy(term.constants, source.begin());
            return true;
        case ObservationType::kPreviousAction:
            if (previous_action.size() != source.size())
                return false;
            std::ranges::copy(previous_action, source.begin());
            return true;
        }
        return false;
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
        result = quantize(result + quantize(term.offset));
        if (term.clip)
            result = std::clamp(result, term.clip->lower, term.clip->upper);
        if (!std::isfinite(result))
            return false;
        output = result;
        return true;
    }

    const std::vector<ObservationTerm>& terms_;
    std::vector<std::vector<std::size_t>> term_slots_;
    std::vector<Slot> slots_;
    std::vector<double> source_;
};

} // namespace rmcs_rl
