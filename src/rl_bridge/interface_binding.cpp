#include "rl_bridge/interface_binding.hpp"

#include <cmath>
#include <stdexcept>

#include "rl_bridge/utility.hpp"

namespace rmcs_rl {

const char* binding_name(Binding binding) {
    switch (binding) {
    case Binding::kDouble: return "double";
    case Binding::kBool: return "bool";
    case Binding::kInt: return "int";
    case Binding::kSize: return "std::size_t";
    case Binding::kVector3: return "Eigen::Vector3d";
    case Binding::kDirectionVector: return "BaseLink::DirectionVector";
    case Binding::kQuaternion: return "Eigen::Quaterniond";
    }
    return "unknown";
}

bool binding_matches_type(Binding binding, const std::type_info& type) {
    switch (binding) {
    case Binding::kDouble: return type == typeid(double);
    case Binding::kBool: return type == typeid(bool);
    case Binding::kInt: return type == typeid(int);
    case Binding::kSize: return type == typeid(std::size_t);
    case Binding::kVector3: return type == typeid(Eigen::Vector3d);
    case Binding::kDirectionVector:
        return type == typeid(rmcs_description::BaseLink::DirectionVector);
    case Binding::kQuaternion: return type == typeid(Eigen::Quaterniond);
    }
    return false;
}

std::size_t acquire_slot(
    rmcs_executor::Component& component, const std::string& path,
    const std::vector<Binding>& candidates, bool required,
    const rmcs_executor::Component::OutputInfoMap& output_map, std::vector<Slot>& slots,
    const char* context) {
    const auto output = output_map.find(path);
    if (output == output_map.end()) {
        if (!required)
            return kNoSlot;
        throw std::runtime_error(
            "RlBridge: required input interface \"" + path
            + "\" was not produced by any component (" + context
            + "); check the interface path or add default= to make it optional");
    }
    if (output->second.kind != rmcs_executor::InterfaceKind::Normal)
        throw std::runtime_error(
            "RlBridge: input interface \"" + path
            + "\" exists but is an Event interface; Normal required (" + context + ")");
    const std::type_info& producer_type = output->second.type.get();

    for (std::size_t i = 0; i < slots.size(); ++i)
        if (slots[i].path == path && binding_matches_type(slots[i].binding, producer_type))
            return i;

    Binding selected = Binding::kDouble;
    bool found = false;
    for (const auto binding : candidates)
        if (binding_matches_type(binding, producer_type)) {
            selected = binding;
            found = true;
            break;
        }
    if (!found) {
        std::vector<std::string> expected;
        for (const auto binding : candidates)
            expected.push_back(binding_name(binding));
        throw std::runtime_error(
            "RlBridge: cannot bind observation interface \"" + path + "\": producer declares \""
            + pretty_type(producer_type) + "\" but the term accepts { " + join(expected, ", ")
            + " }. Either fix the term (take=/transform=) or pin type= explicitly.");
    }

    Slot slot;
    slot.path = path;
    slot.binding = selected;
    slot.required = required;
    switch (selected) {
    case Binding::kDouble:
        slot.double_value = std::make_unique<rmcs_executor::Component::InputInterface<double>>();
        component.register_input(path, *slot.double_value, required);
        break;
    case Binding::kBool:
        slot.bool_value = std::make_unique<rmcs_executor::Component::InputInterface<bool>>();
        component.register_input(path, *slot.bool_value, required);
        break;
    case Binding::kInt:
        slot.int_value = std::make_unique<rmcs_executor::Component::InputInterface<int>>();
        component.register_input(path, *slot.int_value, required);
        break;
    case Binding::kSize:
        slot.size_value =
            std::make_unique<rmcs_executor::Component::InputInterface<std::size_t>>();
        component.register_input(path, *slot.size_value, required);
        break;
    case Binding::kVector3:
        slot.vector3_value =
            std::make_unique<rmcs_executor::Component::InputInterface<Eigen::Vector3d>>();
        component.register_input(path, *slot.vector3_value, required);
        break;
    case Binding::kDirectionVector:
        slot.direction_vector_value = std::make_unique<
            rmcs_executor::Component::InputInterface<rmcs_description::BaseLink::DirectionVector>>();
        component.register_input(path, *slot.direction_vector_value, required);
        break;
    case Binding::kQuaternion:
        slot.quaternion_value =
            std::make_unique<rmcs_executor::Component::InputInterface<Eigen::Quaterniond>>();
        component.register_input(path, *slot.quaternion_value, required);
        break;
    }
    slots.push_back(std::move(slot));
    return slots.size() - 1;
}

bool read_double(const Slot& slot, double& value) {
    switch (slot.binding) {
    case Binding::kDouble:
        if (!slot.double_value->ready())
            return false;
        value = **slot.double_value;
        return true;
    case Binding::kBool:
        if (!slot.bool_value->ready())
            return false;
        value = **slot.bool_value ? 1.0 : 0.0;
        return true;
    case Binding::kInt:
        if (!slot.int_value->ready())
            return false;
        value = static_cast<double>(**slot.int_value);
        return true;
    case Binding::kSize:
        if (!slot.size_value->ready())
            return false;
        value = static_cast<double>(**slot.size_value);
        return true;
    default: return false;
    }
}

bool read_unsigned(const Slot& slot, std::uint64_t& value) {
    double raw = 0.0;
    if (!read_double(slot, raw))
        return false;
    if (!std::isfinite(raw) || raw < 0.0)
        return false;
    value = static_cast<std::uint64_t>(raw);
    return true;
}

} // namespace rmcs_rl
