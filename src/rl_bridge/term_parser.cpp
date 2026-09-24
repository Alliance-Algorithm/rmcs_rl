#include "rl_bridge/term_parser.hpp"

#include <map>
#include <stdexcept>

#include "rl_bridge/utility.hpp"

namespace rmcs_rl {

namespace {

ObsTerm finish_common(
    ObsTerm term, const std::map<std::string, std::string>& tokens, const std::string& spec) {
    parse_index(tokens, spec, term.has_index, term.index);
    term.scale = double_token(tokens, "scale", 1.0, spec);
    if (term.scale == 0.0)
        throw std::invalid_argument("RlBridge: scale=0 is not allowed (term '" + spec + "')");
    parse_clip(tokens, spec, term.has_clip, term.clip_min, term.clip_max);

    const auto default_value = tokens.find("default");
    if (default_value != tokens.end()) {
        if (!(term.kind == TermKind::kPath && term.take == Take::kScalar))
            throw std::invalid_argument(
                "RlBridge: default= is only valid for scalar path "
                "terms "
                "(term '"
                + spec + "')");
        term.default_value = parse_double(default_value->second, "default");
        if (!is_finite(term.default_value))
            throw std::invalid_argument(
                "RlBridge: default must be finite (term '" + spec + "')");
        term.has_default = true;
    }

    const auto name = tokens.find("name");
    if (name != tokens.end()) {
        if (name->second.empty())
            throw std::invalid_argument("RlBridge: name= must not be empty (term '" + spec + "')");
        term.id = name->second;
    }
    return term;
}

} // namespace

ObsTerm parse_obs_term(const std::string& spec, const TermParseContext& context) {
    if (spec.empty())
        throw std::invalid_argument("RlBridge: observation term must not be empty");
    const auto tokens = parse_tokens(spec);
    ObsTerm term;

    std::string type = string_token(tokens, "type", "");

    if (type == "joint_pos" || type == "joint_vel" || type == "joint_torque") {
        term.kind = type == "joint_pos" ? TermKind::kJointPos
                  : type == "joint_vel" ? TermKind::kJointVel
                                        : TermKind::kJointTorque;
        const auto joints = tokens.find("joints");
        if (joints == tokens.end() || joints->second.empty())
            throw std::invalid_argument(
                "RlBridge: joint_* term requires joints=a,b (term '" + spec + "')");
        for (const auto& name : split_by(joints->second, ',')) {
            if (name.empty())
                throw std::invalid_argument(
                    "RlBridge: empty joint name in joints= (term '" + spec + "')");
            const auto iter = context.joints.index.find(name);
            if (iter == context.joints.index.end())
                throw std::invalid_argument(
                    "RlBridge: unknown joint '" + name + "' (term '" + spec + "')");
            term.joints.push_back(iter->second);
            term.joint_names.push_back(name);
        }
        term.relative = bool_token(tokens, "relative", false, spec);
        term.zero = bool_token(tokens, "zero", false, spec);
        if (term.kind != TermKind::kJointPos && (term.relative || term.zero))
            throw std::invalid_argument(
                "RlBridge: relative/zero are only valid for joint_pos (term '" + spec + "')");
        if (term.relative && term.zero)
            throw std::invalid_argument(
                "RlBridge: joint_pos cannot be both relative and zero (term '" + spec + "')");
        for (const auto joint : term.joints) {
            if (term.kind == TermKind::kJointPos && term.relative) {
                const auto base = default_joint_pos(context.node, context.joints, joint);
                if (!base.has_value())
                    throw std::invalid_argument(
                        "RlBridge: joint_pos relative term '" + spec
                        + "' requires default_joint_pos." + context.joints.names[joint]);
                term.joint_defaults.push_back(*base);
            } else {
                term.joint_defaults.push_back(0.0);
            }
        }
        term.dim = term.joints.size();
        const std::string flag = term.zero ? "zero" : (term.relative ? "rel" : "abs");
        const std::string prefix = type == "joint_pos" ? "joint_pos"
                                 : type == "joint_vel" ? "joint_vel"
                                                       : "joint_torque";
        term.id = prefix + ":" + flag + ":" + join(term.joint_names, "+");
        validate_tokens(
            tokens, {"type", "joints", "relative", "zero", "index", "scale", "clip", "name"}, spec);
        return finish_common(term, tokens, spec);
    }
    if (type == "last_action") {
        term.kind = TermKind::kLastAction;
        if (const auto indices = tokens.find("indices"); indices != tokens.end())
            term.action_indices = parse_index_list(indices->second, context.action_size, "last_action");
        else
            for (std::size_t i = 0; i < context.action_size; ++i)
                term.action_indices.push_back(i);
        term.dim = term.action_indices.size();
        std::vector<std::string> names;
        for (const auto index : term.action_indices)
            names.push_back(std::to_string(index));
        term.id = "last_action:" + join(names, "+");
        validate_tokens(tokens, {"type", "indices", "index", "scale", "clip", "name"}, spec);
        return finish_common(term, tokens, spec);
    }
    if (type == "constant") {
        term.kind = TermKind::kConstant;
        const auto value = tokens.find("value");
        if (value == tokens.end() || value->second.empty())
            throw std::invalid_argument(
                "RlBridge: constant term requires value=... (term '" + spec + "')");
        for (const auto& piece : split_by(value->second, ','))
            term.constants.push_back(parse_double(piece, "constant value"));
        term.dim = term.constants.size();
        std::vector<std::string> names;
        for (const double item : term.constants)
            names.push_back(format_number(item));
        term.id = "constant:" + join(names, "+");
        validate_tokens(tokens, {"type", "value", "index", "scale", "clip", "name"}, spec);
        return finish_common(term, tokens, spec);
    }

    const auto path = tokens.find("path");
    if (path == tokens.end() || path->second.empty())
        throw std::invalid_argument(
            "RlBridge: term requires path=... (term '" + spec
            + "') unless it is joint_*/last_action/constant");
    term.kind = TermKind::kPath;
    term.path = path->second;
    const std::string take = string_token(tokens, "take", "");
    const std::string transform = string_token(tokens, "transform", "");

    if (transform == "projected_gravity") {
        if (!take.empty() && take != "gravity")
            throw std::invalid_argument(
                "RlBridge: transform=projected_gravity conflicts with "
                "take="
                + take + " (term '" + spec + "')");
        term.take = Take::kGravity;
        term.id = "gravity:" + term.path;
        term.dim = 3;
        if (!type.empty()) {
            if (type != "quaternion")
                throw std::invalid_argument(
                    "RlBridge: projected_gravity requires a "
                    "quaternion interface; type="
                    + type + " is not quaternion (term '" + spec + "')");
            term.has_binding = true;
            term.binding = Binding::kQuaternion;
        }
    } else if (transform == "gravity") {
        throw std::invalid_argument(
            "RlBridge: transform=gravity is not supported; use "
            "transform=projected_gravity "
            "(term '"
            + spec + "')");
    } else if (!transform.empty()) {
        throw std::invalid_argument(
            "RlBridge: unknown transform='" + transform + "' (term '" + spec + "')");
    } else if (take.empty()) {
        term.take = Take::kScalar;
        term.id = term.path;
        term.dim = 1;
    } else if (take == "x" || take == "y" || take == "z") {
        term.take = Take::kComponent;
        term.component = std::string{"xyz"}.find(take);
        term.id = "vec3c:" + term.path + ":" + take;
        term.dim = 1;
    } else if (take == "vec3" || take == "vec" || take == "all" || take == "vector") {
        term.take = Take::kVector;
        term.id = "vec3:" + term.path;
        term.dim = 3;
    } else if (take == "gravity") {
        term.take = Take::kGravity;
        term.id = "gravity:" + term.path;
        term.dim = 3;
    } else {
        throw std::invalid_argument(
            "RlBridge: unknown take='" + take
            + "' (use x|y|z|vec3|gravity, or omit for scalar) (term '" + spec + "')");
    }

    if (!type.empty()) {
        term.has_binding = true;
        if (type == "scalar" || type == "double") {
            term.binding = Binding::kDouble;
        } else if (type == "bool") {
            term.binding = Binding::kBool;
        } else if (type == "int") {
            term.binding = Binding::kInt;
        } else if (type == "size" || type == "size_t") {
            term.binding = Binding::kSize;
        } else if (type == "vector3" || type == "vec3") {
            term.binding = Binding::kVector3;
        } else if (type == "direction_vector") {
            term.binding = Binding::kDirectionVector;
        } else if (type == "quaternion") {
            term.binding = Binding::kQuaternion;
        } else {
            throw std::invalid_argument(
                "RlBridge: unknown type='" + type + "' (term '" + spec + "')");
        }
        const bool scalar_binding = term.binding == Binding::kDouble || term.binding == Binding::kBool
                                  || term.binding == Binding::kInt || term.binding == Binding::kSize;
        const bool vector_binding =
            term.binding == Binding::kVector3 || term.binding == Binding::kDirectionVector;
        if (term.take == Take::kScalar && !scalar_binding)
            throw std::invalid_argument(
                "RlBridge: type=" + type
                + " needs take=vec3 or take=x|y|z (a whole vector is not a scalar) (term '"
                + spec + "')");
        if ((term.take == Take::kComponent || term.take == Take::kVector) && !vector_binding)
            throw std::invalid_argument(
                "RlBridge: take=" + take
                + " needs a vector interface "
                  "(type=vector3|direction_vector), got type="
                + type + " (term '" + spec + "')");
        if (term.take == Take::kGravity && term.binding != Binding::kQuaternion)
            throw std::invalid_argument(
                "RlBridge: transform=projected_gravity needs "
                "type=quaternion (term '"
                + spec + "')");
    }

    validate_tokens(
        tokens, {"path", "take", "transform", "type", "index", "scale", "clip", "default", "name"},
        spec);
    return finish_common(term, tokens, spec);
}

ActionTerm parse_action_term(const std::string& spec) {
    ActionTerm term;
    const auto tokens = parse_tokens(spec);
    const auto index = tokens.find("index");
    if (index == tokens.end())
        throw std::invalid_argument("RlBridge: action term requires index= (term '" + spec + "')");
    const auto index_value = parse_integer(index->second, "action index");
    if (index_value < 0)
        throw std::invalid_argument("RlBridge: action index must be >= 0 (term '" + spec + "')");
    term.index = static_cast<std::size_t>(index_value);

    const auto output = tokens.find("output");
    if (output == tokens.end() || output->second.empty())
        throw std::invalid_argument(
            "RlBridge: action term requires output=<interface path> (term '" + spec + "')");
    term.output = output->second;
    term.id = string_token(tokens, "name", term.output);
    if (term.id.empty())
        throw std::invalid_argument("RlBridge: action name= must not be empty (term '" + spec + "')");
    term.scale = double_token(tokens, "scale", 1.0, spec);
    if (term.scale == 0.0)
        throw std::invalid_argument("RlBridge: action scale=0 is not allowed (term '" + spec + "')");
    parse_clip(tokens, spec, term.has_clip, term.clip_min, term.clip_max);
    validate_tokens(tokens, {"index", "output", "name", "scale", "clip"}, spec);
    return term;
}

void assign_observation_indices(std::vector<ObsTerm>& terms) {
    std::size_t cursor = 0;
    for (auto& term : terms) {
        if (term.has_index && term.index != cursor)
            throw std::invalid_argument(
                "RlBridge: observation term '" + term.id
                + "' declares "
                  "index="
                + std::to_string(term.index) + " but the running offset is "
                + std::to_string(cursor) + " (index= is an assertion, not a reorder)");
        term.index = cursor;
        cursor += term.dim;
    }
}

} // namespace rmcs_rl
