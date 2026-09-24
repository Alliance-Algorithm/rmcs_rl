
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/subscription.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_rl/msg/action.hpp>
#include <rmcs_rl/msg/observation.hpp>

#include "rl_layout.hpp"

namespace rmcs_rl {

namespace {

constexpr std::size_t kNoSlot = std::numeric_limits<std::size_t>::max();

std::vector<std::string> split_by(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::string current;
    std::istringstream stream{text};
    while (std::getline(stream, current, delimiter))
        parts.push_back(current);
    return parts;
}

std::vector<std::string> split_whitespace(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream{text};
    std::string token;
    while (stream >> token)
        tokens.push_back(token);
    return tokens;
}

std::string format_number(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return buffer;
}

std::string join(const std::vector<std::string>& parts, const char* separator) {
    std::string text;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i != 0)
            text += separator;
        text += parts[i];
    }
    return text;
}

double parse_double(const std::string& text, const std::string& context) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size())
            throw std::invalid_argument("trailing characters");
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("RlBridge: '" + text + "' is not a number (" + context + ")");
    }
}

std::int64_t parse_integer(const std::string& text, const std::string& context) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoll(text, &consumed);
        if (consumed != text.size())
            throw std::invalid_argument("trailing characters");
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument("RlBridge: '" + text + "' is not an integer (" + context + ")");
    }
}

bool parse_boolean(const std::string& text, const std::string& context) {
    if (text == "true" || text == "1")
        return true;
    if (text == "false" || text == "0")
        return false;
    throw std::invalid_argument(
        "RlBridge: '" + text + "' is not a boolean (true/false) (" + context + ")");
}

bool is_finite(double value) { return std::isfinite(value); }

std::string pretty_type(const std::type_info& type) {
    if (type == typeid(double))
        return "double";
    if (type == typeid(float))
        return "float";
    if (type == typeid(bool))
        return "bool";
    if (type == typeid(int))
        return "int";
    if (type == typeid(std::size_t))
        return "std::size_t";
    if (type == typeid(Eigen::Vector3d))
        return "Eigen::Vector3d";
    if (type == typeid(rmcs_description::BaseLink::DirectionVector))
        return "rmcs_description::BaseLink::DirectionVector";
    if (type == typeid(Eigen::Quaterniond))
        return "Eigen::Quaterniond";
    return std::string{type.name()} + " (unknown)";
}

} // namespace

class RlBridge final
    : public rmcs_executor::Component
    , public rclcpp::Node {

    enum class TermKind { kPath, kJointPos, kJointVel, kJointTorque, kLastAction, kConstant };

    enum class Take { kScalar, kComponent, kVector, kGravity };

    enum class Binding {
        kDouble,
        kBool,
        kInt,
        kSize,
        kVector3,
        kDirectionVector,
        kQuaternion,
    };

    struct ObsTerm {
        TermKind kind = TermKind::kPath;
        Take take = Take::kScalar;

        std::string path;
        std::string id;
        std::size_t dim = 1;

        std::size_t index = 0;
        bool has_index = false;
        int component = 0;
        bool has_binding = false;
        Binding binding = Binding::kDouble;

        std::vector<std::size_t> joints;
        std::vector<std::string> joint_names;
        std::vector<std::size_t> joint_slots;
        std::vector<double> joint_defaults;
        bool relative = false;
        bool zero = false;

        std::vector<std::size_t> action_indices;
        std::vector<double> constants;

        double scale = 1.0;
        bool has_clip = false;
        double clip_min = 0.0;
        double clip_max = 0.0;
        bool has_default = false;
        double default_value = 0.0;

        std::size_t slot = kNoSlot;
    };

    struct ActionTerm {
        std::size_t index = 0;
        std::string output;
        std::string id;
        double scale = 1.0;
        bool has_clip = false;
        double clip_min = 0.0;
        double clip_max = 0.0;
    };

    struct Slot {
        std::string path;
        Binding binding = Binding::kDouble;
        bool required = true;
        std::unique_ptr<InputInterface<double>> double_value;
        std::unique_ptr<InputInterface<bool>> bool_value;
        std::unique_ptr<InputInterface<int>> int_value;
        std::unique_ptr<InputInterface<std::size_t>> size_value;
        std::unique_ptr<InputInterface<Eigen::Vector3d>> vector3_value;
        std::unique_ptr<InputInterface<rmcs_description::BaseLink::DirectionVector>>
            direction_vector_value;
        std::unique_ptr<InputInterface<Eigen::Quaterniond>> quaternion_value;
    };

    struct ActionSnapshot {
        std::uint64_t obs_seq = 0;
        std::uint64_t layout_hash = 0;
        std::uint64_t model_id = 0;
        std::vector<double> action;
        std::chrono::steady_clock::time_point received{};
    };

public:
    explicit RlBridge()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {

        rl_base_ = string_or_("rl_base", "/rl");

        load_joint_config_();

        const auto action_specs = string_array_or_("action_terms");
        if (action_specs.empty())
            throw std::invalid_argument("RlBridge: required parameter 'action_terms' is missing");
        std::unordered_set<std::string> output_paths;
        std::set<std::size_t> action_indices;
        for (const auto& spec : action_specs) {
            auto term = parse_action_term_(spec);
            if (!output_paths.emplace(term.output).second)
                throw std::invalid_argument(
                    "RlBridge: two action terms map to the same output "
                    "interface '"
                    + term.output + "'");
            if (!action_indices.emplace(term.index).second)
                throw std::invalid_argument(
                    "RlBridge: duplicate action index " + std::to_string(term.index)
                    + " (each slot must appear exactly once)");
            action_terms_.push_back(std::move(term));
        }
        for (std::size_t i = 0; i < action_terms_.size(); ++i)
            if (action_terms_[i].index != i)
                throw std::invalid_argument(
                    "RlBridge: action indices must form the complete "
                    "permutation 0.."
                    + std::to_string(action_terms_.size() - 1) + "; got index="
                    + std::to_string(action_terms_[i].index) + " at position " + std::to_string(i));
        action_size_ = action_terms_.size();
        if (const auto declared = integer_or_("rl_action_size");
            declared.has_value() && static_cast<std::size_t>(*declared) != action_size_)
            throw std::invalid_argument(
                "RlBridge: rl_action_size=" + std::to_string(*declared) + " but action_terms has "
                + std::to_string(action_size_) + " entries");
        last_actions_.assign(action_size_, 0.0);
        written_.assign(action_size_, 0.0);

        for (const auto& term : action_terms_) {
            action_outputs_.push_back(std::make_unique<OutputInterface<double>>());
            register_output(term.output, *action_outputs_.back(), 0.0);
            own_output_paths_.insert(term.output);
        }

        const auto observation_specs = string_array_or_("observation_terms");
        if (observation_specs.empty())
            throw std::invalid_argument(
                "RlBridge: required parameter 'observation_terms' is "
                "missing");
        for (const auto& spec : observation_specs)
            obs_terms_.push_back(parse_obs_term_(spec));

        std::size_t cursor = 0;
        for (auto& term : obs_terms_) {
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
        obs_frame_size_ = cursor;
        const auto history_length = integer_or_("history_length").value_or(1);
        if (history_length < 1 || history_length > 64)
            throw std::invalid_argument("RlBridge: history_length must be in [1, 64]");
        history_length_ = static_cast<std::size_t>(history_length);
        obs_size_ = obs_frame_size_ * history_length_;
        if (const auto declared = integer_or_("rl_obs_size");
            declared.has_value() && static_cast<std::size_t>(*declared) != obs_size_)
            throw std::invalid_argument(
                "RlBridge: rl_obs_size=" + std::to_string(*declared)
                + " but observation frame/history contract is " + std::to_string(obs_frame_size_)
                + "x" + std::to_string(history_length_) + "=" + std::to_string(obs_size_));

        policy_rate_ = number_or_("policy_rate", 50.0);
        if (!(policy_rate_ > 0.0) || !is_finite(policy_rate_))
            throw std::invalid_argument("RlBridge: policy_rate must be finite and > 0");
        pub_period_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / policy_rate_));
        max_action_age_ = number_or_("max_action_age", 2.0 / policy_rate_);
        if (!(max_action_age_ > 0.0) || !is_finite(max_action_age_))
            throw std::invalid_argument("RlBridge: max_action_age must be finite and > 0");
        expected_model_id_ = parse_u64_(string_or_("expected_model_id", "0"), "expected_model_id");

        const std::string invalid = string_or_("invalid_value", "nan");
        if (invalid == "nan")
            invalid_mode_ = InvalidMode::kNaN;
        else if (invalid == "zero")
            invalid_mode_ = InvalidMode::kZero;
        else if (invalid == "hold")
            invalid_mode_ = InvalidMode::kHold;
        else
            throw std::invalid_argument(
                "RlBridge: invalid_value must be nan|zero|hold "
                "(quote it in YAML: invalid_value: \"nan\" — an unquoted "
                "nan is parsed as a float), got '"
                + string_or_("invalid_value", "<non-string>") + "'");

        enable_path_ = string_or_("enable_interface", rl_base_ + "/enable");
        enable_default_ = bool_or_("enable_default", false);
        reset_path_ = string_or_("reset_interface", "");

        register_output(rl_base_ + "/valid", valid_output_, 0.0);
        register_output(rl_base_ + "/healthy", healthy_output_, 0.0);
        register_output(
            rl_base_ + "/action_age", action_age_output_, std::numeric_limits<double>::quiet_NaN());
        register_output(rl_base_ + "/obs_seq", obs_seq_output_, std::size_t{0});
        own_output_paths_.insert(rl_base_ + "/valid");
        own_output_paths_.insert(rl_base_ + "/healthy");
        own_output_paths_.insert(rl_base_ + "/action_age");
        own_output_paths_.insert(rl_base_ + "/obs_seq");

        obs_publisher_ = create_publisher<rmcs_rl::msg::Observation>(
            rl_base_ + "/obs", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort());
        action_subscription_ = create_subscription<rmcs_rl::msg::Action>(
            rl_base_ + "/action", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort(),
            [this](rmcs_rl::msg::Action::UniquePtr message) { on_action_(std::move(message)); });

        incoming_.action.assign(action_size_, 0.0);
        read_snapshot_.action.assign(action_size_, 0.0);
    }

    void before_pairing(const OutputInfoMap& output_map) override {
        for (auto& term : obs_terms_) {
            switch (term.kind) {
            case TermKind::kConstant:
            case TermKind::kLastAction: break;
            case TermKind::kJointPos:
            case TermKind::kJointVel:
            case TermKind::kJointTorque: {
                for (std::size_t j = 0; j < term.joints.size(); ++j) {
                    const auto& joint = joint_names_[term.joints[j]];
                    const char field = term.kind == TermKind::kJointPos ? 'a'
                                     : term.kind == TermKind::kJointVel ? 'v'
                                                                        : 't';
                    const auto path = joint_path_(joint, field);
                    term.joint_slots.push_back(
                        acquire_slot_(path, {Binding::kDouble}, true, output_map, "joint term"));
                }
                break;
            }
            case TermKind::kPath: {
                std::vector<Binding> candidates;
                switch (term.take) {
                case Take::kScalar:
                    candidates = {Binding::kDouble, Binding::kBool, Binding::kInt, Binding::kSize};
                    break;
                case Take::kComponent:
                case Take::kVector:
                    candidates = {Binding::kVector3, Binding::kDirectionVector};
                    break;
                case Take::kGravity: candidates = {Binding::kQuaternion}; break;
                }
                term.slot = acquire_slot_(
                    term.path, candidates, !term.has_default, output_map, "observation term");
                break;
            }
            }
        }

        if (!enable_path_.empty())
            enable_slot_ = acquire_slot_(
                enable_path_, {Binding::kBool, Binding::kDouble}, false, output_map,
                "enable interface");

        if (!reset_path_.empty())
            reset_slot_ = acquire_slot_(
                reset_path_, {Binding::kSize, Binding::kInt, Binding::kDouble}, false, output_map,
                "reset interface");

        for (const auto& slot : slots_)
            if (own_output_paths_.count(slot->path) != 0)
                throw std::runtime_error(
                    "RlBridge: interface \"" + slot->path
                    + "\" is produced by RlBridge itself (self reference)");

        obs_signature_ = obs_layout_signature_();
        actions_signature_ = actions_layout_signature_();
        layout_hash_ =
            rmcs_rl::layout_hash(obs_signature_, actions_signature_, obs_size_, action_size_);

        log_layout_();
        RCLCPP_INFO(
            get_logger(), "contract: obs_size=%zu action_size=%zu policy_rate=%.3f Hz", obs_size_,
            action_size_, policy_rate_);
        RCLCPP_INFO(
            get_logger(), "layout_hash=%s (obs/action signatures below)",
            hex16(layout_hash_).c_str());
        RCLCPP_INFO(get_logger(), "  obs signature    : %s", obs_signature_.c_str());
        RCLCPP_INFO(get_logger(), "  action signature : %s", actions_signature_.c_str());
        RCLCPP_INFO(
            get_logger(), "topics: %s/obs -> %s/action (best_effort, keep_last=1)",
            rl_base_.c_str(), rl_base_.c_str());
    }

    void update() override {
        const auto now = std::chrono::steady_clock::now();

        if (reset_slot_ != kNoSlot) {
            std::uint64_t reset_count = 0;
            if (read_unsigned_(reset_slot_, reset_count) && reset_count != last_reset_count_) {
                last_reset_count_ = reset_count;
                reset_runtime_();
            }
        }

        if (!pub_started_ || now - last_pub_time_ >= pub_period_) {
            std::vector<double> obs;
            if (build_observation_(obs)) {
                if (history_.empty())
                    history_.assign(history_length_, obs);
                else {
                    history_.pop_front();
                    history_.push_back(obs);
                }
                std::vector<double> stacked;
                stacked.reserve(obs_size_);
                for (const auto& frame : history_)
                    stacked.insert(stacked.end(), frame.begin(), frame.end());
                publish_observation_(stacked, now);
                ++pub_ok_count_;
            } else {
                ++obs_invalid_count_;
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 1000,
                    "observation sources not ready or non-finite; no obs published (%llu frames "
                    "skipped)",
                    static_cast<unsigned long long>(obs_invalid_count_));
            }
        }

        ActionSnapshot& snapshot = read_snapshot_;
        const bool has_snapshot = try_read_action_(snapshot);

        double age = std::numeric_limits<double>::quiet_NaN();
        bool seq_ok = false;
        bool finite = true;
        if (has_snapshot) {
            age = obs_age_of_(snapshot.obs_seq, now);
            seq_ok = (snapshot.obs_seq == pub_seq_) || (snapshot.obs_seq + 1 == pub_seq_);
            finite = std::all_of(snapshot.action.begin(), snapshot.action.end(), [](double value) {
                return std::isfinite(value);
            });
            if (snapshot.layout_hash != layout_hash_) {
                if (contract_ok_) {
                    contract_ok_ = false;
                    RCLCPP_FATAL(
                        get_logger(),
                        "contract mismatch: action layout_hash=%s != bridge layout_hash=%s "
                        "(policy process and bridge disagree; refusing to output actions)",
                        hex16(snapshot.layout_hash).c_str(), hex16(layout_hash_).c_str());
                }
            } else if (expected_model_id_ != 0 && snapshot.model_id != expected_model_id_) {
                if (contract_ok_) {
                    contract_ok_ = false;
                    RCLCPP_FATAL(
                        get_logger(), "model mismatch: action model_id=%s != expected_model_id=%s",
                        hex16(snapshot.model_id).c_str(), hex16(expected_model_id_).c_str());
                }
            }
        }

        const bool enabled = read_enable_();
        const bool fresh = has_snapshot && is_finite(age) && age <= max_action_age_;
        const bool valid = enabled && contract_ok_ && has_snapshot && fresh && seq_ok && finite;

        write_actions_(valid, snapshot);

        if (valid)
            last_actions_ = snapshot.action;

        *valid_output_ = valid ? 1.0 : 0.0;
        *healthy_output_ = (contract_ok_ && fresh) ? 1.0 : 0.0;
        *action_age_output_ = age;
        *obs_seq_output_ = pub_seq_;

        if (valid != last_valid_) {
            if (valid) {
                RCLCPP_INFO(
                    get_logger(), "valid=1 (obs_seq=%llu model_id=%s action_age=%.4f s)",
                    static_cast<unsigned long long>(snapshot.obs_seq),
                    hex16(snapshot.model_id).c_str(), age);
            } else {
                RCLCPP_WARN(
                    get_logger(), "valid=0: %s",
                    invalid_reason_(enabled, contract_ok_, has_snapshot, fresh, seq_ok, finite)
                        .c_str());
            }
            last_valid_ = valid;
        }
    }

private:
    enum class InvalidMode { kNaN, kZero, kHold };

    std::optional<double> number_(const std::string& name) {
        rclcpp::Parameter parameter;
        try {
            if (!get_parameter(name, parameter))
                return std::nullopt;
        } catch (const std::exception&) {
            return std::nullopt;
        }
        switch (parameter.get_type()) {
        case rclcpp::ParameterType::PARAMETER_DOUBLE: return parameter.as_double();
        case rclcpp::ParameterType::PARAMETER_INTEGER:
            return static_cast<double>(parameter.as_int());
        case rclcpp::ParameterType::PARAMETER_NOT_SET: return std::nullopt;
        default: throw std::invalid_argument("RlBridge: parameter '" + name + "' must be a number");
        }
    }

    double number_or_(const std::string& name, double fallback) {
        const auto value = number_(name);
        if (!value.has_value())
            return fallback;
        if (!is_finite(*value))
            throw std::invalid_argument("RlBridge: parameter '" + name + "' is not finite");
        return *value;
    }

    static std::uint64_t parse_u64_(const std::string& text, const std::string& name) {
        try {
            std::size_t consumed = 0;
            const bool hex = text.rfind("0x", 0) == 0 || text.rfind("0X", 0) == 0;
            const auto value = std::stoull(hex ? text.substr(2) : text, &consumed, hex ? 16 : 10);
            if (consumed != (hex ? text.size() - 2 : text.size()))
                throw std::invalid_argument("trailing characters");
            return static_cast<std::uint64_t>(value);
        } catch (const std::exception&) {
            throw std::invalid_argument(
                "RlBridge: parameter '" + name
                + "' must be a decimal or 0x-prefixed 64-bit id, got '" + text + "'");
        }
    }

    std::optional<std::int64_t> integer_or_(const std::string& name) {
        const auto value = number_(name);
        if (!value.has_value())
            return std::nullopt;
        const auto rounded = std::llround(*value);
        if (std::abs(*value - static_cast<double>(rounded)) > 1e-9)
            throw std::invalid_argument("RlBridge: parameter '" + name + "' must be an integer");
        return rounded;
    }

    bool bool_or_(const std::string& name, bool fallback) {
        rclcpp::Parameter parameter;
        try {
            if (!get_parameter(name, parameter))
                return fallback;
        } catch (const std::exception&) {
            return fallback;
        }
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
            return fallback;
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL)
            throw std::invalid_argument("RlBridge: parameter '" + name + "' must be a boolean");
        return parameter.as_bool();
    }

    std::string string_or_(const std::string& name, const std::string& fallback) {
        rclcpp::Parameter parameter;
        try {
            if (!get_parameter(name, parameter))
                return fallback;
        } catch (const std::exception&) {
            return fallback;
        }
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET)
            return fallback;
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING)
            throw std::invalid_argument("RlBridge: parameter '" + name + "' must be a string");
        return parameter.as_string();
    }

    std::vector<std::string> string_array_or_(const std::string& name) {
        std::vector<std::string> value;
        try {
            if (!get_parameter(name, value))
                return {};
        } catch (const std::exception&) {
            throw std::invalid_argument(
                "RlBridge: parameter '" + name + "' must be a list of strings");
        }
        return value;
    }

    void load_joint_config_() {
        joint_names_ = string_array_or_("joint_names");
        joint_base_path_ = string_or_("joint_base_path", "");
        if (!joint_names_.empty() && joint_base_path_.empty())
            throw std::invalid_argument(
                "RlBridge: joint_base_path is required when joint_names is "
                "set");
        for (std::size_t i = 0; i < joint_names_.size(); ++i) {
            if (joint_names_[i].empty())
                throw std::invalid_argument("RlBridge: joint_names contains an empty name");
            if (!joint_index_.emplace(joint_names_[i], i).second)
                throw std::invalid_argument(
                    "RlBridge: duplicate joint name '" + joint_names_[i] + "'");
        }

        const std::string default_angle = string_or_("joint_angle_suffix", "/angle");
        const std::string default_velocity = string_or_("joint_velocity_suffix", "/velocity");
        const std::string default_torque = string_or_("joint_torque_suffix", "/torque");
        for (const auto& joint : joint_names_) {
            angle_suffix_[joint] = default_angle;
            velocity_suffix_[joint] = default_velocity;
            torque_suffix_[joint] = default_torque;
        }

        constexpr const char* prefix = "joint_interface_overrides.";
        for (const auto& name : list_parameters({"joint_interface_overrides"}, 10).names) {
            if (name.rfind(prefix, 0) != 0)
                continue;
            const std::string remainder = name.substr(std::string{prefix}.size());
            const auto separator = remainder.find_last_of('.');
            if (separator == std::string::npos)
                throw std::invalid_argument(
                    "RlBridge: joint_interface_overrides entry '" + name
                    + "' must be <joint>.<angle|velocity|torque>");
            const std::string joint = remainder.substr(0, separator);
            const std::string field = remainder.substr(separator + 1);
            if (joint_index_.count(joint) == 0)
                throw std::invalid_argument(
                    "RlBridge: joint_interface_overrides references unknown joint '" + joint + "'");
            const std::string value = string_or_(name, "");
            if (field == "angle")
                angle_suffix_[joint] = value;
            else if (field == "velocity")
                velocity_suffix_[joint] = value;
            else if (field == "torque")
                torque_suffix_[joint] = value;
            else
                throw std::invalid_argument(
                    "RlBridge: joint_interface_overrides field '" + field
                    + "' is not angle/velocity/torque");
        }
    }

    std::string joint_path_(const std::string& joint, char field) const {
        const auto suffix = field == 'a' ? angle_suffix_.at(joint)
                          : field == 'v' ? velocity_suffix_.at(joint)
                                         : torque_suffix_.at(joint);
        return joint_base_path_ + "/" + joint + suffix;
    }

    std::optional<double> default_joint_pos_(std::size_t joint) {
        const auto name = "default_joint_pos." + joint_names_[joint];
        const auto value = number_(name);
        if (!value.has_value())
            return std::nullopt;
        if (!is_finite(*value))
            throw std::invalid_argument("RlBridge: parameter '" + name + "' is not finite");
        return value;
    }

    static bool binding_matches_type_(Binding binding, const std::type_info& type) {
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

    static const char* binding_name_(Binding binding) {
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

    std::size_t acquire_slot_(
        const std::string& path, const std::vector<Binding>& candidates, bool required,
        const OutputInfoMap& output_map, const char* context) {
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

        for (std::size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i]->path == path && binding_matches_type_(slots_[i]->binding, producer_type))
                return i;

        Binding selected = Binding::kDouble;
        bool found = false;
        for (const auto binding : candidates)
            if (binding_matches_type_(binding, producer_type)) {
                selected = binding;
                found = true;
                break;
            }
        if (!found) {
            std::vector<std::string> expected;
            for (const auto binding : candidates)
                expected.push_back(binding_name_(binding));
            throw std::runtime_error(
                "RlBridge: cannot bind observation interface \"" + path + "\": producer declares \""
                + pretty_type(producer_type) + "\" but the term accepts { " + join(expected, ", ")
                + " }. Either fix the term (take=/transform=) or pin type= explicitly.");
        }

        auto slot = std::make_unique<Slot>();
        slot->path = path;
        slot->binding = selected;
        slot->required = required;
        switch (selected) {
        case Binding::kDouble:
            slot->double_value = std::make_unique<InputInterface<double>>();
            register_input(path, *slot->double_value, required);
            break;
        case Binding::kBool:
            slot->bool_value = std::make_unique<InputInterface<bool>>();
            register_input(path, *slot->bool_value, required);
            break;
        case Binding::kInt:
            slot->int_value = std::make_unique<InputInterface<int>>();
            register_input(path, *slot->int_value, required);
            break;
        case Binding::kSize:
            slot->size_value = std::make_unique<InputInterface<std::size_t>>();
            register_input(path, *slot->size_value, required);
            break;
        case Binding::kVector3:
            slot->vector3_value = std::make_unique<InputInterface<Eigen::Vector3d>>();
            register_input(path, *slot->vector3_value, required);
            break;
        case Binding::kDirectionVector:
            slot->direction_vector_value =
                std::make_unique<InputInterface<rmcs_description::BaseLink::DirectionVector>>();
            register_input(path, *slot->direction_vector_value, required);
            break;
        case Binding::kQuaternion:
            slot->quaternion_value = std::make_unique<InputInterface<Eigen::Quaterniond>>();
            register_input(path, *slot->quaternion_value, required);
            break;
        }
        slots_.push_back(std::move(slot));
        return slots_.size() - 1;
    }

    static std::map<std::string, std::string> parse_tokens_(const std::string& spec) {
        std::map<std::string, std::string> tokens;
        for (const auto& token : split_whitespace(spec)) {
            const auto equals = token.find('=');
            if (equals == std::string::npos || equals == 0)
                throw std::invalid_argument(
                    "RlBridge: term token '" + token + "' must be key=value (term '" + spec + "')");
            const std::string key = token.substr(0, equals);
            const std::string value = token.substr(equals + 1);
            if (!tokens.emplace(key, value).second)
                throw std::invalid_argument(
                    "RlBridge: duplicate key '" + key + "' in term '" + spec + "'");
        }
        return tokens;
    }

    static void validate_tokens_(
        const std::map<std::string, std::string>& tokens, const std::vector<std::string>& allowed,
        const std::string& spec) {
        for (const auto& [key, ignored] : tokens) {
            (void)ignored;
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                throw std::invalid_argument(
                    "RlBridge: unknown key '" + key + "' in term '" + spec + "'");
        }
    }

    static double double_token_(
        const std::map<std::string, std::string>& tokens, const std::string& key, double fallback,
        const std::string& spec) {
        const auto token = tokens.find(key);
        if (token == tokens.end())
            return fallback;
        const double value = parse_double(token->second, key + " in term '" + spec + "'");
        if (!is_finite(value))
            throw std::invalid_argument(
                "RlBridge: " + key + " must be finite (term '" + spec + "')");
        return value;
    }

    static bool bool_token_(
        const std::map<std::string, std::string>& tokens, const std::string& key, bool fallback,
        const std::string& spec) {
        const auto token = tokens.find(key);
        if (token == tokens.end())
            return fallback;
        return parse_boolean(token->second, key + " in term '" + spec + "'");
    }

    static std::string string_token_(
        const std::map<std::string, std::string>& tokens, const std::string& key,
        const std::string& fallback) {
        const auto token = tokens.find(key);
        if (token == tokens.end())
            return fallback;
        return token->second;
    }

    static void parse_clip_(
        const std::map<std::string, std::string>& tokens, const std::string& spec, bool& has_clip,
        double& clip_min, double& clip_max) {
        const auto clip = tokens.find("clip");
        if (clip == tokens.end())
            return;
        const auto separator = clip->second.find(':');
        if (separator == std::string::npos) {
            const double symmetric = parse_double(clip->second, "clip");
            if (!(symmetric > 0.0) || !is_finite(symmetric))
                throw std::invalid_argument(
                    "RlBridge: single-value clip must be finite and > 0 (term '" + spec + "')");
            clip_min = -symmetric;
            clip_max = symmetric;
        } else {
            clip_min = parse_double(clip->second.substr(0, separator), "clip min");
            clip_max = parse_double(clip->second.substr(separator + 1), "clip max");
        }
        if (!is_finite(clip_min) || !is_finite(clip_max) || clip_min > clip_max)
            throw std::invalid_argument("RlBridge: invalid clip range (term '" + spec + "')");
        has_clip = true;
    }

    static void parse_index_(
        const std::map<std::string, std::string>& tokens, const std::string& spec, bool& has_index,
        std::size_t& index) {
        const auto token = tokens.find("index");
        if (token == tokens.end())
            return;
        const auto value = parse_integer(token->second, "index in term '" + spec + "'");
        if (value < 0)
            throw std::invalid_argument("RlBridge: index must be >= 0 (term '" + spec + "')");
        has_index = true;
        index = static_cast<std::size_t>(value);
    }

    static std::vector<std::size_t>
        parse_index_list_(const std::string& text, std::size_t limit, const std::string& what) {
        std::vector<std::size_t> indices;
        for (const auto& piece : split_by(text, ',')) {
            if (piece.empty())
                throw std::invalid_argument("RlBridge: empty entry in " + what + " '" + text + "'");
            const auto range = piece.find("..");
            if (range == std::string::npos) {
                const auto index = parse_integer(piece, what);
                if (index < 0 || static_cast<std::size_t>(index) >= limit)
                    throw std::invalid_argument(
                        "RlBridge: " + what + " index " + std::to_string(index)
                        + " out of range [0, " + std::to_string(limit) + ")");
                indices.push_back(static_cast<std::size_t>(index));
            } else {
                const auto first = parse_integer(piece.substr(0, range), what);
                const auto last = parse_integer(piece.substr(range + 2), what);
                if (first < 0 || last < first || static_cast<std::size_t>(last) >= limit)
                    throw std::invalid_argument(
                        "RlBridge: invalid " + what + " range '" + piece + "' (limit "
                        + std::to_string(limit) + ")");
                for (auto index = first; index <= last; ++index)
                    indices.push_back(static_cast<std::size_t>(index));
            }
        }
        return indices;
    }

    ObsTerm parse_obs_term_(const std::string& spec) {
        if (spec.empty())
            throw std::invalid_argument("RlBridge: observation term must not be empty");
        const auto tokens = parse_tokens_(spec);
        ObsTerm term;

        std::string type = string_token_(tokens, "type", "");

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
                const auto iter = joint_index_.find(name);
                if (iter == joint_index_.end())
                    throw std::invalid_argument(
                        "RlBridge: unknown joint '" + name + "' (term '" + spec + "')");
                term.joints.push_back(iter->second);
                term.joint_names.push_back(name);
            }
            term.relative = bool_token_(tokens, "relative", false, spec);
            term.zero = bool_token_(tokens, "zero", false, spec);
            if (term.kind != TermKind::kJointPos && (term.relative || term.zero))
                throw std::invalid_argument(
                    "RlBridge: relative/zero are only valid for joint_pos (term '" + spec + "')");
            if (term.relative && term.zero)
                throw std::invalid_argument(
                    "RlBridge: joint_pos cannot be both relative and zero (term '" + spec + "')");
            for (const auto joint : term.joints) {
                if (term.kind == TermKind::kJointPos && term.relative) {
                    const auto base = default_joint_pos_(joint);
                    if (!base.has_value())
                        throw std::invalid_argument(
                            "RlBridge: joint_pos relative term '" + spec
                            + "' requires default_joint_pos." + joint_names_[joint]);
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
            validate_tokens_(
                tokens, {"type", "joints", "relative", "zero", "index", "scale", "clip", "name"},
                spec);
            return finish_common_(term, tokens, spec);
        }
        if (type == "last_action") {
            term.kind = TermKind::kLastAction;
            if (const auto indices = tokens.find("indices"); indices != tokens.end())
                term.action_indices =
                    parse_index_list_(indices->second, action_size_, "last_action");
            else
                for (std::size_t i = 0; i < action_size_; ++i)
                    term.action_indices.push_back(i);
            term.dim = term.action_indices.size();
            std::vector<std::string> names;
            for (const auto index : term.action_indices)
                names.push_back(std::to_string(index));
            term.id = "last_action:" + join(names, "+");
            validate_tokens_(tokens, {"type", "indices", "index", "scale", "clip", "name"}, spec);
            return finish_common_(term, tokens, spec);
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
            validate_tokens_(tokens, {"type", "value", "index", "scale", "clip", "name"}, spec);
            return finish_common_(term, tokens, spec);
        }

        const auto path = tokens.find("path");
        if (path == tokens.end() || path->second.empty())
            throw std::invalid_argument(
                "RlBridge: term requires path=... (term '" + spec
                + "') unless it is joint_*/last_action/constant");
        term.kind = TermKind::kPath;
        term.path = path->second;
        const std::string take = string_token_(tokens, "take", "");
        const std::string transform = string_token_(tokens, "transform", "");

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
            const bool scalar_binding =
                term.binding == Binding::kDouble || term.binding == Binding::kBool
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

        validate_tokens_(
            tokens,
            {"path", "take", "transform", "type", "index", "scale", "clip", "default", "name"},
            spec);
        return finish_common_(term, tokens, spec);
    }

    ObsTerm finish_common_(
        ObsTerm term, const std::map<std::string, std::string>& tokens, const std::string& spec) {
        parse_index_(tokens, spec, term.has_index, term.index);
        term.scale = double_token_(tokens, "scale", 1.0, spec);
        if (term.scale == 0.0)
            throw std::invalid_argument("RlBridge: scale=0 is not allowed (term '" + spec + "')");
        parse_clip_(tokens, spec, term.has_clip, term.clip_min, term.clip_max);

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
                throw std::invalid_argument(
                    "RlBridge: name= must not be empty (term '" + spec + "')");
            term.id = name->second;
        }
        return term;
    }

    ActionTerm parse_action_term_(const std::string& spec) {
        ActionTerm term;
        const auto tokens = parse_tokens_(spec);
        const auto index = tokens.find("index");
        if (index == tokens.end())
            throw std::invalid_argument(
                "RlBridge: action term requires index= (term '" + spec + "')");
        const auto index_value = parse_integer(index->second, "action index");
        if (index_value < 0)
            throw std::invalid_argument(
                "RlBridge: action index must be >= 0 (term '" + spec + "')");
        term.index = static_cast<std::size_t>(index_value);

        const auto output = tokens.find("output");
        if (output == tokens.end() || output->second.empty())
            throw std::invalid_argument(
                "RlBridge: action term requires output=<interface path> (term '" + spec + "')");
        term.output = output->second;
        term.id = string_token_(tokens, "name", term.output);
        if (term.id.empty())
            throw std::invalid_argument(
                "RlBridge: action name= must not be empty (term '" + spec + "')");
        term.scale = double_token_(tokens, "scale", 1.0, spec);
        if (term.scale == 0.0)
            throw std::invalid_argument(
                "RlBridge: action scale=0 is not allowed (term '" + spec + "')");
        parse_clip_(tokens, spec, term.has_clip, term.clip_min, term.clip_max);
        validate_tokens_(tokens, {"index", "output", "name", "scale", "clip"}, spec);
        return term;
    }

    std::string obs_layout_signature_() const {
        std::string signature = "v3-history=" + std::to_string(history_length_);
        for (const auto& term : obs_terms_) {
            signature += "|" + term.id;
            if (term.scale != 1.0)
                signature += "*" + format_number(term.scale);
            signature += "@" + std::to_string(term.dim);
        }
        return signature;
    }

    std::string actions_layout_signature_() const {
        std::string signature = "v2";
        for (const auto& term : action_terms_) {
            signature += "|#" + std::to_string(term.index) + ":" + term.id;
            if (term.scale != 1.0)
                signature += "*" + format_number(term.scale);
        }
        return signature;
    }

    void log_layout_() const {
        RCLCPP_INFO(get_logger(), "observation layout (obs_size=%zu):", obs_size_);
        for (const auto& term : obs_terms_) {
            std::ostringstream line;
            line << "  [" << term.index << ":" << term.index + term.dim << ") " << term.id
                 << "  scale=" << format_number(term.scale);
            if (term.has_default)
                line << " default=" << format_number(term.default_value);
            if (term.has_clip)
                line << " clip=" << format_number(term.clip_min) << ":"
                     << format_number(term.clip_max);
            if (!term.path.empty()) {
                line << " <- " << term.path;
                if (term.slot != kNoSlot)
                    line << " (" << binding_name_(slots_[term.slot]->binding) << ")";
                else
                    line << " (default)";
            }
            RCLCPP_INFO(get_logger(), "%s", line.str().c_str());
        }
        RCLCPP_INFO(get_logger(), "action layout (action_size=%zu):", action_size_);
        for (const auto& term : action_terms_) {
            std::ostringstream line;
            line << "  #" << term.index << " " << term.id << " -> " << term.output;
            if (term.scale != 1.0)
                line << " scale=" << format_number(term.scale);
            RCLCPP_INFO(get_logger(), "%s", line.str().c_str());
        }
    }

    bool read_double_(std::size_t slot, double& value) const {
        const auto& entry = *slots_[slot];
        switch (entry.binding) {
        case Binding::kDouble:
            if (!entry.double_value->ready())
                return false;
            value = **entry.double_value;
            return true;
        case Binding::kBool:
            if (!entry.bool_value->ready())
                return false;
            value = **entry.bool_value ? 1.0 : 0.0;
            return true;
        case Binding::kInt:
            if (!entry.int_value->ready())
                return false;
            value = static_cast<double>(**entry.int_value);
            return true;
        case Binding::kSize:
            if (!entry.size_value->ready())
                return false;
            value = static_cast<double>(**entry.size_value);
            return true;
        default: return false;
        }
    }

    bool read_unsigned_(std::size_t slot, std::uint64_t& value) const {
        double raw = 0.0;
        if (!read_double_(slot, raw))
            return false;
        if (!is_finite(raw) || raw < 0.0)
            return false;
        value = static_cast<std::uint64_t>(raw);
        return true;
    }

    bool read_enable_() const {
        if (enable_slot_ == kNoSlot)
            return enable_default_;
        double raw = 0.0;
        if (!read_double_(enable_slot_, raw))
            return enable_default_;
        return raw != 0.0;
    }

    bool build_observation_(std::vector<double>& obs) const {
        obs.assign(obs_size_, 0.0);
        const auto push = [&obs](
                              std::size_t index, double value, double scale, bool has_clip,
                              double clip_min, double clip_max, bool& ok) {
            double result = value * scale;
            if (has_clip)
                result = std::clamp(result, clip_min, clip_max);
            if (!is_finite(result))
                ok = false;
            obs[index] = result;
        };

        for (const auto& term : obs_terms_) {
            bool ok = true;
            switch (term.kind) {
            case TermKind::kPath: {
                if (term.slot == kNoSlot) {
                    push(
                        term.index, term.default_value, term.scale, term.has_clip, term.clip_min,
                        term.clip_max, ok);
                    break;
                }
                const auto& entry = *slots_[term.slot];
                if (term.take == Take::kScalar) {
                    double value = 0.0;
                    if (!read_double_(term.slot, value))
                        return false;
                    push(
                        term.index, value, term.scale, term.has_clip, term.clip_min, term.clip_max,
                        ok);
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
                        term.index, value, term.scale, term.has_clip, term.clip_min, term.clip_max,
                        ok);
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
                        if (!read_double_(term.joint_slots[j], value))
                            return false;
                        if (term.relative)
                            value -= term.joint_defaults[j];
                    }
                    push(
                        term.index + j, value, term.scale, term.has_clip, term.clip_min,
                        term.clip_max, ok);
                }
                break;
            }
            case TermKind::kJointVel:
            case TermKind::kJointTorque: {
                for (std::size_t j = 0; j < term.joints.size(); ++j) {
                    double value = 0.0;
                    if (!read_double_(term.joint_slots[j], value))
                        return false;
                    push(
                        term.index + j, value, term.scale, term.has_clip, term.clip_min,
                        term.clip_max, ok);
                }
                break;
            }
            case TermKind::kLastAction: {
                for (std::size_t j = 0; j < term.action_indices.size(); ++j)
                    push(
                        term.index + j, last_actions_[term.action_indices[j]], term.scale,
                        term.has_clip, term.clip_min, term.clip_max, ok);
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

    void publish_observation_(
        const std::vector<double>& obs, std::chrono::steady_clock::time_point now) {
        rmcs_rl::msg::Observation message;
        message.header.stamp = get_clock()->now();
        message.header.frame_id = "";
        message.obs_seq = ++pub_seq_;
        message.layout_hash = layout_hash_;
        message.obs = obs;
        obs_publisher_->publish(message);

        prev_pub_seq_ = pub_seq_ - 1;
        prev_pub_time_ = last_pub_time_;
        last_pub_time_ = now;
        pub_started_ = true;
    }

    void on_action_(rmcs_rl::msg::Action::UniquePtr message) {
        const auto received = std::chrono::steady_clock::now();
        if (message->action.size() != action_size_) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 1000, "ignoring action with %zu values (expected %zu)",
                message->action.size(), action_size_);
            return;
        }

        const std::uint64_t sequence = action_sequence_.load(std::memory_order_relaxed);
        action_sequence_.store(sequence + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);

        incoming_.obs_seq = message->obs_seq;
        incoming_.layout_hash = message->layout_hash;
        incoming_.model_id = message->model_id;
        incoming_.received = received;
        std::copy(message->action.begin(), message->action.end(), incoming_.action.begin());

        std::atomic_thread_fence(std::memory_order_release);
        action_sequence_.store(sequence + 2, std::memory_order_relaxed);
    }

    bool try_read_action_(ActionSnapshot& snapshot) {
        for (int attempt = 0; attempt < 4; ++attempt) {
            const std::uint64_t before = action_sequence_.load(std::memory_order_relaxed);
            if (before == 0 || (before & 1U) != 0)
                return false;
            std::atomic_thread_fence(std::memory_order_acquire);
            snapshot = incoming_;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (action_sequence_.load(std::memory_order_relaxed) == before)
                return true;
        }
        return false;
    }

    double obs_age_of_(std::uint64_t obs_seq, std::chrono::steady_clock::time_point now) const {
        if (!pub_started_)
            return std::numeric_limits<double>::quiet_NaN();
        if (obs_seq == pub_seq_)
            return std::chrono::duration<double>(now - last_pub_time_).count();
        if (obs_seq + 1 == pub_seq_)
            return std::chrono::duration<double>(now - prev_pub_time_).count();
        return std::numeric_limits<double>::quiet_NaN();
    }

    void write_actions_(bool valid, const ActionSnapshot& snapshot) {
        for (std::size_t i = 0; i < action_terms_.size(); ++i) {
            const auto& term = action_terms_[i];
            double value = 0.0;
            if (valid) {
                value = snapshot.action[i] * term.scale;
                if (term.has_clip)
                    value = std::clamp(value, term.clip_min, term.clip_max);
                written_[i] = value;
            } else {
                switch (invalid_mode_) {
                case InvalidMode::kNaN: value = std::numeric_limits<double>::quiet_NaN(); break;
                case InvalidMode::kZero: value = 0.0; break;
                case InvalidMode::kHold: value = written_[i]; break;
                }
            }
            **action_outputs_[i] = value;
        }
    }

    void reset_runtime_() {
        std::fill(last_actions_.begin(), last_actions_.end(), 0.0);
        std::fill(written_.begin(), written_.end(), 0.0);
        history_.clear();
        pub_started_ = false;
        prev_pub_seq_ = 0;
    }

    std::string invalid_reason_(
        bool enabled, bool contract_ok, bool has_snapshot, bool fresh, bool seq_ok,
        bool finite) const {
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

    std::vector<std::string> joint_names_;
    std::string joint_base_path_;
    std::unordered_map<std::string, std::size_t> joint_index_;
    std::unordered_map<std::string, std::string> angle_suffix_;
    std::unordered_map<std::string, std::string> velocity_suffix_;
    std::unordered_map<std::string, std::string> torque_suffix_;

    std::vector<ObsTerm> obs_terms_;
    std::vector<ActionTerm> action_terms_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::unique_ptr<OutputInterface<double>>> action_outputs_;
    std::unordered_set<std::string> own_output_paths_;

    std::size_t obs_size_ = 0;
    std::size_t obs_frame_size_ = 0;
    std::size_t history_length_ = 1;
    std::size_t action_size_ = 0;

    std::string rl_base_;
    double policy_rate_ = 50.0;
    double max_action_age_ = 0.04;
    std::uint64_t expected_model_id_ = 0;
    InvalidMode invalid_mode_ = InvalidMode::kNaN;
    std::string enable_path_;
    bool enable_default_ = false;
    std::string reset_path_;
    std::size_t enable_slot_ = kNoSlot;
    std::size_t reset_slot_ = kNoSlot;

    std::string obs_signature_;
    std::string actions_signature_;
    std::uint64_t layout_hash_ = 0;

    std::vector<double> last_actions_;
    std::vector<double> written_;
    std::deque<std::vector<double>> history_;

    std::chrono::nanoseconds pub_period_{std::chrono::milliseconds(20)};
    std::chrono::steady_clock::time_point last_pub_time_{};
    std::chrono::steady_clock::time_point prev_pub_time_{};
    bool pub_started_ = false;
    std::uint64_t pub_seq_ = 0;
    std::uint64_t prev_pub_seq_ = 0;
    std::uint64_t last_reset_count_ = 0;
    std::uint64_t obs_invalid_count_ = 0;
    std::uint64_t pub_ok_count_ = 0;
    bool contract_ok_ = true;
    bool last_valid_ = false;

    ActionSnapshot incoming_;
    ActionSnapshot read_snapshot_;
    alignas(64) std::atomic<std::uint64_t> action_sequence_{0};

    OutputInterface<double> valid_output_{};
    OutputInterface<double> healthy_output_{};
    OutputInterface<double> action_age_output_{};
    OutputInterface<std::size_t> obs_seq_output_{};

    rclcpp::Publisher<rmcs_rl::msg::Observation>::SharedPtr obs_publisher_;
    rclcpp::Subscription<rmcs_rl::msg::Action>::SharedPtr action_subscription_;
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_rl::RlBridge, rmcs_executor::Component)
