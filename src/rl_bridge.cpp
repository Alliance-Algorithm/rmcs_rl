#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/subscription.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_rl/msg/action.hpp>
#include <rmcs_rl/msg/observation.hpp>

#include "rl_bridge/action_channel.hpp"
#include "rl_bridge/interface_binding.hpp"
#include "rl_bridge/joint_config.hpp"
#include "rl_bridge/observation.hpp"
#include "rl_bridge/parameters.hpp"
#include "rl_bridge/term_parser.hpp"
#include "rl_bridge/types.hpp"
#include "rl_bridge/utility.hpp"
#include "rl_layout.hpp"

namespace rmcs_rl {

class RlBridge final
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    explicit RlBridge()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {

        rl_base_ = string_or(*this, "rl_base", "/rl");

        joint_config_ = load_joint_config(*this);

        load_action_terms_();
        load_observation_terms_();
        load_runtime_parameters_();

        register_status_outputs_();
        setup_topics_();

        action_channel_.resize(action_size_);
        read_snapshot_.action.assign(action_size_, 0.0);
    }

    void before_pairing(const OutputInfoMap& output_map) override {
        bind_observation_slots_(output_map);
        bind_control_slots_(output_map);

        for (const auto& slot : slots_)
            if (own_output_paths_.count(slot.path) != 0)
                throw std::runtime_error(
                    "RlBridge: interface \"" + slot.path
                    + "\" is produced by RlBridge itself (self reference)");

        obs_signature_ = obs_layout_signature(obs_terms_, history_length_);
        actions_signature_ = actions_layout_signature(action_terms_);
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

        maybe_handle_reset_();
        maybe_publish_observation_(now);

        ActionSnapshot& snapshot = read_snapshot_;
        const bool has_snapshot = action_channel_.try_read(snapshot);

        double age = std::numeric_limits<double>::quiet_NaN();
        bool seq_ok = false;
        bool finite = true;
        if (has_snapshot) {
            age = obs_age_of(snapshot.obs_seq, now);
            seq_ok = (snapshot.obs_seq == pub_seq_) || (snapshot.obs_seq + 1 == pub_seq_);
            finite = std::all_of(snapshot.action.begin(), snapshot.action.end(), [](double value) {
                return std::isfinite(value);
            });
            check_contract_(snapshot);
        }

        const bool enabled = read_enable_();
        const bool fresh = has_snapshot && is_finite(age) && age <= max_action_age_;
        const bool valid = enabled && contract_ok_ && has_snapshot && fresh && seq_ok && finite;

        write_actions(valid, snapshot, action_terms_, invalid_mode_, written_, action_outputs_);

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
                    invalid_reason(enabled, contract_ok_, has_snapshot, fresh, seq_ok, finite)
                        .c_str());
            }
            last_valid_ = valid;
        }
    }

private:
    void load_action_terms_() {
        const auto action_specs = string_array_or(*this, "action_terms");
        if (action_specs.empty())
            throw std::invalid_argument("RlBridge: required parameter 'action_terms' is missing");
        std::unordered_set<std::string> output_paths;
        std::set<std::size_t> action_indices;
        for (const auto& spec : action_specs) {
            auto term = parse_action_term(spec);
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
        if (const auto declared = integer_parameter(*this, "rl_action_size");
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
    }

    void load_observation_terms_() {
        const auto observation_specs = string_array_or(*this, "observation_terms");
        if (observation_specs.empty())
            throw std::invalid_argument(
                "RlBridge: required parameter 'observation_terms' is "
                "missing");
        const TermParseContext context{*this, joint_config_, action_size_};
        for (const auto& spec : observation_specs)
            obs_terms_.push_back(parse_obs_term(spec, context));

        assign_observation_indices(obs_terms_);
        obs_frame_size_ = [&] {
            std::size_t cursor = 0;
            for (const auto& term : obs_terms_)
                cursor += term.dim;
            return cursor;
        }();
        const auto history_length = integer_parameter(*this, "history_length").value_or(1);
        if (history_length < 1 || history_length > 64)
            throw std::invalid_argument("RlBridge: history_length must be in [1, 64]");
        history_length_ = static_cast<std::size_t>(history_length);
        obs_size_ = obs_frame_size_ * history_length_;
        if (const auto declared = integer_parameter(*this, "rl_obs_size");
            declared.has_value() && static_cast<std::size_t>(*declared) != obs_size_)
            throw std::invalid_argument(
                "RlBridge: rl_obs_size=" + std::to_string(*declared)
                + " but observation frame/history contract is " + std::to_string(obs_frame_size_)
                + "x" + std::to_string(history_length_) + "=" + std::to_string(obs_size_));
    }

    void load_runtime_parameters_() {
        policy_rate_ = number_or(*this, "policy_rate", 50.0);
        if (!(policy_rate_ > 0.0) || !is_finite(policy_rate_))
            throw std::invalid_argument("RlBridge: policy_rate must be finite and > 0");
        pub_period_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / policy_rate_));
        max_action_age_ = number_or(*this, "max_action_age", 2.0 / policy_rate_);
        if (!(max_action_age_ > 0.0) || !is_finite(max_action_age_))
            throw std::invalid_argument("RlBridge: max_action_age must be finite and > 0");
        expected_model_id_ = parse_u64(string_or(*this, "expected_model_id", "0"), "expected_model_id");

        const std::string invalid = string_or(*this, "invalid_value", "nan");
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
                + string_or(*this, "invalid_value", "<non-string>") + "'");

        enable_path_ = string_or(*this, "enable_interface", rl_base_ + "/enable");
        enable_default_ = bool_or(*this, "enable_default", false);
        reset_path_ = string_or(*this, "reset_interface", "");
    }

    void register_status_outputs_() {
        register_output(rl_base_ + "/valid", valid_output_, 0.0);
        register_output(rl_base_ + "/healthy", healthy_output_, 0.0);
        register_output(
            rl_base_ + "/action_age", action_age_output_, std::numeric_limits<double>::quiet_NaN());
        register_output(rl_base_ + "/obs_seq", obs_seq_output_, std::size_t{0});
        own_output_paths_.insert(rl_base_ + "/valid");
        own_output_paths_.insert(rl_base_ + "/healthy");
        own_output_paths_.insert(rl_base_ + "/action_age");
        own_output_paths_.insert(rl_base_ + "/obs_seq");
    }

    void setup_topics_() {
        obs_publisher_ = create_publisher<rmcs_rl::msg::Observation>(
            rl_base_ + "/obs", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort());
        action_subscription_ = create_subscription<rmcs_rl::msg::Action>(
            rl_base_ + "/action", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort(),
            [this](rmcs_rl::msg::Action::UniquePtr message) { on_action_(std::move(message)); });
    }

    void bind_observation_slots_(const OutputInfoMap& output_map) {
        for (auto& term : obs_terms_) {
            switch (term.kind) {
            case TermKind::kConstant:
            case TermKind::kLastAction: break;
            case TermKind::kJointPos:
            case TermKind::kJointVel:
            case TermKind::kJointTorque: {
                for (std::size_t j = 0; j < term.joints.size(); ++j) {
                    const auto& joint = joint_config_.names[term.joints[j]];
                    const char field = term.kind == TermKind::kJointPos ? 'a'
                                     : term.kind == TermKind::kJointVel ? 'v'
                                                                        : 't';
                    const auto path = joint_path(joint_config_, joint, field);
                    term.joint_slots.push_back(acquire_slot(
                        *this, path, {Binding::kDouble}, true, output_map, slots_, "joint term"));
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
                term.slot = acquire_slot(
                    *this, term.path, candidates, !term.has_default, output_map, slots_,
                    "observation term");
                break;
            }
            }
        }
    }

    void bind_control_slots_(const OutputInfoMap& output_map) {
        if (!enable_path_.empty())
            enable_slot_ = acquire_slot(
                *this, enable_path_, {Binding::kBool, Binding::kDouble}, false, output_map, slots_,
                "enable interface");

        if (!reset_path_.empty())
            reset_slot_ = acquire_slot(
                *this, reset_path_, {Binding::kSize, Binding::kInt, Binding::kDouble}, false,
                output_map, slots_, "reset interface");
    }

    void maybe_handle_reset_() {
        if (reset_slot_ == kNoSlot)
            return;
        std::uint64_t reset_count = 0;
        if (read_unsigned(slots_[reset_slot_], reset_count) && reset_count != last_reset_count_) {
            last_reset_count_ = reset_count;
            reset_runtime_();
        }
    }

    void maybe_publish_observation_(std::chrono::steady_clock::time_point now) {
        if (pub_started_ && now - last_pub_time_ < pub_period_)
            return;

        std::vector<double> obs;
        if (build_observation(obs_terms_, slots_, last_actions_, obs_size_, obs)) {
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

    bool read_enable_() const {
        if (enable_slot_ == kNoSlot)
            return enable_default_;
        double raw = 0.0;
        if (!read_double(slots_[enable_slot_], raw))
            return enable_default_;
        return raw != 0.0;
    }

    void check_contract_(const ActionSnapshot& snapshot) {
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
                    line << " (" << binding_name(slots_[term.slot].binding) << ")";
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
        if (message->action.size() != action_size_) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 1000, "ignoring action with %zu values (expected %zu)",
                message->action.size(), action_size_);
            return;
        }
        action_channel_.store(*message);
    }

    double obs_age_of(std::uint64_t obs_seq, std::chrono::steady_clock::time_point now) const {
        if (!pub_started_)
            return std::numeric_limits<double>::quiet_NaN();
        if (obs_seq == pub_seq_)
            return std::chrono::duration<double>(now - last_pub_time_).count();
        if (obs_seq + 1 == pub_seq_)
            return std::chrono::duration<double>(now - prev_pub_time_).count();
        return std::numeric_limits<double>::quiet_NaN();
    }

    void reset_runtime_() {
        reset_action_state(last_actions_, written_);
        history_.clear();
        pub_started_ = false;
        prev_pub_seq_ = 0;
    }

    JointConfig joint_config_;

    std::vector<ObsTerm> obs_terms_;
    std::vector<ActionTerm> action_terms_;
    std::vector<Slot> slots_;
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

    ActionChannel action_channel_;
    ActionSnapshot read_snapshot_;

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
