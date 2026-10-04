#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
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

#include <rmcs_rl/rl_bridge/action_channel.hpp>
#include <rmcs_rl/rl_bridge/config.hpp>
#include <rmcs_rl/rl_bridge/interface_binding.hpp>
#include <rmcs_rl/rl_bridge/joint_config.hpp>
#include <rmcs_rl/rl_bridge/observation.hpp>
#include <rmcs_rl/rl_bridge/observation_history.hpp>
#include <rmcs_rl/rl_bridge/observation_timeline.hpp>
#include <rmcs_rl/rl_bridge/types.hpp>
#include <rmcs_rl/rl_bridge/utility.hpp>
#include <rmcs_rl/rl_layout.hpp>

namespace rmcs_rl {

class RlBridge final
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    explicit RlBridge()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true))
        , config_(load_bridge_config(*this))
        , history_(config_.obs_frame_size, config_.history_length) {

        for (const auto& term : config_.action_terms) {
            action_outputs_.push_back(std::make_unique<OutputInterface<double>>());
            register_output(term.output, *action_outputs_.back(), 0.0);
            own_output_paths_.insert(term.output);
        }
        register_output(config_.rl_base + "/valid", valid_output_, 0.0);
        register_output(config_.rl_base + "/healthy", healthy_output_, 0.0);
        register_output(
            config_.rl_base + "/action_age", action_age_output_,
            std::numeric_limits<double>::quiet_NaN());
        register_output(config_.rl_base + "/obs_seq", obs_seq_output_, std::size_t{0});
        own_output_paths_.insert(config_.rl_base + "/valid");
        own_output_paths_.insert(config_.rl_base + "/healthy");
        own_output_paths_.insert(config_.rl_base + "/action_age");
        own_output_paths_.insert(config_.rl_base + "/obs_seq");
        setup_topics_();
        last_actions_.assign(config_.action_size, 0.0);
        written_.assign(config_.action_size, 0.0);
        action_channel_.resize(config_.action_size);
        read_snapshot_.action.assign(config_.action_size, 0.0);
    }

    void before_pairing(const OutputInfoMap& output_map) override {
        bind_observation_slots_(output_map);
        bind_control_slots_(output_map);

        for (const auto& slot : slots_)
            if (own_output_paths_.count(slot.path) != 0)
                throw std::runtime_error(
                    "RlBridge: interface \"" + slot.path
                    + "\" is produced by RlBridge itself (self reference)");

        obs_signature_ = obs_layout_signature(config_.obs_terms, config_.history_length);
        actions_signature_ = actions_layout_signature(config_.action_terms);
        layout_hash_ = rmcs_rl::layout_hash(
            obs_signature_, actions_signature_, config_.obs_size, config_.action_size);

        log_layout_();
        RCLCPP_INFO(
            get_logger(), "contract: obs_size=%zu action_size=%zu policy_rate=%.3f Hz",
            config_.obs_size, config_.action_size, config_.policy_rate);
        RCLCPP_INFO(
            get_logger(), "layout_hash=%s (obs/action signatures below)",
            hex16(layout_hash_).c_str());
        RCLCPP_INFO(get_logger(), "  obs signature    : %s", obs_signature_.c_str());
        RCLCPP_INFO(get_logger(), "  action signature : %s", actions_signature_.c_str());
        RCLCPP_INFO(
            get_logger(), "topics: %s/obs -> %s/action (best_effort, keep_last=1)",
            config_.rl_base.c_str(), config_.rl_base.c_str());
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
            age = timeline_.age(snapshot.obs_seq, now);
            seq_ok = std::isfinite(age);
            finite = std::all_of(snapshot.action.begin(), snapshot.action.end(), [](double value) {
                return std::isfinite(value);
            });
            check_contract_(snapshot);
        }

        const bool enabled = read_enable_();
        const bool fresh = has_snapshot && is_finite(age) && age <= config_.max_action_age;
        const bool valid = enabled && contract_ok_ && has_snapshot && fresh && seq_ok && finite;

        write_actions(
            valid, snapshot, config_.action_terms, config_.invalid_mode, written_, action_outputs_);

        if (valid)
            last_actions_ = snapshot.action;

        *valid_output_ = valid ? 1.0 : 0.0;
        *healthy_output_ = (contract_ok_ && fresh) ? 1.0 : 0.0;
        *action_age_output_ = age;
        *obs_seq_output_ = timeline_.sequence();

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
    void setup_topics_() {
        obs_publisher_ = create_publisher<rmcs_rl::msg::Observation>(
            config_.rl_base + "/obs", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort());
        action_subscription_ = create_subscription<rmcs_rl::msg::Action>(
            config_.rl_base + "/action", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort(),
            [this](rmcs_rl::msg::Action::UniquePtr message) { on_action_(std::move(message)); });
    }

    void bind_observation_slots_(const OutputInfoMap& output_map) {
        for (auto& term : config_.obs_terms) {
            switch (term.kind) {
            case TermKind::kConstant:
            case TermKind::kLastAction: break;
            case TermKind::kJointPos:
            case TermKind::kJointVel:
            case TermKind::kJointTorque: {
                for (std::size_t j = 0; j < term.joints.size(); ++j) {
                    const auto& joint = config_.joint_config.names[term.joints[j]];
                    const char field = term.kind == TermKind::kJointPos ? 'a'
                                     : term.kind == TermKind::kJointVel ? 'v'
                                                                        : 't';
                    const auto path = joint_path(config_.joint_config, joint, field);
                    term.joint_slots.push_back(acquire_slot(
                        *this, path, {Binding::kDouble}, true, output_map, slots_, "joint term"));
                }
                break;
            }
            case TermKind::kPath: {
                std::vector<Binding> candidates;
                if (term.has_binding) {
                    candidates = {term.binding};
                } else {
                    switch (term.take) {
                    case Take::kScalar:
                        candidates = {
                            Binding::kDouble, Binding::kBool, Binding::kInt, Binding::kSize};
                        break;
                    case Take::kComponent:
                    case Take::kVector:
                        candidates = {Binding::kVector3, Binding::kDirectionVector};
                        break;
                    case Take::kGravity: candidates = {Binding::kQuaternion}; break;
                    }
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
        if (!config_.enable_path.empty())
            enable_slot_ = acquire_slot(
                *this, config_.enable_path, {Binding::kBool, Binding::kDouble}, false, output_map,
                slots_, "enable interface");

        if (!config_.reset_path.empty())
            reset_slot_ = acquire_slot(
                *this, config_.reset_path, {Binding::kSize, Binding::kInt, Binding::kDouble}, false,
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
        if (!timeline_.due(now, config_.pub_period))
            return;

        std::vector<double> obs;
        if (build_observation(
                config_.obs_terms, slots_, last_actions_, config_.obs_frame_size, obs)) {
            publish_observation_(history_.append(obs), now);
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
            return config_.enable_default;
        double raw = 0.0;
        if (!read_double(slots_[enable_slot_], raw))
            return config_.enable_default;
        return std::isfinite(raw) && raw != 0.0;
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
        }
    }

    void log_layout_() const {
        RCLCPP_INFO(get_logger(), "observation layout (obs_size=%zu):", config_.obs_size);
        for (const auto& term : config_.obs_terms) {
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
        RCLCPP_INFO(get_logger(), "action layout (action_size=%zu):", config_.action_size);
        for (const auto& term : config_.action_terms) {
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
        message.obs_seq = timeline_.publish(now);
        message.layout_hash = layout_hash_;
        message.obs = obs;
        obs_publisher_->publish(message);
    }

    void on_action_(rmcs_rl::msg::Action::UniquePtr message) {
        if (message->action.size() != config_.action_size) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 1000, "ignoring action with %zu values (expected %zu)",
                message->action.size(), config_.action_size);
            return;
        }
        action_channel_.store(*message);
    }

    void reset_runtime_() {
        reset_action_state(last_actions_, written_);
        history_.reset();
        timeline_.reset();
    }

    std::vector<Slot> slots_;
    std::vector<std::unique_ptr<OutputInterface<double>>> action_outputs_;

    OutputInterface<double> valid_output_{};
    OutputInterface<double> healthy_output_{};
    OutputInterface<double> action_age_output_{};
    OutputInterface<std::size_t> obs_seq_output_{};

    BridgeConfig config_;
    std::unordered_set<std::string> own_output_paths_;

    std::size_t enable_slot_ = kNoSlot;
    std::size_t reset_slot_ = kNoSlot;

    std::string obs_signature_;
    std::string actions_signature_;
    std::uint64_t layout_hash_ = 0;

    std::vector<double> last_actions_;
    std::vector<double> written_;
    ObservationHistory history_;

    ObservationTimeline timeline_;
    std::uint64_t last_reset_count_ = 0;
    std::uint64_t obs_invalid_count_ = 0;
    bool contract_ok_ = true;
    bool last_valid_ = false;

    ActionChannel action_channel_;
    ActionSnapshot read_snapshot_;

    rclcpp::Publisher<rmcs_rl::msg::Observation>::SharedPtr obs_publisher_;
    rclcpp::Subscription<rmcs_rl::msg::Action>::SharedPtr action_subscription_;
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_rl::RlBridge, rmcs_executor::Component)
