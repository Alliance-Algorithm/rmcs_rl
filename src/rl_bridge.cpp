#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>

#include "policy/policy_worker.hpp"
#include "rl_bridge/runtime_state.hpp"
#include <rmcs_rl/rl_bridge/config.hpp>
#include <rmcs_rl/rl_bridge/interface_binding.hpp>
#include <rmcs_rl/rl_bridge/joint_config.hpp>
#include <rmcs_rl/rl_bridge/observation.hpp>
#include <rmcs_rl/rl_bridge/observation_frame_gate.hpp>
#include <rmcs_rl/rl_bridge/types.hpp>
#include <rmcs_rl/rl_bridge/utility.hpp>

namespace rmcs_rl {

class PolicyActionPublisher final : public rmcs_executor::Component {
public:
    using Publish = std::function<void(std::optional<std::uint64_t>)>;

    PolicyActionPublisher(
        const BridgeConfig& config, const PolicyModel::Info& model, Publish publish)
        : publish_(std::move(publish))
        , reset_path_(config.reset_path)
        , terms_(config.action_terms)
        , invalid_mode_(config.invalid_mode)
        , written_(config.action_size) {
        for (const auto& term : config.action_terms) {
            action_outputs_.push_back(std::make_unique<OutputInterface<double>>());
            register_output(term.output, *action_outputs_.back(), 0.0);
            raw_outputs_.push_back(std::make_unique<OutputInterface<double>>());
            register_output(term.output + "/raw", *raw_outputs_.back(), 0.0);
            for (const auto& [suffix, value] :
                 {std::pair{"/scale", term.scale},
                  std::pair{"/offset", term.offset},
                  std::pair{
                      "/clip_min",
                      term.has_clip ? term.clip_min : -std::numeric_limits<double>::infinity()},
                  std::pair{
                      "/clip_max",
                      term.has_clip ? term.clip_max : std::numeric_limits<double>::infinity()}}) {
                contract_outputs_.push_back(std::make_unique<OutputInterface<double>>());
                register_output(term.output + suffix, *contract_outputs_.back(), value);
            }
        }
        register_output(config.rl_base + "/valid", valid_output_, 0.0);
        register_output(config.rl_base + "/healthy", healthy_output_, 0.0);
        register_output(
            config.rl_base + "/action_age", action_age_output_,
            std::numeric_limits<double>::quiet_NaN());
        register_output(config.rl_base + "/obs_seq", obs_seq_output_, std::size_t{0});
        register_output(config.rl_base + "/action_seq", action_seq_output_, std::size_t{0});
        register_output(config.rl_base + "/action_context", action_context_output_, std::size_t{0});
        register_output(config.rl_base + "/action_steady_ns", action_ns_output_, std::uint64_t{0});
        register_output(
            config.rl_base + "/observation_size", observation_size_output_, config.obs_size);
        register_output(config.rl_base + "/action_size", action_size_output_, config.action_size);
        register_output(
            config.rl_base + "/history_length", history_length_output_, config.history_length);
        register_output(
            config.rl_base + "/feature_size", feature_size_output_, config.obs_frame_size);
        register_output(config.rl_base + "/policy_rate", policy_rate_output_, config.policy_rate);
        register_output(
            config.rl_base + "/model_spec_sha256", model_spec_output_, model.model_spec_sha256);
        register_output(
            config.rl_base + "/observation_contract_sha256", observation_contract_output_,
            model.observation_contract_sha256);
        register_output(
            config.rl_base + "/observation_frame_sequence_interface", frame_sequence_path_output_,
            config.observation_frame_sequence_path);
        register_output(
            config.rl_base + "/observation_frame_interface", frame_path_output_,
             config.observation_frame_sequence_path.empty() || config.obs_terms.size() != 1
                 ? std::string{} : config.obs_terms.front().path);
        register_output(config.rl_base + "/performance/inference_us", inference_time_us_, 0.0);
    }

    void before_pairing(const OutputInfoMap& output_map) override {
        if (!reset_path_.empty())
            reset_slot_ = acquire_slot(
                *this, reset_path_, {Binding::kSize, Binding::kInt, Binding::kDouble}, true,
                output_map, slots_, "policy reset interface");
    }

    void update() override {
        std::uint64_t count = 0;
        const bool have_reset = reset_slot_ != kNoSlot && read_unsigned(slots_[reset_slot_], count);
        publish_(have_reset ? std::optional{count} : std::nullopt);
    }

    void reset() { std::fill(written_.begin(), written_.end(), 0.0); }

    void publish(
        const PolicyWorker::Result& action, bool valid, bool healthy, double age,
        std::uint64_t observation_sequence) {
        for (std::size_t i = 0; i < terms_.size(); ++i) {
            double value;
            if (valid) {
                value = map_action(action.action[i], terms_[i]);
                written_[i] = value;
            } else if (invalid_mode_ == InvalidMode::kNaN) {
                value = std::numeric_limits<double>::quiet_NaN();
            } else if (invalid_mode_ == InvalidMode::kZero) {
                value = 0.0;
            } else {
                value = written_[i];
            }
            **action_outputs_[i] = value;
            **raw_outputs_[i] = valid ? action.action[i] : std::numeric_limits<double>::quiet_NaN();
        }
        *valid_output_ = valid ? 1.0 : 0.0;
        *healthy_output_ = healthy ? 1.0 : 0.0;
        *action_age_output_ = age;
        *obs_seq_output_ = observation_sequence;
        *action_seq_output_ = action.sequence;
        *action_context_output_ = action.context;
        *action_ns_output_ =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           action.submitted_at.time_since_epoch())
                                           .count());
        *inference_time_us_ = action.inference_microseconds;
    }

private:
    Publish publish_;
    std::string reset_path_;
    std::vector<Slot> slots_;
    std::size_t reset_slot_ = kNoSlot;
    std::vector<ActionTerm> terms_;
    InvalidMode invalid_mode_;
    std::vector<double> written_;
    std::vector<std::unique_ptr<OutputInterface<double>>> action_outputs_, contract_outputs_;
    std::vector<std::unique_ptr<OutputInterface<double>>> raw_outputs_;
    OutputInterface<double> valid_output_, healthy_output_, action_age_output_, inference_time_us_;
    OutputInterface<double> policy_rate_output_;
    OutputInterface<std::string> model_spec_output_, observation_contract_output_;
    OutputInterface<std::string> frame_sequence_path_output_, frame_path_output_;
    OutputInterface<std::size_t> obs_seq_output_, action_seq_output_, action_context_output_;
    OutputInterface<std::uint64_t> action_ns_output_;
    OutputInterface<std::size_t> observation_size_output_, action_size_output_;
    OutputInterface<std::size_t> history_length_output_, feature_size_output_;
};

class RlBridge final
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    explicit RlBridge()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true))
        , config_(load_bridge_config(*this))
        , history_(config_.obs_frame_size, config_.history_length, config_.history_period)
        , action_feedback_(config_.action_size, feedback_capacity_(config_))
        , worker_(config_.policy)
        , frame_buffer_(config_.obs_frame_size)
        , observation_buffer_(config_.obs_size)
        , result_buffer_{} {
        action_.action.resize(config_.action_size);
        result_buffer_.action.reserve(config_.action_size);

        action_source_ = create_partner_component<PolicyActionPublisher>(
            get_component_name() + "_actions", config_, worker_.info(),
            [this](std::optional<std::uint64_t> reset) { publish_actions_(reset); });
        for (const auto& term : config_.action_terms)
            own_output_paths_.insert(term.output);
        for (const auto& term : config_.obs_terms) {
            for (std::size_t i = 0; i < term.dim; ++i) {
                auto name = term.id;
                if (term.dim > 1) {
                    if (term.take == Take::kVector || term.take == Take::kGravity)
                        name += "/element_" + std::to_string(i);
                    else if (!term.joint_names.empty())
                        name += "/" + term.joint_names[i];
                    else if (term.take == Take::kArray)
                        name += "/element_" + std::to_string(i);
                    else
                        name += "/" + std::to_string(i);
                }
                observation_outputs_.push_back(std::make_unique<OutputInterface<double>>());
                const auto path = config_.rl_base + "/observation/" + name;
                register_output(path, *observation_outputs_.back(), 0.0);
                own_output_paths_.insert(path);
            }
        }
        own_output_paths_.insert(config_.rl_base + "/valid");
        own_output_paths_.insert(config_.rl_base + "/healthy");
        own_output_paths_.insert(config_.rl_base + "/action_age");
        own_output_paths_.insert(config_.rl_base + "/obs_seq");
        last_actions_.assign(config_.action_size, 0.0);
        register_output(config_.rl_base + "/input/frame", observation_frame_output_, std::vector<double>(config_.obs_frame_size));
        own_output_paths_.insert(config_.rl_base + "/input/frame");
    }

    void before_pairing(const OutputInfoMap& output_map) override {
        bind_observation_slots_(output_map);
        bind_control_slots_(output_map);

        for (const auto& slot : slots_)
            if (own_output_paths_.count(slot.path) != 0)
                throw std::runtime_error(
                    "RlBridge: interface \"" + slot.path
                    + "\" is produced by RlBridge itself (self reference)");

        log_layout_();
        RCLCPP_INFO(
            get_logger(), "contract: obs_size=%zu action_size=%zu policy_rate=%.3f Hz",
            config_.obs_size, config_.action_size, config_.policy_rate);
        RCLCPP_INFO(
            get_logger(), "in-process policy worker: model=%s, obs=%zu, action=%zu",
            worker_.info().path.c_str(), worker_.info().obs_size, worker_.info().action_size);
    }

    void update() override {
        const auto now = std::chrono::steady_clock::now();
        maybe_handle_reset_();
        const bool enabled = read_enable_();
        double flat_hold_value = 0.0;
        const bool flat_hold = flat_hold_slot_ != kNoSlot
                            && read_double(slots_[flat_hold_slot_], flat_hold_value)
                            && std::isfinite(flat_hold_value) && flat_hold_value != 0.0;
        if (enabled != last_enabled_ || flat_hold != last_flat_hold_) {
            reset_runtime_();
            last_enabled_ = enabled;
            last_flat_hold_ = flat_hold;
        }
        if (!enabled) {
            std::fill(last_actions_.begin(), last_actions_.end(), 0.0);
            action_source_->reset();
            history_.reset();
        }
        if (enabled) {
            if (frame_sequence_slot_ != kNoSlot) {
                const auto& sequence = slots_[frame_sequence_slot_].size_value;
                const auto decision = frame_gate_.observe(sequence->ready() ? **sequence : 0);
                if (decision == ObservationFrameGate::Decision::kSubmit)
                    maybe_submit_observation_(now);
                else if (decision == ObservationFrameGate::Decision::kReset)
                    reset_runtime_(false);
            } else if (timeline_.due(now, config_.pub_period)) {
                maybe_submit_observation_(now);
            }
        }
    }

private:
    void publish_actions_(std::optional<std::uint64_t> reset) {
        const auto now = std::chrono::steady_clock::now();
        if (reset && *reset != last_reset_count_) {
            last_reset_count_ = *reset;
            reset_runtime_();
        }
        const bool enabled = last_enabled_;
        const bool flat_hold = last_flat_hold_;
        receive_policy_result_();
        const bool has_snapshot = action_ready_;

        const double age = has_snapshot
                             ? std::chrono::duration<double>(now - action_.submitted_at).count()
                             : std::numeric_limits<double>::quiet_NaN();
        const bool seq_ok = has_snapshot && action_.generation == generation_
                         && action_.sequence <= observation_sequence_;
        const bool finite =
            has_snapshot
            && std::all_of(action_.action.begin(), action_.action.end(), [](double value) {
                   return std::isfinite(value);
               });

        const bool fresh = has_snapshot && is_finite(age) && age <= config_.inference_timeout
                        && age <= config_.max_action_age;
        const bool valid = enabled && !worker_failed_ && has_snapshot && fresh && seq_ok && finite;

        if (valid && flat_hold)
            std::fill(action_.action.begin(), action_.action.end(), 0.0);
        if (valid && action_.sequence != last_feedback_obs_seq_) {
            for (std::size_t i = 0; i < config_.action_size; ++i) {
                const auto& term = config_.action_terms[i];
                last_actions_[i] =
                    term.scale == 0.0 ? 0.0
                                      : (map_action(action_.action[i], term) - term.offset) / term.scale;
            }
            action_feedback_.store(action_.submitted_at, last_actions_);
            last_feedback_obs_seq_ = action_.sequence;
        }
        action_source_->publish(
            action_, valid, worker_.healthy() && !worker_failed_, age, observation_sequence_);

        if (last_valid_ && !valid)
            reset_runtime_(false);
        if (valid != last_valid_) {
            if (valid) {
                RCLCPP_INFO(
                    get_logger(), "valid=1 (obs_seq=%llu action_age=%.4f s)",
                    static_cast<unsigned long long>(action_.sequence), age);
            } else {
                RCLCPP_WARN(
                    get_logger(), "valid=0: %s",
                    invalid_reason(enabled, !worker_failed_, has_snapshot, fresh, seq_ok, finite)
                        .c_str());
            }
            last_valid_ = valid;
        }
    }

    void receive_policy_result_() {
        if (worker_.try_take(result_buffer_)) {
            if (result_buffer_.generation == generation_
                && result_buffer_.sequence > action_.sequence
                && result_buffer_.sequence <= observation_sequence_ && result_buffer_.success) {
                std::swap(action_, result_buffer_);
                action_ready_ = true;
            } else if (result_buffer_.generation == generation_ && !result_buffer_.success) {
                worker_failed_ = true;
                action_ready_ = false;
                RCLCPP_ERROR_THROTTLE(
                    get_logger(), *get_clock(), 1000, "RL inference failed: %s",
                    result_buffer_.error.c_str());
            }
        }
    }

    static std::size_t feedback_capacity_(const BridgeConfig& config) {
        if (config.history_period.count() == 0)
            return 2;
        const auto delay_samples = static_cast<std::size_t>(std::ceil(
            static_cast<double>(config.pub_period.count())
            / static_cast<double>(config.history_period.count())));
        return config.history_length + delay_samples + 4;
    }

    void bind_observation_slots_(const OutputInfoMap& output_map) {
        bind_observation_terms(
            config_.obs_terms, config_.joint_config,
            [this, &output_map](
                const std::string& path, const std::vector<Binding>& candidates, bool required,
                const char* context) {
                return acquire_slot(*this, path, candidates, required, output_map, slots_, context);
            });
    }

    void bind_control_slots_(const OutputInfoMap& output_map) {
        if (!config_.observation_frame_sequence_path.empty())
            frame_sequence_slot_ = acquire_slot(
                *this, config_.observation_frame_sequence_path, {Binding::kSize}, true, output_map,
                slots_, "observation frame sequence interface");
        if (!config_.enable_path.empty())
            enable_slot_ = acquire_slot(
                *this, config_.enable_path, {Binding::kBool, Binding::kDouble}, false, output_map,
                slots_, "enable interface");

        if (!config_.reset_path.empty())
            reset_slot_ = acquire_slot(
                *this, config_.reset_path, {Binding::kSize, Binding::kInt, Binding::kDouble}, false,
                output_map, slots_, "reset interface");
        if (!config_.context_path.empty())
            context_slot_ = acquire_slot(
                *this, config_.context_path, {Binding::kSize, Binding::kInt}, true, output_map,
                slots_, "policy context interface");
        if (!config_.flat_hold_path.empty())
            flat_hold_slot_ = acquire_slot(
                *this, config_.flat_hold_path, {Binding::kBool, Binding::kDouble}, true, output_map,
                slots_, "flat hold interface");
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

    void maybe_submit_observation_(std::chrono::steady_clock::time_point now) {
        if (config_.history_period.count() > 0)
            last_actions_ = action_feedback_.sample(now, config_.history_period);
        if (build_observation(
                config_.obs_terms, slots_, last_actions_, config_.obs_frame_size, frame_buffer_)) {
            const auto& stacked = history_.append(frame_buffer_, now);
            observation_sequence_ = ++next_observation_sequence_;
            timeline_.publish(now, config_.pub_period);
            std::copy(stacked.begin(), stacked.end(), observation_buffer_.begin());
            for (std::size_t i = 0; i < observation_outputs_.size(); ++i)
                **observation_outputs_[i] = frame_buffer_[i];
            std::copy(frame_buffer_.begin(), frame_buffer_.end(), observation_frame_output_->begin());
            std::uint64_t context = 0;
            if (context_slot_ != kNoSlot && !read_unsigned(slots_[context_slot_], context))
                throw std::runtime_error("RlBridge: invalid inference context");
            worker_.submit(
                std::span<const double>{observation_buffer_}, observation_sequence_, generation_,
                now, context);
        } else {
            reset_runtime_(false);
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

    void reset_runtime_(bool reset_frame_gate = true) {
        if (reset_frame_gate)
            frame_gate_.reset();
        ++generation_;
        action_ready_ = false;
        observation_sequence_ = next_observation_sequence_;
        worker_failed_ = false;
        std::fill(last_actions_.begin(), last_actions_.end(), 0.0);
        action_source_->reset();
        history_.reset();
        action_feedback_.reset();
        last_feedback_obs_seq_ = 0;
        timeline_.reset();
        std::fill(action_.action.begin(), action_.action.end(), 0.0);
        for (auto& output : observation_outputs_)
            **output = 0.0;
        std::fill(observation_frame_output_->begin(), observation_frame_output_->end(), 0.0);
    }

    std::vector<Slot> slots_;
    OutputInterface<std::vector<double>> observation_frame_output_;
    std::vector<std::unique_ptr<OutputInterface<double>>> observation_outputs_;
    std::shared_ptr<PolicyActionPublisher> action_source_;

    BridgeConfig config_;
    std::unordered_set<std::string> own_output_paths_;

    std::size_t enable_slot_ = kNoSlot;
    std::size_t reset_slot_ = kNoSlot;
    std::size_t flat_hold_slot_ = kNoSlot;
    std::size_t context_slot_ = kNoSlot;
    std::size_t frame_sequence_slot_ = kNoSlot;

    std::vector<double> last_actions_;
    ObservationHistory history_;
    ActionFeedbackHistory action_feedback_;
    PolicyWorker worker_;
    ObservationTimeline timeline_;
    ObservationFrameGate frame_gate_;
    std::vector<double> frame_buffer_;
    std::vector<double> observation_buffer_;
    PolicyWorker::Result action_;
    PolicyWorker::Result result_buffer_;
    std::uint64_t last_feedback_obs_seq_ = 0;

    std::uint64_t generation_ = 0;
    std::uint64_t next_observation_sequence_ = 0;
    std::uint64_t observation_sequence_ = 0;
    std::uint64_t last_reset_count_ = 0;
    std::uint64_t obs_invalid_count_ = 0;
    bool worker_failed_ = false;
    bool action_ready_ = false;
    bool last_valid_ = false;
    bool last_enabled_ = false;
    bool last_flat_hold_ = false;
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_rl::RlBridge, rmcs_executor::Component)
