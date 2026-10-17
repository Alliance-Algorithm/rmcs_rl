#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
#include <rmcs_rl/rl_bridge/types.hpp>
#include <rmcs_rl/rl_bridge/utility.hpp>

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
        , history_(config_.obs_frame_size, config_.history_length, config_.history_period)
        , action_feedback_(config_.action_size, feedback_capacity_(config_))
        , worker_(config_.policy_worker)
        , frame_buffer_(config_.obs_frame_size)
        , observation_buffer_(config_.obs_size)
        , result_buffer_{} {
        result_buffer_.action.reserve(config_.action_size);

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
        register_output(
            config_.rl_base + "/observation_size", observation_size_output_, config_.obs_size);
        register_output(config_.rl_base + "/action_size", action_size_output_, config_.action_size);
        register_output(
            config_.rl_base + "/history_length", history_length_output_, config_.history_length);
        register_output(
            config_.rl_base + "/feature_size", feature_size_output_, config_.obs_frame_size);
        register_output(config_.rl_base + "/performance/inference_us", inference_time_us_, 0.0);
        own_output_paths_.insert(config_.rl_base + "/valid");
        own_output_paths_.insert(config_.rl_base + "/healthy");
        own_output_paths_.insert(config_.rl_base + "/action_age");
        own_output_paths_.insert(config_.rl_base + "/obs_seq");
        last_actions_.assign(config_.action_size, 0.0);
        written_.assign(config_.action_size, 0.0);
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
            worker_->info().path.c_str(), worker_->info().obs_size, worker_->info().action_size);
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
            reset_action_state(last_actions_, written_);
            history_.reset();
        }
        if (worker_->try_take(result_buffer_)) {
            if (result_buffer_.generation == generation_
                && result_buffer_.sequence > action_sequence_
                && result_buffer_.sequence <= observation_sequence_ && result_buffer_.success) {
                action_sequence_ = result_buffer_.sequence;
                action_generation_ = result_buffer_.generation;
                action_submitted_at_ = result_buffer_.submitted_at;
                action_completed_at_ = result_buffer_.completed_at;
                std::copy(
                    result_buffer_.action.begin(), result_buffer_.action.end(),
                    action_buffer_.begin());
                action_ready_ = true;
                *inference_time_us_ = result_buffer_.inference_microseconds;
                if (action_sequence_ != last_feedback_obs_seq_) {
                    action_feedback_.store(action_submitted_at_, action_buffer_);
                    last_feedback_obs_seq_ = action_sequence_;
                }
            } else if (result_buffer_.generation == generation_ && !result_buffer_.success) {
                worker_failed_ = true;
                action_ready_ = false;
                RCLCPP_ERROR_THROTTLE(
                    get_logger(), *get_clock(), 1000, "RL inference failed: %s",
                    result_buffer_.error.c_str());
            }
        }
        const bool has_snapshot = action_ready_;

        const double age = has_snapshot
                             ? std::chrono::duration<double>(now - action_submitted_at_).count()
                             : std::numeric_limits<double>::quiet_NaN();
        const bool seq_ok = has_snapshot && action_generation_ == generation_
                         && action_sequence_ <= observation_sequence_;
        const bool finite =
            has_snapshot
            && std::all_of(action_buffer_.begin(), action_buffer_.end(), [](double value) {
                   return std::isfinite(value);
               });

        const bool fresh = has_snapshot && is_finite(age) && age <= config_.inference_timeout;
        const bool valid = enabled && !worker_failed_ && has_snapshot && fresh && seq_ok && finite;

        // The action fed into history must match the flat-ground command that is executed.
        if (valid && flat_hold) {
            std::fill(action_buffer_.begin(), action_buffer_.end(), 0.0);
            if (action_sequence_ != last_feedback_obs_seq_) {
                action_feedback_.store(action_submitted_at_, action_buffer_);
                last_feedback_obs_seq_ = action_sequence_;
            }
        }
        write_actions(
            valid, std::span<const double>{action_buffer_}, config_.action_terms,
            config_.invalid_mode, written_, action_outputs_);

        *valid_output_ = valid ? 1.0 : 0.0;
        *healthy_output_ = (worker_->healthy() && !worker_failed_ && fresh) ? 1.0 : 0.0;
        *action_age_output_ = age;
        *obs_seq_output_ = observation_sequence_;

        if (enabled && timeline_.due(now, config_.pub_period))
            maybe_submit_observation_(now);

        if (last_valid_ && !valid)
            reset_runtime_();
        if (valid != last_valid_) {
            if (valid) {
                RCLCPP_INFO(
                    get_logger(), "valid=1 (obs_seq=%llu action_age=%.4f s)",
                    static_cast<unsigned long long>(action_sequence_), age);
            } else {
                RCLCPP_WARN(
                    get_logger(), "valid=0: %s",
                    invalid_reason(enabled, !worker_failed_, has_snapshot, fresh, seq_ok, finite)
                        .c_str());
            }
            last_valid_ = valid;
        }
    }

private:
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
        if (!config_.enable_path.empty())
            enable_slot_ = acquire_slot(
                *this, config_.enable_path, {Binding::kBool, Binding::kDouble}, false, output_map,
                slots_, "enable interface");

        if (!config_.reset_path.empty())
            reset_slot_ = acquire_slot(
                *this, config_.reset_path, {Binding::kSize, Binding::kInt, Binding::kDouble}, false,
                output_map, slots_, "reset interface");
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
            worker_->submit(
                std::span<const double>{observation_buffer_}, observation_sequence_, generation_,
                now);
        } else {
            reset_runtime_();
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

    void reset_runtime_() {
        ++generation_;
        action_ready_ = false;
        observation_sequence_ = next_observation_sequence_;
        worker_failed_ = false;
        reset_action_state(last_actions_, written_);
        history_.reset();
        action_feedback_.reset();
        last_feedback_obs_seq_ = 0;
        timeline_.reset();
        std::fill(action_buffer_.begin(), action_buffer_.end(), 0.0);
    }

    std::vector<Slot> slots_;
    std::vector<std::unique_ptr<OutputInterface<double>>> action_outputs_;

    OutputInterface<double> valid_output_{};
    OutputInterface<double> healthy_output_{};
    OutputInterface<double> action_age_output_{};
    OutputInterface<std::size_t> obs_seq_output_{};
    OutputInterface<std::size_t> observation_size_output_;
    OutputInterface<std::size_t> action_size_output_;
    OutputInterface<std::size_t> history_length_output_;
    OutputInterface<std::size_t> feature_size_output_;
    OutputInterface<double> inference_time_us_;

    BridgeConfig config_;
    std::unordered_set<std::string> own_output_paths_;

    std::size_t enable_slot_ = kNoSlot;
    std::size_t reset_slot_ = kNoSlot;
    std::size_t flat_hold_slot_ = kNoSlot;

    std::vector<double> last_actions_;
    std::vector<double> written_;
    ObservationHistory history_;
    ActionFeedbackHistory action_feedback_;
    std::shared_ptr<PolicyWorker> worker_;
    ObservationTimeline timeline_;
    std::vector<double> frame_buffer_;
    std::vector<double> observation_buffer_;
    std::vector<double> action_buffer_;
    PolicyWorker::Result result_buffer_;
    std::uint64_t last_feedback_obs_seq_ = 0;

    std::uint64_t generation_ = 0;
    std::uint64_t next_observation_sequence_ = 0;
    std::uint64_t observation_sequence_ = 0;
    std::uint64_t action_sequence_ = 0;
    std::uint64_t action_generation_ = 0;
    std::chrono::steady_clock::time_point action_submitted_at_{};
    std::chrono::steady_clock::time_point action_completed_at_{};
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
