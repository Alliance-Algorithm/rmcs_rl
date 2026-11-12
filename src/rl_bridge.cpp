#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_utility/rclcpp/node_mixin.hpp>

#include "policy/policy_worker.hpp"
#include "rl_bridge/config.hpp"
#include "rl_bridge/observation_builder.hpp"
#include "rl_bridge/observation_frame_gate.hpp"
#include "rl_bridge/observation_history.hpp"
#include "rl_bridge/observation_timeline.hpp"
#include "rl_bridge/policy_actions.hpp"
#include "rl_bridge/policy_status.hpp"

namespace rmcs_rl {

class RlBridge
    : public rmcs_executor::Component
    , public rclcpp::Node
    , public rmcs_utility::NodeMixin {

    using Clock = PolicyWorker::Clock;

public:
    RlBridge()
        : Node{get_component_name(), node::options()}
        , config_(load_bridge_config(*this))
        , action_publisher_(
              create_partner_component<ActionPublisher>(get_component_name() + "_actions", *this))
        , worker_(config_.policy)
        , actions_(*action_publisher_, config_.action_terms, config_.invalid_action)
        , status_(*action_publisher_, *this, config_, worker_.info())
        , observation_(config_.observation_terms)
        , history_(config_.frame_size, config_.history_length)
        , timeline_(config_.policy_period)
        , previous_action_(config_.action_terms.size()) {
        action_.action.resize(config_.action_terms.size());
        result_buffer_.action.reserve(config_.action_terms.size());

        if (!config_.reset_interface.empty()) {
            register_input(config_.reset_interface, reset_count_, false);
            action_publisher_->register_input(config_.reset_interface, publisher_reset_count_);
        }
        if (!config_.enable_interface.empty())
            register_input(config_.enable_interface, enable_, false);
        if (!config_.frame_sequence_interface.empty())
            register_input(config_.frame_sequence_interface, frame_sequence_);

        register_output(
            config_.rl_base + "/input/frame", frame_, std::vector<double>(config_.frame_size));
    }

    RlBridge(const RlBridge&) = delete;
    RlBridge& operator=(const RlBridge&) = delete;
    RlBridge(RlBridge&&) = delete;
    RlBridge& operator=(RlBridge&&) = delete;

    ~RlBridge() override = default;

    void before_pairing(const OutputInfoMap& output_map) override {
        observation_.bind(*this, output_map);
        log_layout();
    }

    void before_updating() override {
        if (!enable_.ready())
            enable_.make_and_bind_directly(config_.enable_default);
        if (!reset_count_.ready())
            reset_count_.make_and_bind_directly(std::size_t{0});
        if (!publisher_reset_count_.ready())
            publisher_reset_count_.make_and_bind_directly(std::size_t{0});
    }

    void update() override {
        status_.update_worker(worker_);
        update_reset_count(*reset_count_);
        update_enabled(*enable_);
        if (last_enabled_)
            update_observation(Clock::now());
    }

    void update_actions() {
        update_reset_count(*publisher_reset_count_);

        const auto now = Clock::now();
        receive_result();
        const auto check = check_action(now);
        if (check.valid) {
            actions_.publish(action_.action);
            update_previous_action();
        } else {
            actions_.publish_invalid();
        }
        status_.update_action({
            .result = action_,
            .ready = action_ready_,
            .valid = check.valid,
            .healthy = worker_.healthy() && !worker_failed_,
            .age = check.age,
            .observation_sequence = observation_sequence_,
        });
        update_validity(check);
    }

private:
    struct ActionCheck {
        bool ready = false;
        bool fresh = false;
        bool sequence_ok = false;
        bool finite = false;
        bool valid = false;
        double age = std::numeric_limits<double>::quiet_NaN();
    };

    void update_reset_count(std::size_t reset_count) {
        if (reset_count == last_reset_count_)
            return;
        last_reset_count_ = reset_count;
        frame_gate_.reset();
        reset_session();
        worker_.reset();
    }

    void update_enabled(bool enabled) {
        if (enabled == last_enabled_)
            return;
        last_enabled_ = enabled;
        frame_gate_.reset();
        reset_session();
    }

    void update_observation(Clock::time_point now) {
        if (config_.frame_sequence_interface.empty()) {
            if (timeline_.due(now))
                submit_observation(now);
            return;
        }
        switch (frame_gate_.observe(*frame_sequence_)) {
        case ObservationFrameGate::Decision::kSubmit: submit_observation(now); break;
        case ObservationFrameGate::Decision::kReset: reset_session(); break;
        case ObservationFrameGate::Decision::kWait: break;
        }
    }

    void submit_observation(Clock::time_point now) {
        if (const auto built = observation_.build(previous_action_, *frame_); !built) {
            reset_session();
            ++skipped_frames_;
            if (now - last_skip_warning_ >= std::chrono::seconds{1}) {
                last_skip_warning_ = now;
                node::warn(
                    "observation '{}' is not ready or non-finite; {} frames skipped", built.error(),
                    skipped_frames_);
            }
            return;
        }
        const auto observation = history_.append(*frame_);
        observation_sequence_ = ++next_observation_sequence_;
        timeline_.advance(now);
        worker_.submit(observation, observation_sequence_, generation_, now);
    }

    void receive_result() {
        if (!worker_.try_take(result_buffer_) || result_buffer_.generation != generation_)
            return;
        if (!result_buffer_.success) {
            worker_failed_ = true;
            action_ready_ = false;
            node::error("RL inference failed: {}", result_buffer_.error);
            return;
        }
        if (result_buffer_.sequence > action_.sequence
            && result_buffer_.sequence <= observation_sequence_) {
            std::swap(action_, result_buffer_);
            action_ready_ = true;
        }
    }

    ActionCheck check_action(Clock::time_point now) const {
        if (!action_ready_)
            return {};
        ActionCheck check{.ready = true};
        check.age = std::chrono::duration<double>(now - action_.submitted_at).count();
        check.fresh = std::isfinite(check.age) && check.age <= config_.inference_timeout
                   && check.age <= config_.max_action_age;
        check.sequence_ok =
            action_.generation == generation_ && action_.sequence <= observation_sequence_;
        check.finite =
            std::ranges::all_of(action_.action, [](double value) { return std::isfinite(value); });
        check.valid =
            last_enabled_ && !worker_failed_ && check.fresh && check.sequence_ok && check.finite;
        return check;
    }

    std::string_view invalid_reason(const ActionCheck& check) const {
        if (worker_failed_)
            return "inference worker failed";
        if (!last_enabled_)
            return "disabled by enable interface";
        if (!check.ready)
            return "no inference result available";
        if (!check.finite)
            return "action contains non-finite values";
        if (!check.sequence_ok)
            return "action is from an invalid or previous reset generation";
        if (!check.fresh)
            return "inference result is stale";
        return "unknown";
    }

    void update_previous_action() {
        if (action_.sequence == feedback_sequence_)
            return;
        actions_.normalize(action_.action, previous_action_);
        feedback_sequence_ = action_.sequence;
    }

    void update_validity(const ActionCheck& check) {
        if (check.valid == last_valid_)
            return;
        last_valid_ = check.valid;
        if (check.valid) {
            node::info("valid=1 (obs_seq={} action_age={:.4f} s)", action_.sequence, check.age);
            return;
        }
        node::warn("valid=0: {}", invalid_reason(check));
        reset_session();
    }

    void reset_session() {
        ++generation_;
        action_ready_ = false;
        worker_failed_ = false;
        observation_sequence_ = next_observation_sequence_;
        feedback_sequence_ = 0;
        std::ranges::fill(action_.action, 0.0);
        std::ranges::fill(previous_action_, 0.0);
        actions_.reset();
        status_.reset();
        history_.reset();
        timeline_.reset();
        std::ranges::fill(*frame_, 0.0);
    }

    void log_layout() const {
        node::info("observation layout (obs_size={}):", config_.observation_size);
        for (std::size_t i = 0; i < config_.observation_terms.size(); ++i) {
            const auto& term = config_.observation_terms[i];
            auto line = std::format(
                "  [{}:{}) {}  scale={}", term.index, term.index + term.size, term.name,
                term.scale);
            if (term.default_value)
                line += std::format(" default={}", *term.default_value);
            if (term.clip)
                line += std::format(" clip={}:{}", term.clip->lower, term.clip->upper);
            if (!term.source.empty())
                line += std::format(" <- {} ({})", term.source, observation_.source_type(i));
            node::info("{}", line);
        }

        node::info("action layout (action_size={}):", config_.action_terms.size());
        for (std::size_t i = 0; i < config_.action_terms.size(); ++i) {
            const auto& term = config_.action_terms[i];
            auto line = std::format("  #{} {} -> {}", i, term.name, term.output);
            if (term.scale != 1.0)
                line += std::format(" scale={}", term.scale);
            node::info("{}", line);
        }

        const auto& model = worker_.info();
        node::info(
            "contract: obs_size={} action_size={} policy_rate={:.3f} Hz", config_.observation_size,
            config_.action_terms.size(), config_.policy_rate);
        node::info(
            "in-process policy worker: model={}, obs={}, action={}", model.path,
            model.observation_size, model.action_size);
    }

    BridgeConfig config_;

    class ActionPublisher : public rmcs_executor::Component {
    public:
        explicit ActionPublisher(RlBridge& bridge)
            : bridge_(bridge) {}

        void update() override { bridge_.update_actions(); }

    private:
        RlBridge& bridge_;
    };
    std::shared_ptr<ActionPublisher> action_publisher_;

    PolicyWorker worker_;
    PolicyActions actions_;
    PolicyStatus status_;

    ObservationBuilder observation_;
    ObservationHistory history_;
    ObservationTimeline timeline_;
    ObservationFrameGate frame_gate_;

    InputInterface<bool> enable_;
    InputInterface<std::size_t> reset_count_;
    InputInterface<std::size_t> publisher_reset_count_;
    InputInterface<std::size_t> frame_sequence_;
    OutputInterface<std::vector<double>> frame_;

    PolicyWorker::Result action_;
    PolicyWorker::Result result_buffer_;
    std::vector<double> previous_action_;

    std::uint64_t generation_ = 0;
    std::uint64_t next_observation_sequence_ = 0;
    std::uint64_t observation_sequence_ = 0;
    std::uint64_t feedback_sequence_ = 0;
    std::size_t last_reset_count_ = 0;
    bool worker_failed_ = false;
    bool action_ready_ = false;
    bool last_valid_ = false;
    bool last_enabled_ = false;

    std::uint64_t skipped_frames_ = 0;
    Clock::time_point last_skip_warning_{};
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_rl::RlBridge, rmcs_executor::Component)
