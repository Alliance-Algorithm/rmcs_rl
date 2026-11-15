#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_utility/rclcpp/node_mixin.hpp>

#include "policy/policy_worker.hpp"
#include "rl_bridge/action_delay.hpp"
#include "rl_bridge/config.hpp"
#include "rl_bridge/observation_builder.hpp"
#include "rl_bridge/observation_history.hpp"
#include "rl_bridge/policy_actions.hpp"

namespace rmcs_rl {

class RlBridge
    : public rmcs_executor::Component
    , public rclcpp::Node
    , public rmcs_utility::NodeMixin {
public:
    RlBridge()
        : Node{get_component_name(), node::options()}
        , actions_(*action_publisher_, load_action_terms(*this))
        , observation_(load_observation_terms(*this, actions_.size()))
        , history_(observation_.frame_size(), history_length())
        , worker_(
              model_path(), history_length(), observation_.frame_size(), actions_.size(),
              load_expected_metadata(*this)) {
        node::param("rl_base", rl_base_);
        node::param("policy_rate", policy_rate_);
        history_rate_ = node::param_or("history_rate", policy_rate_);
        node::param("max_action_age", max_action_age_);
        std::string enable_interface, reset_interface;
        node::param("enable_interface", enable_interface);
        node::param("reset_interface", reset_interface);

        register_input("/predefined/update_count", update_count_);
        register_input("/predefined/timestamp", timestamp_);
        register_input("/predefined/update_rate", update_rate_);
        register_input(enable_interface, enable_);
        register_input(reset_interface, reset_count_);
        action_publisher_->register_input(reset_interface, publisher_reset_count_);
        if (has_parameter("flat_hold_interface"))
            action_publisher_->register_input(
                get_parameter("flat_hold_interface").as_string(), flat_hold_);

        action_publisher_->register_output(rl_base_ + "/valid", valid_, false);
        action_publisher_->register_output(rl_base_ + "/healthy", healthy_, true);
        action_publisher_->register_output(
            rl_base_ + "/action_seq", action_sequence_, std::size_t{0});
        action_publisher_->register_output(
            rl_base_ + "/policy_rate", policy_rate_output_, policy_rate_);
        action_publisher_->register_output(
            rl_base_ + "/model_metadata", model_metadata_, worker_.metadata());

        frame_.resize(observation_.frame_size());
        previous_action_.resize(actions_.size());
        normalized_.resize(actions_.size());
        action_.action.resize(actions_.size());
        result_buffer_.action.resize(actions_.size());
        node::info(
            "policy {}: obs={}x{} action={} rate={} Hz", model_path(), history_length(),
            observation_.frame_size(), actions_.size(), policy_rate_);
    }

    void before_pairing(const OutputInfoMap& output_map) override {
        observation_.bind(*this, output_map);
    }

    void before_updating() override {
        const double update_rate = *update_rate_;
        if (!(update_rate > 0.) || !std::isfinite(update_rate) || !(history_rate_ > 0.)
            || !(policy_rate_ >= history_rate_) || std::fmod(update_rate, policy_rate_) != 0.
            || std::fmod(policy_rate_, history_rate_) != 0.)
            throw std::invalid_argument(
                "update_rate, policy_rate and history_rate must divide evenly");
        policy_divisor_ = static_cast<std::size_t>(std::lround(update_rate / policy_rate_));
        history_divisor_ = static_cast<std::size_t>(std::lround(update_rate / history_rate_));
        action_delay_ =
            ActionDelay(actions_.size(), history_length() + history_divisor_ / policy_divisor_ + 4);
    }

    void update() override {
        update_reset_count(*reset_count_);
        if (*enable_ != last_enable_) {
            last_enable_ = *enable_;
            reset();
        }
        if (*enable_ && *update_count_ % policy_divisor_ == 0)
            submit_observation();
    }

private:
    struct ActionPublisher : public rmcs_executor::Component {
        std::function<void()> fn;
        template <std::invocable Fn>
        explicit ActionPublisher(Fn&& fn)
            : fn{std::forward<Fn>(fn)} {}
        void update() override { fn(); }
    };

    void publish_actions() {
        // The partner handles reset before any action consumer executes.
        update_reset_count(*publisher_reset_count_);
        const bool hold = flat_hold_.ready() && *flat_hold_;
        if (hold != last_hold_) {
            last_hold_ = hold;
            reset();
        }
        receive_result();
        const bool was_valid = *valid_;
        const bool valid = action_valid();
        if (valid) {
            if (hold)
                std::ranges::fill(action_.action, 0.);
            if (action_.sequence != *action_sequence_) {
                actions_.normalize(action_.action, normalized_);
                action_delay_.store(action_.sequence, normalized_);
            }
            actions_.publish(action_.action);
        } else {
            actions_.publish_invalid();
        }
        *valid_ = valid;
        *healthy_ = worker_.healthy();
        *action_sequence_ = action_.sequence;
        if (was_valid && !valid)
            reset();
    }

    void submit_observation() {
        const auto tick = *update_count_;
        const auto delayed = action_delay_.select(tick - std::min(tick, history_divisor_));
        std::ranges::copy(delayed, previous_action_.begin());
        if (!observation_.build(previous_action_, frame_)) {
            reset();
            return;
        }
        if (!last_history_tick_ || tick - *last_history_tick_ >= history_divisor_) {
            history_.append(frame_);
            last_history_tick_ = tick;
        }
        action_delay_.mark_submission(++observation_sequence_, tick);
        worker_.submit(history_.values(), observation_sequence_, generation_, *timestamp_);
    }

    void receive_result() {
        if (!worker_.try_take(result_buffer_) || result_buffer_.generation != generation_)
            return;
        if (!result_buffer_.success) {
            node::error("inference failed: {}", result_buffer_.error);
            action_ready_ = false;
        } else if (
            result_buffer_.sequence > action_.sequence
            && result_buffer_.sequence <= observation_sequence_) {
            std::swap(action_, result_buffer_);
            action_ready_ = true;
        }
    }

    bool action_valid() const {
        const double age =
            std::chrono::duration<double>(*timestamp_ - action_.submitted_at).count();
        return last_enable_ && action_ready_ && worker_.healthy()
            && action_.generation == generation_ && age <= max_action_age_
            && std::ranges::all_of(
                   action_.action, [](double value) { return std::isfinite(value); });
    }

    void update_reset_count(std::size_t count) {
        if (count == last_reset_count_)
            return;
        last_reset_count_ = count;
        reset();
        worker_.reset();
    }

    void reset() {
        ++generation_;
        action_ready_ = false;
        action_.sequence = action_.generation = 0;
        action_.submitted_at = {};
        action_.success = false;
        action_.error.clear();
        std::ranges::fill(action_.action, 0.);
        action_delay_.reset();
        last_history_tick_.reset();
        std::ranges::fill(frame_, 0.);
        history_.reset();
        *valid_ = false;
        *action_sequence_ = 0;
    }

    std::string model_path() const {
        std::string path;
        node::param("rl_model_path", path);
        if (path.empty() || path.front() != '/')
            path = ament_index_cpp::get_package_share_directory("rmcs_rl") + "/" + path;
        return path;
    }

    std::size_t history_length() const {
        std::int64_t length;
        node::param("history_length", length);
        return static_cast<std::size_t>(length);
    }

    std::shared_ptr<ActionPublisher> action_publisher_{create_partner_component<ActionPublisher>(
        get_component_name() + "_actions", [this] { publish_actions(); })};

    std::string rl_base_;
    double policy_rate_, history_rate_, max_action_age_;
    std::size_t policy_divisor_ = 0, history_divisor_ = 0;
    std::optional<std::size_t> last_history_tick_;
    InputInterface<std::size_t> update_count_;
    InputInterface<PolicyWorker::Clock::time_point> timestamp_;
    InputInterface<double> update_rate_;
    InputInterface<bool> enable_, flat_hold_;
    InputInterface<std::size_t> reset_count_, publisher_reset_count_;
    OutputInterface<bool> valid_, healthy_;
    OutputInterface<std::size_t> action_sequence_;
    OutputInterface<double> policy_rate_output_;
    OutputInterface<std::map<std::string, std::string>> model_metadata_;

    PolicyActions actions_;
    ObservationBuilder observation_;
    ObservationHistory history_;
    ActionDelay action_delay_{0, 0};
    PolicyWorker worker_;

    std::vector<double> frame_, previous_action_, normalized_;
    PolicyWorker::Result action_, result_buffer_;
    std::uint64_t generation_ = 0, observation_sequence_ = 0;
    std::size_t last_reset_count_ = 0;
    bool last_enable_ = false, action_ready_ = false, last_hold_ = false;
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_rl::RlBridge, rmcs_executor::Component)
