#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <rmcs_executor/component.hpp>

#include "policy/policy_worker.hpp"
#include "rl_bridge/config.hpp"

namespace rmcs_rl {

class PolicyStatus {
public:
    using Clock = PolicyWorker::Clock;

    struct Action {
        const PolicyWorker::Result& result;
        bool ready;
        bool valid;
        bool healthy;
        double age;
        std::uint64_t observation_sequence;
    };

    PolicyStatus(
        rmcs_executor::Component& action_component, rmcs_executor::Component& observation_component,
        const BridgeConfig& config, const PolicyModel::Info& model) {
        const auto& base = config.rl_base;
        action_component.register_output(base + "/valid", valid_, 0.0);
        action_component.register_output(base + "/healthy", healthy_, 0.0);
        action_component.register_output(
            base + "/action_age", action_age_, std::numeric_limits<double>::quiet_NaN());
        action_component.register_output(base + "/obs_seq", obs_seq_, std::size_t{0});
        action_component.register_output(base + "/action_seq", action_seq_, std::size_t{0});
        action_component.register_output(
            base + "/action_steady_ns", action_steady_ns_, std::uint64_t{0});

        action_component.register_output(base + "/policy_rate", policy_rate_, config.policy_rate);
        action_component.register_output(
            base + "/model_spec_sha256", model_spec_sha256_, model.model_spec_sha256);
        action_component.register_output(
            base + "/observation_contract_sha256", observation_contract_sha256_,
            model.observation_contract_sha256);

        action_component.register_output(base + "/performance/inference_us", inference_us_, 0.0);
        action_component.register_output(
            base + "/latency/queued_ns", latency_.queued, std::uint64_t{0});
        action_component.register_output(
            base + "/latency/inference_start_ns", latency_.inference_start, std::uint64_t{0});
        action_component.register_output(
            base + "/latency/inference_end_ns", latency_.inference_end, std::uint64_t{0});
        action_component.register_output(
            base + "/latency/result_ready_ns", latency_.result_ready, std::uint64_t{0});
        action_component.register_output(
            base + "/latency/first_publish_ns", latency_.first_publish, std::uint64_t{0});

        observation_component.register_output(
            base + "/latency/submit_attempts", submit_attempts_, std::uint64_t{0});
        observation_component.register_output(
            base + "/latency/submit_rejected", submit_rejected_, std::uint64_t{0});
        observation_component.register_output(
            base + "/latency/pending_overwritten", pending_overwritten_, std::uint64_t{0});
        observation_component.register_output(
            base + "/latency/result_overwritten", result_overwritten_, std::uint64_t{0});
    }

    PolicyStatus(const PolicyStatus&) = delete;
    PolicyStatus& operator=(const PolicyStatus&) = delete;
    PolicyStatus(PolicyStatus&&) = delete;
    PolicyStatus& operator=(PolicyStatus&&) = delete;

    ~PolicyStatus() = default;

    void update_action(const Action& action) {
        const auto& result = action.result;
        *valid_ = action.valid ? 1.0 : 0.0;
        *healthy_ = action.healthy ? 1.0 : 0.0;
        *action_age_ = action.age;
        *obs_seq_ = action.observation_sequence;
        *action_seq_ = result.sequence;
        *action_steady_ns_ = steady_ns(result.submitted_at);

        if (action.ready) {
            *inference_us_ = std::chrono::duration<double, std::micro>(
                                 result.inference_completed_at - result.inference_started_at)
                                 .count();
            *latency_.queued = steady_ns(result.queued_at);
            *latency_.inference_start = steady_ns(result.inference_started_at);
            *latency_.inference_end = steady_ns(result.inference_completed_at);
            *latency_.result_ready = steady_ns(result.ready_at);
        }
        if (action.valid
            && (published_sequence_ != result.sequence
                || published_generation_ != result.generation)) {
            *latency_.first_publish = steady_ns(Clock::now());
            published_sequence_ = result.sequence;
            published_generation_ = result.generation;
        }
    }

    void update_worker(const PolicyWorker& worker) {
        *submit_attempts_ = worker.submit_attempts();
        *submit_rejected_ = worker.submit_rejected();
        *pending_overwritten_ = worker.pending_overwritten();
        *result_overwritten_ = worker.result_overwritten();
    }

    void reset() {
        published_sequence_ = published_generation_ = 0;
        *latency_.queued = *latency_.inference_start = *latency_.inference_end = 0;
        *latency_.result_ready = *latency_.first_publish = 0;
    }

private:
    template <typename T>
    using OutputInterface = rmcs_executor::Component::OutputInterface<T>;

    struct LatencyOutputs {
        OutputInterface<std::uint64_t> queued, inference_start, inference_end, result_ready,
            first_publish;
    };

    static std::uint64_t steady_ns(Clock::time_point time) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count());
    }

    OutputInterface<double> valid_, healthy_, action_age_;
    OutputInterface<std::size_t> obs_seq_, action_seq_;
    OutputInterface<std::uint64_t> action_steady_ns_;

    OutputInterface<double> policy_rate_;
    OutputInterface<std::string> model_spec_sha256_, observation_contract_sha256_;

    OutputInterface<double> inference_us_;
    LatencyOutputs latency_;
    OutputInterface<std::uint64_t> submit_attempts_, submit_rejected_;
    OutputInterface<std::uint64_t> pending_overwritten_, result_overwritten_;

    std::uint64_t published_sequence_ = 0;
    std::uint64_t published_generation_ = 0;
};

} // namespace rmcs_rl
