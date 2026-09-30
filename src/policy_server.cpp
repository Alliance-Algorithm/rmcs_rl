#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_rl/msg/action.hpp>
#include <rmcs_rl/msg/observation.hpp>
#include <rmcs_rl/msg/policy_status.hpp>
#include <rmcs_rl/parameters.hpp>
#include <rmcs_rl/rl_layout.hpp>

#include "policy/policy_model.hpp"

namespace rmcs_rl {

class PolicyServer final : public rclcpp::Node {
public:
    PolicyServer()
        : Node(
              "policy_server",
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {
        const auto rl_base = string_or(*this, "rl_base", "/rl");
        PolicyModel::Config config;
        config.path = string_or(*this, "rl_model_path", "");
        if (config.path.empty())
            throw std::invalid_argument("policy_server: parameter 'rl_model_path' is required");
        if (config.path.front() != '/')
            config.path =
                ament_index_cpp::get_package_share_directory("rmcs_rl") + "/" + config.path;
        config.input_name = string_or(*this, "input_name", "obs");
        config.output_name = string_or(*this, "output_name", "actions");
        config.model_type = string_or(*this, "model_type", "auto");
        if (const auto value = integer_parameter(*this, "sequence_length"); value.has_value()) {
            if (*value < 1)
                throw std::invalid_argument("policy_server: sequence_length must be positive");
            config.sequence_length = static_cast<std::size_t>(*value);
        }
        if (const auto value = integer_parameter(*this, "feature_size"); value.has_value()) {
            if (*value < 1)
                throw std::invalid_argument("policy_server: feature_size must be positive");
            config.feature_size = static_cast<std::size_t>(*value);
        }
        if (const auto value = integer_parameter(*this, "rl_obs_size"); value.has_value()) {
            if (*value < 1)
                throw std::invalid_argument("policy_server: rl_obs_size must be positive");
            config.observation_size = static_cast<std::size_t>(*value);
        }
        if (const auto value = integer_parameter(*this, "rl_action_size"); value.has_value()) {
            if (*value < 1)
                throw std::invalid_argument("policy_server: rl_action_size must be positive");
            config.action_size = static_cast<std::size_t>(*value);
        }
        for (const auto& parameter_name : list_parameters({"extra_inputs"}, 1).names) {
            static constexpr std::string_view kPrefix = "extra_inputs.";
            if (!parameter_name.starts_with(kPrefix))
                continue;
            const auto extra_name = parameter_name.substr(kPrefix.size());
            if (extra_name.empty())
                throw std::invalid_argument(
                    "policy_server: extra_inputs entry must be extra_inputs.<input_name>");
            const auto parameter = get_parameter(parameter_name);
            OnnxRuntime::ExtraInputValues values;
            switch (parameter.get_type()) {
            case rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY:
                values = parameter.as_double_array();
                break;
            case rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY:
                values = parameter.as_integer_array();
                break;
            case rclcpp::ParameterType::PARAMETER_BOOL_ARRAY:
                values = parameter.as_bool_array();
                break;
            default:
                throw std::invalid_argument(
                    "policy_server: " + parameter_name + " must be a float, integer, or bool list");
            }
            if (std::visit([](const auto& items) { return items.empty(); }, values))
                throw std::invalid_argument("policy_server: " + parameter_name + " is empty");
            if (const auto* floats = std::get_if<std::vector<double>>(&values))
                for (const double value : *floats)
                    if (!std::isfinite(value))
                        throw std::invalid_argument(
                            "policy_server: " + parameter_name + " contains a non-finite value");
            config.extra_input_values.emplace(extra_name, std::move(values));
        }
        config.normalization_from_metadata = bool_or(*this, "normalization_from_metadata", true);
        if (const double clip = number_or(*this, "obs_clip", -1.0); clip >= 0.0)
            config.obs_clip = clip;
        if (const double clip = number_or(*this, "action_clip", -1.0); clip >= 0.0)
            config.action_clip = clip;
        model_ = std::make_unique<PolicyModel>(config);

        action_publisher_ = create_publisher<msg::Action>(
            rl_base + "/action", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort());
        observation_subscription_ = create_subscription<msg::Observation>(
            rl_base + "/obs", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort(),
            [this](msg::Observation::UniquePtr message) { on_observation(std::move(message)); });
        if (bool_or(*this, "publish_status", false)) {
            status_publisher_ = create_publisher<msg::PolicyStatus>(
                rl_base + "/policy_status",
                rclcpp::QoS{rclcpp::KeepLast(1)}.transient_local().best_effort());
            const double rate = std::max(number_or(*this, "status_rate", 2.0), 0.1);
            status_timer_ = create_wall_timer(
                std::chrono::duration<double>(1.0 / rate), [this]() { publish_status(); });
        }

        const auto& info = model_->info();
        RCLCPP_INFO(get_logger(), "policy loaded: %s", info.path.c_str());
        RCLCPP_INFO(
            get_logger(), "model_id=%s layout_hash=%s version=%s type=%s input_rank=%zu output_rank=%zu obs=%zu action=%zu",
            hex16(info.model_id).c_str(), hex16(info.layout_hash).c_str(), info.version.c_str(),
            info.model_type.c_str(), info.input_rank, info.output_rank, info.obs_size,
            info.action_size);
        RCLCPP_INFO(get_logger(), "obs signature: %s", info.obs_signature.c_str());
        RCLCPP_INFO(get_logger(), "action signature: %s", info.actions_signature.c_str());
        RCLCPP_INFO(get_logger(), "waiting for obs on %s/obs", rl_base.c_str());
    }

private:
    void on_observation(msg::Observation::UniquePtr message) {
        const auto& info = model_->info();
        if (message->layout_hash != info.layout_hash) {
            if (!layout_mismatch_logged_) {
                layout_mismatch_logged_ = true;
                RCLCPP_ERROR(
                    get_logger(), "layout_hash mismatch: bridge=%s model=%s; refusing actions",
                    hex16(message->layout_hash).c_str(), hex16(info.layout_hash).c_str());
            }
            return;
        }

        const auto started = std::chrono::steady_clock::now();
        msg::Action action;
        action.action.resize(info.action_size);
        std::string error;
        if (!model_->run(message->obs, action.action, error)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000, "policy rejected observation: %s", error.c_str());
            return;
        }
        action.header.stamp = get_clock()->now();
        action.obs_seq = message->obs_seq;
        action.layout_hash = info.layout_hash;
        action.model_id = info.model_id;
        action_publisher_->publish(action);
        const auto elapsed =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started);
        record_inference_time(elapsed.count());
    }

    void record_inference_time(double microseconds) {
        inference_window_[inference_window_cursor_] = microseconds;
        inference_window_cursor_ = (inference_window_cursor_ + 1) % inference_window_.size();
        inference_window_count_ = std::min(inference_window_count_ + 1, inference_window_.size());
    }

    double percentile(double fraction) const {
        if (inference_window_count_ == 0)
            return 0.0;
        std::vector<double> samples(
            inference_window_.begin(), inference_window_.begin() + inference_window_count_);
        std::sort(samples.begin(), samples.end());
        const auto index =
            static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1) + 0.5);
        return samples[std::min(index, samples.size() - 1)];
    }

    void publish_status() {
        const auto& info = model_->info();
        msg::PolicyStatus status;
        status.header.stamp = get_clock()->now();
        status.model_name = info.path;
        status.model_id = info.model_id;
        status.policy_version = info.version;
        status.layout_hash = info.layout_hash;
        status.obs_size = static_cast<std::uint32_t>(info.obs_size);
        status.action_size = static_cast<std::uint32_t>(info.action_size);
        status.inference_p50_us = percentile(0.50);
        status.inference_p99_us = percentile(0.99);
        status_publisher_->publish(status);
    }

    rclcpp::Publisher<msg::Action>::SharedPtr action_publisher_;
    rclcpp::Subscription<msg::Observation>::SharedPtr observation_subscription_;
    rclcpp::Publisher<msg::PolicyStatus>::SharedPtr status_publisher_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    std::unique_ptr<PolicyModel> model_;
    bool layout_mismatch_logged_ = false;
    std::array<double, 256> inference_window_{};
    std::size_t inference_window_cursor_ = 0;
    std::size_t inference_window_count_ = 0;
};

} // namespace rmcs_rl

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<rmcs_rl::PolicyServer>();
        rclcpp::spin(node);
    } catch (const std::exception& error) {
        fprintf(stderr, "[Fatal] policy_server startup failed: %s\n", error.what());
        fflush(stderr);
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
