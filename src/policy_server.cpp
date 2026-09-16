
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rmcs_rl/msg/action.hpp>
#include <rmcs_rl/msg/observation.hpp>
#include <rmcs_rl/msg/policy_status.hpp>

#include "onnxruntime_inference.hpp"
#include "rl_layout.hpp"

namespace rmcs::rl {

namespace {

    std::vector<double> parse_float_list(const std::string& text, const char* what) {
        std::vector<double> values;
        std::size_t begin = 0;
        while (begin <= text.size()) {
            const auto end = text.find_first_of(", \t", begin);
            const auto piece =
                text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
            if (!piece.empty()) {
                try {
                    std::size_t consumed = 0;
                    const double value   = std::stod(piece, &consumed);
                    if (consumed != piece.size()) throw std::invalid_argument("trailing");
                    values.push_back(value);
                } catch (const std::exception&) {
                    throw std::invalid_argument(
                        std::string { "policy_server: metadata " } + what + " has a non-number '"
                        + piece + "'");
                }
            }
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        return values;
    }

    std::optional<double> parse_float(const std::string& text) {
        try {
            std::size_t consumed = 0;
            const double value   = std::stod(text, &consumed);
            if (consumed != text.size()) return std::nullopt;
            return value;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    double parse_clip_metadata(const std::string& text, const char* key) {
        const auto value = parse_float(text);
        if (!value.has_value())
            throw std::runtime_error(
                std::string { "policy_server: metadata " } + key + "=" + text + " is not a float");
        if (!std::isfinite(*value) || *value <= 0.0)
            throw std::runtime_error(std::string { "policy_server: metadata " } + key
                + " must be finite and > 0, got " + text);
        return *value;
    }

} // namespace

class PolicyServer final : public rclcpp::Node {
public:
    PolicyServer()
        : Node("policy_server",
              rclcpp::NodeOptions { }.automatically_declare_parameters_from_overrides(true)) {

        rl_base_ = string_or_("rl_base", "/rl");

        const std::string model_path = string_or_("rl_model_path", "");
        if (model_path.empty())
            throw std::invalid_argument("policy_server: parameter 'rl_model_path' is required");
        resolved_model_path_ = resolve_model_path_(model_path);

        std::string load_error;
        OnnxRuntimeInference::Config inference_config;
        inference_config.model_path  = resolved_model_path_;
        inference_config.input_name  = string_or_("input_name", "obs");
        inference_config.output_name = string_or_("output_name", "actions");
        if (!inference_.load(inference_config, load_error))
            throw std::runtime_error(
                "policy_server: cannot load '" + resolved_model_path_ + "': " + load_error);

        obs_size_    = inference_.input_size();
        action_size_ = inference_.output_size();
        if (obs_size_ == 0 || action_size_ == 0)
            throw std::runtime_error("policy_server: model has zero-sized obs/action tensor");

        std::string model_id_error;
        if (!model_id_of_file(resolved_model_path_, model_id_, model_id_error))
            throw std::runtime_error("policy_server: " + model_id_error);

        const auto obs_layout = inference_.metadata("rmcs_obs_layout");
        const auto actions_layout = inference_.metadata("rmcs_actions_layout");
        if (!obs_layout || obs_layout->empty() || !actions_layout || actions_layout->empty())
            throw std::runtime_error("policy_server: model '" + resolved_model_path_
                + "' is missing metadata 'rmcs_obs_layout' / 'rmcs_actions_layout'; stamp it with "
                  "tool/stamp_layout_metadata.py before deploying (see doc/bridge-design.md §6)");
        obs_signature_     = *obs_layout;
        actions_signature_ = *actions_layout;
        policy_version_    = inference_.metadata("policy_version").value_or("");
        layout_hash_ = rmcs::rl::layout_hash(obs_signature_, actions_signature_, obs_size_, action_size_);

        if (const auto declared = inference_.metadata("policy_layout_hash");
            declared && !declared->empty()) {
            std::uint64_t declared_hash = 0;
            try {
                declared_hash = std::stoull(*declared, nullptr, 16);
            } catch (const std::exception&) {
                throw std::runtime_error("policy_server: metadata 'policy_layout_hash' is not a "
                                         "hex string: '"
                    + *declared + "'");
            }
            if (declared_hash != layout_hash_)
                throw std::runtime_error("policy_server: metadata 'policy_layout_hash'="
                    + *declared + " does not match the hash computed from the layout metadata ("
                    + hex16(layout_hash_) + "); re-stamp the model");
        }

        load_normalization_();

        action_publisher_ = create_publisher<rmcs_rl::msg::Action>(
            rl_base_ + "/action", rclcpp::QoS { rclcpp::KeepLast(1) }.best_effort());
        observation_subscription_ = create_subscription<rmcs_rl::msg::Observation>(
            rl_base_ + "/obs", rclcpp::QoS { rclcpp::KeepLast(1) }.best_effort(),
            [this](rmcs_rl::msg::Observation::UniquePtr message) {
                on_observation_(std::move(message));
            });

        if (bool_or_("publish_status", false)) {
            status_publisher_ = create_publisher<rmcs_rl::msg::PolicyStatus>(
                rl_base_ + "/policy_status",
                rclcpp::QoS { rclcpp::KeepLast(1) }.transient_local().best_effort());
            const double rate = std::max(number_or_("status_rate", 2.0), 0.1);
            status_timer_     = create_wall_timer(
                std::chrono::duration<double>(1.0 / rate), [this]() { publish_status_(); });
        }

        RCLCPP_INFO(get_logger(), "policy loaded: %s", resolved_model_path_.c_str());
        RCLCPP_INFO(get_logger(), "  model_id      : %s", hex16(model_id_).c_str());
        RCLCPP_INFO(get_logger(), "  policy_version: %s",
            policy_version_.empty() ? "(none)" : policy_version_.c_str());
        RCLCPP_INFO(get_logger(), "  obs_size=%zu action_size=%zu", obs_size_, action_size_);
        RCLCPP_INFO(get_logger(), "  obs signature    : %s", obs_signature_.c_str());
        RCLCPP_INFO(get_logger(), "  action signature : %s", actions_signature_.c_str());
        RCLCPP_INFO(get_logger(), "  layout_hash   : %s  (bridge must log the same value)",
            hex16(layout_hash_).c_str());
        if (!obs_mean_.empty())
            RCLCPP_INFO(get_logger(), "  normalization : mean/std from metadata, obs_clip=%s",
                obs_clip_.has_value() ? std::to_string(*obs_clip_).c_str() : "(none)");
        RCLCPP_INFO(get_logger(), "waiting for obs on %s/obs", rl_base_.c_str());
    }

private:
    std::string string_or_(const std::string& name, const std::string& fallback) const {
        rclcpp::Parameter parameter;
        try {
            if (!get_parameter(name, parameter)) return fallback;
        } catch (const std::exception&) {
            return fallback;
        }
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) return fallback;
        return parameter.as_string();
    }

    double number_or_(const std::string& name, double fallback) const {
        rclcpp::Parameter parameter;
        try {
            if (!get_parameter(name, parameter)) return fallback;
        } catch (const std::exception&) {
            return fallback;
        }
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
            return parameter.as_double();
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
            return static_cast<double>(parameter.as_int());
        return fallback;
    }

    bool bool_or_(const std::string& name, bool fallback) const {
        rclcpp::Parameter parameter;
        try {
            if (!get_parameter(name, parameter)) return fallback;
        } catch (const std::exception&) {
            return fallback;
        }
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) return fallback;
        return parameter.as_bool();
    }

    static std::string resolve_model_path_(const std::string& path) {
        if (path.empty() || path.front() == '/') return path;
        try {
            return ament_index_cpp::get_package_share_directory("rmcs_rl") + "/" + path;
        } catch (const std::exception&) {
            return path;
        }
    }

    void load_normalization_() {
        const bool from_metadata = bool_or_("normalization_from_metadata", true);
        if (from_metadata) {
            if (const auto value = inference_.metadata("rmcs_obs_mean"); value && !value->empty())
                obs_mean_ = parse_float_list(*value, "rmcs_obs_mean");
            if (const auto value = inference_.metadata("rmcs_obs_std"); value && !value->empty())
                obs_std_ = parse_float_list(*value, "rmcs_obs_std");
            if (!obs_mean_.empty() && obs_mean_.size() != obs_size_)
                throw std::runtime_error("policy_server: metadata rmcs_obs_mean has "
                    + std::to_string(obs_mean_.size()) + " values, expected "
                    + std::to_string(obs_size_));
            if (!obs_std_.empty() && obs_std_.size() != obs_size_)
                throw std::runtime_error("policy_server: metadata rmcs_obs_std has "
                    + std::to_string(obs_std_.size()) + " values, expected "
                    + std::to_string(obs_size_));
            if (!obs_mean_.empty() && !obs_std_.empty()) {
                for (const double sigma : obs_std_)
                    if (!(sigma > 0.0) || !std::isfinite(sigma))
                        throw std::runtime_error(
                            "policy_server: metadata rmcs_obs_std must be finite and > 0");
            } else {
                obs_mean_.clear();
                obs_std_.clear();
            }
            if (const auto value = inference_.metadata("rmcs_obs_clip"); value && !value->empty())
                obs_clip_ = parse_clip_metadata(*value, "rmcs_obs_clip");
            if (const auto value = inference_.metadata("rmcs_action_clip");
                value && !value->empty())
                action_clip_ = parse_clip_metadata(*value, "rmcs_action_clip");
        }

        const double obs_clip_param = number_or_("obs_clip", -1.0);
        if (obs_clip_param >= 0.0) obs_clip_ = obs_clip_param;
        const double action_clip_param = number_or_("action_clip", -1.0);
        if (action_clip_param >= 0.0) action_clip_ = action_clip_param;

        obs_buffer_.assign(obs_size_, 0.0F);
        action_buffer_.assign(action_size_, 0.0F);
    }

    void on_observation_(rmcs_rl::msg::Observation::UniquePtr message) {
        if (message->layout_hash != layout_hash_) {
            if (!layout_mismatch_logged_) {
                layout_mismatch_logged_ = true;
                RCLCPP_FATAL(get_logger(),
                    "layout_hash mismatch: bridge sent %s but this model expects %s "
                    "(model '%s'); refusing to answer (bridge will report valid=0). "
                    "Check that the deployment YAML and the model metadata describe the same "
                    "observation/action terms.",
                    hex16(message->layout_hash).c_str(), hex16(layout_hash_).c_str(),
                    resolved_model_path_.c_str());
            }
            ++rejected_count_;
            return;
        }
        if (message->obs.size() != obs_size_) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                "obs has %zu values, model expects %zu; ignoring", message->obs.size(), obs_size_);
            ++rejected_count_;
            return;
        }

        const auto started = std::chrono::steady_clock::now();

        for (std::size_t i = 0; i < obs_size_; ++i) {
            double value = message->obs[i];
            if (!obs_mean_.empty()) value = (value - obs_mean_[i]) / obs_std_[i];
            if (obs_clip_.has_value()) value = std::clamp(value, -*obs_clip_, *obs_clip_);
            if (!std::isfinite(value)) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                    "obs[%zu] is not finite; ignoring frame", i);
                ++rejected_count_;
                return;
            }
            obs_buffer_[i] = static_cast<float>(value);
        }

        if (!inference_.run(obs_buffer_, action_buffer_)) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
                "ONNX Runtime rejected the obs frame; ignoring");
            ++rejected_count_;
            return;
        }

        rmcs_rl::msg::Action action;
        action.header.stamp = get_clock()->now();
        action.obs_seq      = message->obs_seq;
        action.layout_hash  = layout_hash_;
        action.model_id     = model_id_;
        action.action.resize(action_size_);
        for (std::size_t i = 0; i < action_size_; ++i) {
            double value = static_cast<double>(action_buffer_[i]);
            if (action_clip_.has_value()) value = std::clamp(value, -*action_clip_, *action_clip_);
            action.action[i] = value;
        }
        action_publisher_->publish(action);

        const auto elapsed = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - started);
        record_inference_time_(elapsed.count());
        ++served_count_;
    }

    void record_inference_time_(double microseconds) {
        inference_window_[inference_window_cursor_] = microseconds;
        inference_window_cursor_ = (inference_window_cursor_ + 1) % inference_window_.size();
        inference_window_count_  = std::min(inference_window_count_ + 1, inference_window_.size());
    }

    double percentile_(double fraction) const {
        if (inference_window_count_ == 0) return 0.0;
        std::vector<double> samples(
            inference_window_.begin(), inference_window_.begin() + inference_window_count_);
        std::sort(samples.begin(), samples.end());
        const auto index = static_cast<std::size_t>(
            fraction * static_cast<double>(samples.size() - 1) + 0.5);
        return samples[std::min(index, samples.size() - 1)];
    }

    void publish_status_() {
        if (!status_publisher_) return;
        rmcs_rl::msg::PolicyStatus status;
        status.header.stamp       = get_clock()->now();
        status.model_name         = resolved_model_path_;
        status.model_id           = model_id_;
        status.policy_version     = policy_version_;
        status.layout_hash        = layout_hash_;
        status.obs_size           = static_cast<std::uint32_t>(obs_size_);
        status.action_size        = static_cast<std::uint32_t>(action_size_);
        status.inference_p50_us   = percentile_(0.50);
        status.inference_p99_us   = percentile_(0.99);
        status_publisher_->publish(status);
    }

    std::string rl_base_;
    std::string resolved_model_path_;
    std::string obs_signature_;
    std::string actions_signature_;
    std::string policy_version_;
    std::size_t obs_size_    = 0;
    std::size_t action_size_ = 0;
    std::uint64_t layout_hash_ = 0;
    std::uint64_t model_id_    = 0;

    std::vector<double> obs_mean_;
    std::vector<double> obs_std_;
    std::optional<double> obs_clip_;
    std::optional<double> action_clip_;

    std::vector<float> obs_buffer_;
    std::vector<float> action_buffer_;

    std::uint64_t served_count_   = 0;
    std::uint64_t rejected_count_ = 0;
    bool layout_mismatch_logged_  = false;

    std::array<double, 256> inference_window_ { };
    std::size_t inference_window_cursor_ = 0;
    std::size_t inference_window_count_  = 0;

    OnnxRuntimeInference inference_;
    rclcpp::Publisher<rmcs_rl::msg::Action>::SharedPtr action_publisher_;
    rclcpp::Subscription<rmcs_rl::msg::Observation>::SharedPtr observation_subscription_;
    rclcpp::Publisher<rmcs_rl::msg::PolicyStatus>::SharedPtr status_publisher_;
    rclcpp::TimerBase::SharedPtr status_timer_;
};

} // namespace rmcs::rl

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<rmcs::rl::PolicyServer>();
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
