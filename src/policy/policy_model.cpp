#include "policy/policy_model.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "policy/onnx_runtime.hpp"

namespace rmcs_rl {
namespace {

double parse_number(const std::string& text, const char* key) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value))
        throw std::invalid_argument(std::string{"non-finite or invalid metadata: "} + key);
    return value;
}

std::vector<double> parse_numbers(const std::string& text, const char* key) {
    std::vector<double> result;
    for (std::size_t begin = 0; begin < text.size();) {
        begin = text.find_first_not_of(", \t", begin);
        if (begin == std::string::npos)
            break;
        const auto end = text.find_first_of(", \t", begin);
        result.push_back(parse_number(text.substr(begin, end - begin), key));
        if (end == std::string::npos)
            break;
        begin = end;
    }
    return result;
}

bool all_finite(std::span<const float> values) {
    return std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
}

} // namespace

struct PolicyModel::Impl {
    explicit Impl(const PolicyConfig& config)
        : runtime(config) {
        info.path = config.path;
        info.model_spec_sha256 = metadata("leg_target_model_spec_sha256")
                                     .value_or(metadata("rmcs_model_spec_sha256").value_or(""));
        info.observation_contract_sha256 =
            metadata("rmcs_observation_contract_sha256").value_or("");
        info.observation_size = runtime.input_size();
        info.action_size = runtime.output_size();

        check_contract(config);
        load_preprocessing(config);

        observation_buffer.resize(info.observation_size);
        action_buffer.resize(info.action_size);
        runtime.run(observation_buffer, action_buffer);
        if (!all_finite(action_buffer))
            throw std::runtime_error(
                "startup inference produced non-finite actions: " + config.path);
    }

    std::optional<std::string> metadata(const char* key) const { return runtime.metadata(key); }

    void check_contract(const PolicyConfig& config) const {
        if (!config.expected_model_spec_sha256.empty()
            && info.model_spec_sha256 != config.expected_model_spec_sha256)
            throw std::invalid_argument("model_spec_sha256 does not match the configured contract");
        for (const auto& [key, expected] : config.expected_metadata)
            if (metadata(key.c_str()) != std::optional<std::string>{expected})
                throw std::invalid_argument("model metadata mismatch: expected_metadata." + key);

        if (!config.expected_observation_contract_sha256.empty()
            && info.observation_contract_sha256 != config.expected_observation_contract_sha256)
            throw std::invalid_argument(
                "model observation contract does not match its preprocessor");
        if (!config.expected_observation_version.empty()
            && metadata("rmcs_observation_version").value_or("")
                   != config.expected_observation_version)
            throw std::invalid_argument(
                "model observation version does not match its preprocessor");
        if (!config.expected_obs_normalization.empty()) {
            if ((config.expected_obs_normalization != "embedded"
                 && config.expected_obs_normalization != "none")
                || metadata("rmcs_obs_normalization").value_or("")
                       != config.expected_obs_normalization)
                throw std::invalid_argument(
                    "model observation normalization does not match the configured training "
                    "contract");
            if (metadata("rmcs_obs_mean") || metadata("rmcs_obs_std"))
                throw std::invalid_argument(
                    "explicit normalization cannot also declare external mean/std");
        }

        if (!std::isfinite(config.expected_policy_rate) || config.expected_policy_rate < 0.0)
            throw std::invalid_argument("expected policy rate must be finite and non-negative");
        const auto rate = metadata("policy_hz");
        if (!config.expected_observation_contract_sha256.empty() && !rate)
            throw std::invalid_argument(
                "explicit observation contract requires policy_hz metadata");
        if (rate) {
            const double model_rate = parse_number(*rate, "policy_hz");
            if (model_rate <= 0.0
                || (config.expected_policy_rate > 0.0
                    && std::abs(model_rate - config.expected_policy_rate) > 1e-6))
                throw std::invalid_argument("model policy_hz metadata does not match policy_rate");
        }
    }

    void load_preprocessing(const PolicyConfig& config) {
        if (config.normalization_from_metadata) {
            const auto mean = metadata("rmcs_obs_mean");
            const auto deviation = metadata("rmcs_obs_std");
            if (mean.has_value() != deviation.has_value())
                throw std::invalid_argument("rmcs_obs_mean and rmcs_obs_std must appear together");
            if (mean && deviation) {
                obs_mean = parse_numbers(*mean, "rmcs_obs_mean");
                obs_std = parse_numbers(*deviation, "rmcs_obs_std");
                if (obs_mean.size() != info.observation_size
                    || obs_std.size() != info.observation_size)
                    throw std::invalid_argument("normalization metadata does not match obs_size");
                if (std::ranges::any_of(obs_std, [](double value) { return value <= 0; }))
                    throw std::invalid_argument("rmcs_obs_std must be positive");
            }
            obs_clip = clip_metadata("rmcs_obs_clip");
            action_clip = clip_metadata("rmcs_action_clip");
        }
        if (config.obs_clip)
            obs_clip = config.obs_clip;
        if (config.action_clip)
            action_clip = config.action_clip;
        for (const auto clip : {obs_clip, action_clip})
            if (clip && (!std::isfinite(*clip) || *clip < 0.0))
                throw std::invalid_argument("clip parameter must be finite and non-negative");
    }

    std::optional<double> clip_metadata(const char* key) const {
        const auto text = metadata(key);
        if (!text || text->empty())
            return std::nullopt;
        const double value = parse_number(*text, key);
        if (value <= 0.0)
            throw std::invalid_argument(std::string{key} + " must be positive");
        return value;
    }

    std::expected<void, std::string>
        run(std::span<const double> observation, std::span<double> action) {
        if (observation.size() != info.observation_size || action.size() != info.action_size)
            return std::unexpected{"observation/action dimensions do not match model"};

        for (std::size_t i = 0; i < observation.size(); ++i) {
            double value = observation[i];
            if (!obs_mean.empty())
                value = (value - obs_mean[i]) / obs_std[i];
            if (obs_clip)
                value = std::clamp(value, -*obs_clip, *obs_clip);
            observation_buffer[i] = static_cast<float>(value);
            if (!std::isfinite(observation_buffer[i]))
                return std::unexpected{"non-finite observation at index " + std::to_string(i)};
        }

        try {
            runtime.run(observation_buffer, action_buffer);
        } catch (const std::exception& exception) {
            return std::unexpected{exception.what()};
        }
        if (!all_finite(action_buffer))
            return std::unexpected{"policy output contains non-finite values"};

        for (std::size_t i = 0; i < action.size(); ++i) {
            const double value = action_buffer[i];
            action[i] = action_clip ? std::clamp(value, -*action_clip, *action_clip) : value;
        }
        return {};
    }

    OnnxRuntime runtime;
    Info info;

    std::vector<double> obs_mean;
    std::vector<double> obs_std;
    std::optional<double> obs_clip;
    std::optional<double> action_clip;

    std::vector<float> observation_buffer;
    std::vector<float> action_buffer;
};

PolicyModel::PolicyModel(const PolicyConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

PolicyModel::~PolicyModel() = default;

const PolicyModel::Info& PolicyModel::info() const { return impl_->info; }

std::expected<void, std::string>
    PolicyModel::run(std::span<const double> observation, std::span<double> action) {
    return impl_->run(observation, action);
}

} // namespace rmcs_rl
