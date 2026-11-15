#include "policy/policy_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rmcs_rl {

PolicyModel::PolicyModel(
    const std::string& path, std::size_t history_length, std::size_t frame_size,
    std::size_t action_size, const std::map<std::string, std::string>& expected_metadata) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    session_ = Ort::Session{env_, path.c_str(), options};
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
        throw std::invalid_argument("model shape does not match configuration");

    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
    const auto input_type = session_.GetInputTypeInfo(0);
    const auto output_type = session_.GetOutputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    const auto input_shape = input_info.GetShape();
    const auto output_shape = output_info.GetShape();
    const auto size = static_cast<std::int64_t>(history_length * frame_size);
    switch (input_shape.size()) {
    case 1: input_shape_ = {size}; break;
    case 2: input_shape_ = {1, size}; break;
    case 3:
        input_shape_ = {
            1, static_cast<std::int64_t>(history_length), static_cast<std::int64_t>(frame_size)};
        break;
    default: throw std::invalid_argument("model shape does not match configuration");
    }
    bool matches = input_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                && output_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                && !output_shape.empty() && output_shape.size() <= 3;
    for (std::size_t i = 0; i < input_shape.size(); ++i)
        matches &= input_shape[i] <= 0 || input_shape[i] == input_shape_[i];
    std::size_t output_size = 1;
    bool dynamic_output = false;
    for (std::size_t i = 0; i < output_shape.size(); ++i) {
        if (i == 0 && output_shape.size() > 1)
            matches &= output_shape[i] <= 0 || output_shape[i] == 1;
        else if (output_shape[i] <= 0)
            dynamic_output = true;
        else
            output_size *= static_cast<std::size_t>(output_shape[i]);
    }
    if (!matches || (!dynamic_output && output_size != action_size))
        throw std::invalid_argument("model shape does not match configuration");
    input_.resize(static_cast<std::size_t>(size));
    output_.resize(action_size);

    const auto model_metadata = session_.GetModelMetadata();
    for (const auto& key : model_metadata.GetCustomMetadataMapKeysAllocated(allocator))
        metadata_.emplace(
            key.get(), model_metadata.LookupCustomMetadataMapAllocated(key.get(), allocator).get());
    const auto metadata = [this](const std::string& key) -> std::optional<std::string> {
        const auto found = metadata_.find(key);
        return found == metadata_.end() ? std::nullopt : std::optional{found->second};
    };
    for (const auto& [key, value] : expected_metadata)
        if (metadata(key) != std::optional<std::string>{value})
            throw std::invalid_argument("model metadata mismatch: " + key);

    const auto numbers = [](const std::string& text) {
        std::vector<double> values;
        for (std::size_t begin = 0; begin < text.size();) {
            const auto end = text.find(',', begin);
            values.push_back(std::stod(text.substr(begin, end - begin)));
            if (end == std::string::npos)
                break;
            begin = end + 1;
        }
        return values;
    };
    if (const auto mean = metadata("rmcs_obs_mean"))
        obs_mean_ = numbers(*mean);
    if (const auto deviation = metadata("rmcs_obs_std"))
        obs_std_ = numbers(*deviation);
    if ((!obs_mean_.empty() || !obs_std_.empty())
        && (obs_mean_.size() != input_.size() || obs_std_.size() != input_.size()
            || !std::ranges::all_of(obs_mean_, [](double v) { return std::isfinite(v); })
            || !std::ranges::all_of(obs_std_, [](double v) { return std::isfinite(v) && v > 0.; })))
        throw std::invalid_argument("model normalization does not match configuration");

    const auto clip = [&metadata](const char* key) -> std::optional<double> {
        const auto text = metadata(key);
        if (!text || text->empty())
            return std::nullopt;
        const double value = std::stod(*text);
        if (!std::isfinite(value) || value <= 0.)
            throw std::invalid_argument(std::string{"invalid model clipping metadata: "} + key);
        return value;
    };
    obs_clip_ = clip("rmcs_obs_clip");
    action_clip_ = clip("rmcs_action_clip");
}

void PolicyModel::run(std::span<const double> observation, std::span<double> action) {
    for (std::size_t i = 0; i < input_.size(); ++i) {
        double value = observation[i];
        if (!obs_mean_.empty())
            value = (value - obs_mean_[i]) / obs_std_[i];
        if (obs_clip_)
            value = std::clamp(value, -*obs_clip_, *obs_clip_);
        input_[i] = static_cast<float>(value);
    }
    const auto tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_.data(), input_.size(), input_shape_.data(), input_shape_.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    const auto outputs =
        session_.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
    // A dynamic output dimension can only be checked after inference.
    if (outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != output_.size())
        throw std::runtime_error("model shape does not match configuration");
    std::copy_n(outputs[0].GetTensorData<float>(), output_.size(), output_.begin());
    for (std::size_t i = 0; i < output_.size(); ++i) {
        const double value = output_[i];
        action[i] = action_clip_ ? std::clamp(value, -*action_clip_, *action_clip_) : value;
    }
}

} // namespace rmcs_rl
