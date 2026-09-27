#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace rmcs_rl {

class OnnxRuntimeInference {
public:
    struct Config {
        std::string model_path;
        std::string input_name = "obs";
        std::string output_name = "actions";
        std::size_t input_size = 0;
        std::size_t output_size = 0;
    };

    OnnxRuntimeInference() = default;

    OnnxRuntimeInference(const OnnxRuntimeInference&) = delete;
    OnnxRuntimeInference& operator=(const OnnxRuntimeInference&) = delete;

    bool load(const Config& config) {
        std::string ignored;
        return load(config, ignored);
    }

    bool load(const Config& config, std::string& error) {
        config_ = config;
        session_.reset();
        error.clear();
        try {
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_options_.SetIntraOpNumThreads(1);
            session_options_.SetInterOpNumThreads(1);
            session_ =
                std::make_unique<Ort::Session>(env_, config_.model_path.c_str(), session_options_);

            if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
                error = "model must have exactly 1 input and 1 output, got "
                      + std::to_string(session_->GetInputCount()) + " input(s) / "
                      + std::to_string(session_->GetOutputCount()) + " output(s)";
                session_.reset();
                return false;
            }

            const auto actual_input = session_->GetInputNameAllocated(0, allocator_);
            const auto actual_output = session_->GetOutputNameAllocated(0, allocator_);
            if (actual_input.get() != config_.input_name
                || actual_output.get() != config_.output_name) {
                error = "tensor names must be '" + config_.input_name + "' / '"
                      + config_.output_name + "', got '" + actual_input.get() + "' / '"
                      + actual_output.get() + "'";
                session_.reset();
                return false;
            }

            const auto input_type_info = session_->GetInputTypeInfo(0);
            const auto output_type_info = session_->GetOutputTypeInfo(0);
            const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
            const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
            if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                error = "tensor element type must be float32";
                session_.reset();
                return false;
            }

            const auto input_shape = input_info.GetShape();
            const auto output_shape = output_info.GetShape();
            if (input_shape.size() != 2 || output_shape.size() != 2) {
                error = "tensor rank must be 2 ([batch, N])";
                session_.reset();
                return false;
            }
            if ((input_shape[0] != 1 && input_shape[0] != -1)
                || (output_shape[0] != 1 && output_shape[0] != -1)) {
                error = "batch dimension must be 1 or dynamic";
                session_.reset();
                return false;
            }
            if (input_shape[1] <= 0 || output_shape[1] <= 0) {
                error = "feature dimensions must be concrete positive values";
                session_.reset();
                return false;
            }

            const auto model_input_size = static_cast<std::size_t>(input_shape[1]);
            const auto model_output_size = static_cast<std::size_t>(output_shape[1]);
            if (config_.input_size != 0 && config_.input_size != model_input_size) {
                error = "configured rl_obs_size=" + std::to_string(config_.input_size)
                      + " but model input shape is [1," + std::to_string(model_input_size) + "]";
                session_.reset();
                return false;
            }
            if (config_.output_size != 0 && config_.output_size != model_output_size) {
                error = "configured rl_action_size=" + std::to_string(config_.output_size)
                      + " but model output shape is [1," + std::to_string(model_output_size) + "]";
                session_.reset();
                return false;
            }

            config_.input_size = model_input_size;
            config_.output_size = model_output_size;
            // This deployment path runs one observation at a time. Normalize a
            // symbolic batch dimension to 1 for Ort::CreateTensor below.
            input_shape_ = {1, static_cast<std::int64_t>(model_input_size)};
            output_shape_ = {1, static_cast<std::int64_t>(model_output_size)};
            input_buffer_.assign(config_.input_size, 0.0F);
            output_buffer_.assign(config_.output_size, 0.0F);
            memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            return true;
        } catch (const Ort::Exception& exception) {
            error = std::string{"ONNX Runtime: "} + exception.what();
            session_.reset();
            return false;
        }
    }

    [[nodiscard]] bool ready() const { return session_ != nullptr; }

    [[nodiscard]] const std::string& model_path() const { return config_.model_path; }
    [[nodiscard]] std::size_t input_size() const { return config_.input_size; }
    [[nodiscard]] std::size_t output_size() const { return config_.output_size; }

    [[nodiscard]] std::optional<std::string> metadata(const std::string& key) const {
        if (!session_)
            return std::nullopt;
        try {
            auto model_metadata = session_->GetModelMetadata();
            auto value = model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator_);
            if (!value)
                return std::nullopt;
            return std::string(value.get());
        } catch (const Ort::Exception&) {
            return std::nullopt;
        }
    }

    bool run(std::span<const float> input, std::span<float> output) {
        if (!session_ || input.size() < config_.input_size || output.size() < config_.output_size)
            return false;
        try {
            std::copy(
                input.begin(), input.begin() + static_cast<std::ptrdiff_t>(config_.input_size),
                input_buffer_.begin());
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, input_buffer_.data(), config_.input_size, input_shape_.data(),
                input_shape_.size());
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, output_buffer_.data(), config_.output_size, output_shape_.data(),
                output_shape_.size());

            const char* input_names[] = {config_.input_name.c_str()};
            const char* output_names[] = {config_.output_name.c_str()};
            auto outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            if (outputs.size() != 1 || !outputs[0].IsTensor())
                return false;
            const float* data = outputs[0].GetTensorData<float>();
            std::copy(
                data, data + static_cast<std::ptrdiff_t>(config_.output_size), output.begin());
            return true;
        } catch (const Ort::Exception&) {
            return false;
        }
    }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "rmcs_rl"};
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_{nullptr};
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::int64_t> input_shape_;
    std::vector<std::int64_t> output_shape_;
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    Config config_;
};

} // namespace rmcs_rl
