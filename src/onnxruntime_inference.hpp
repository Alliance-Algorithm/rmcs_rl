#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace rmcs::rl {
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

    /// 加载模型并校验合同。失败返回 false 并复位会话。
    bool load(const Config& config) {
        config_ = config;
        session_.reset();
        try {
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_options_.SetIntraOpNumThreads(1);
            session_options_.SetInterOpNumThreads(1);
            session_ = std::make_unique<Ort::Session>(
                env_, config_.model_path.c_str(), session_options_);

            if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
                return false;

            const auto actual_input = session_->GetInputNameAllocated(0, allocator_);
            const auto actual_output = session_->GetOutputNameAllocated(0, allocator_);
            if (actual_input.get() != config_.input_name
                || actual_output.get() != config_.output_name)
                return false;

            // TensorTypeAndShapeInfo 是 TypeInfo 的非拥有视图，先取 type info 再查询
            const auto input_type_info = session_->GetInputTypeInfo(0);
            const auto output_type_info = session_->GetOutputTypeInfo(0);
            const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
            const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
            if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                return false;

            const auto input_shape = input_info.GetShape();
            const auto output_shape = output_info.GetShape();
            if (input_shape.size() != 2 || output_shape.size() != 2
                || static_cast<std::size_t>(input_shape[1]) != config_.input_size
                || static_cast<std::size_t>(output_shape[1]) != config_.output_size)
                return false;

            input_shape_.assign(input_shape.begin(), input_shape.end());
            output_shape_.assign(output_shape.begin(), output_shape.end());
            memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            return true;
        } catch (const Ort::Exception&) {
            session_.reset();
            return false;
        }
    }

    [[nodiscard]] bool ready() const { return session_ != nullptr; }

    [[nodiscard]] const std::string& model_path() const { return config_.model_path; }
    [[nodiscard]] std::size_t input_size() const { return config_.input_size; }
    [[nodiscard]] std::size_t output_size() const { return config_.output_size; }

    /// 同步推理。input/output 为 float 序列，长度须 >= 合同大小。失败返回 false。
    bool run(std::span<const float> input, std::span<float> output) {
        if (!session_ || input.size() < config_.input_size || output.size() < config_.output_size)
            return false;
        try {
            std::vector<float> output_data(config_.output_size, 0.0F);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(input.data()), config_.input_size,
                input_shape_.data(), input_shape_.size());
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, output_data.data(), config_.output_size, output_shape_.data(),
                output_shape_.size());

            const char* input_names[] = {config_.input_name.c_str()};
            const char* output_names[] = {config_.output_name.c_str()};
            auto outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            if (outputs.size() != 1 || !outputs[0].IsTensor())
                return false;
            const float* data = outputs[0].GetTensorData<float>();
            std::copy(data, data + config_.output_size, output.begin());
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
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    Config config_;
};

} // namespace rmcs::rl
