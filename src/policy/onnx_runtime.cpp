#include "onnx_runtime.hpp"

#include <algorithm>
#include <stdexcept>

namespace rmcs_rl {

OnnxRuntime::OnnxRuntime(
    const std::string& model_path, const std::string& input_name, const std::string& output_name)
    : input_name_(input_name)
    , output_name_(output_name) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    session_ = Ort::Session{env_, model_path.c_str(), options};

    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
        throw std::invalid_argument("model must have exactly one input and one output");
    const auto actual_input = session_.GetInputNameAllocated(0, allocator_);
    const auto actual_output = session_.GetOutputNameAllocated(0, allocator_);
    if (actual_input.get() != input_name_ || actual_output.get() != output_name_)
        throw std::invalid_argument(
            "tensor names must be '" + input_name_ + "' / '" + output_name_ + "', got '"
            + actual_input.get() + "' / '" + actual_output.get() + "'");

    const auto input_type = session_.GetInputTypeInfo(0);
    const auto output_type = session_.GetOutputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::invalid_argument("tensor element type must be float32");

    const auto input_shape = input_info.GetShape();
    const auto output_shape = output_info.GetShape();
    if (input_shape.size() != 2 || output_shape.size() != 2)
        throw std::invalid_argument("tensor rank must be 2 ([batch, N])");
    if ((input_shape[0] != 1 && input_shape[0] != -1)
        || (output_shape[0] != 1 && output_shape[0] != -1))
        throw std::invalid_argument("batch dimension must be 1 or dynamic");
    if (input_shape[1] <= 0 || output_shape[1] <= 0)
        throw std::invalid_argument("feature dimensions must be concrete positive values");

    // The server evaluates one robot observation, even when the exported batch is dynamic.
    input_shape_ = {1, input_shape[1]};
    input_buffer_.resize(static_cast<std::size_t>(input_shape[1]));
    output_size_ = static_cast<std::size_t>(output_shape[1]);
}

std::optional<std::string> OnnxRuntime::metadata(const char* key) const {
    const auto model_metadata = session_.GetModelMetadata();
    const auto value = model_metadata.LookupCustomMetadataMapAllocated(key, allocator_);
    if (!value)
        return std::nullopt;
    return std::string{value.get()};
}

void OnnxRuntime::run(std::span<const float> input, std::span<float> output) {
    if (input.size() != input_size() || output.size() != output_size())
        throw std::invalid_argument("inference buffers do not match model dimensions");
    std::copy(input.begin(), input.end(), input_buffer_.begin());
    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_buffer_.data(), input_buffer_.size(), input_shape_.data(),
        input_shape_.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    const auto outputs =
        session_.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()
        || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != output_size())
        throw std::runtime_error("inference output does not match model dimensions");
    std::copy_n(outputs[0].GetTensorData<float>(), output_size(), output.begin());
}

} // namespace rmcs_rl
