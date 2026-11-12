#include "policy/onnx_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace rmcs_rl {
namespace {

std::string lower(std::string value) {
    std::ranges::transform(value, value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool batch_dimension_ok(std::int64_t dimension) { return dimension == 1 || dimension == -1; }

std::size_t concrete_dimension(std::int64_t dimension) {
    return dimension > 0 ? static_cast<std::size_t>(dimension) : 0;
}

} // namespace

OnnxRuntime::OnnxRuntime(const PolicyConfig& config) {
    if (config.path.empty())
        throw std::invalid_argument("model path must not be empty");
    const auto model_type = lower(config.model_type);
    if (model_type != "auto" && model_type != "mlp" && model_type != "transformer"
        && model_type != "generic")
        throw std::invalid_argument(
            "model_type must be auto, mlp, transformer, or generic (got '" + config.model_type
            + "')");

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    session_ = Ort::Session{env_, config.path.c_str(), options};

    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
        throw std::invalid_argument("model must have exactly one input and one output");
    input_name_ = session_.GetInputNameAllocated(0, allocator_).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator_).get();

    const auto input_type = session_.GetInputTypeInfo(0);
    const auto output_type = session_.GetOutputTypeInfo(0);
    if (input_type.GetONNXType() != ONNX_TYPE_TENSOR
        || output_type.GetONNXType() != ONNX_TYPE_TENSOR)
        throw std::invalid_argument("obs and actions must be tensors");
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::invalid_argument("obs and actions element types must be float32");

    const auto input_shape = input_info.GetShape();
    const auto output_shape = output_info.GetShape();
    if (input_shape.empty() || input_shape.size() > 3)
        throw std::invalid_argument(
            "supported policy input ranks are 1 ([N]), 2 ([batch,N]), and 3 "
            "([batch,sequence,feature])");
    if (input_shape.size() >= 2 && !batch_dimension_ok(input_shape[0]))
        throw std::invalid_argument("input batch dimension must be 1 or dynamic");
    if (output_shape.empty() || output_shape.size() > 3)
        throw std::invalid_argument("supported policy output ranks are 1, 2, or 3");
    if (output_shape.size() >= 2 && !batch_dimension_ok(output_shape[0]))
        throw std::invalid_argument("output batch dimension must be 1 or dynamic");
    if (model_type == "mlp" && input_shape.size() != 2 && input_shape.size() != 3)
        throw std::invalid_argument(
            "model_type=mlp requires input shape [batch,obs] or [batch,history,feature]");
    if (model_type == "transformer" && input_shape.size() != 3)
        throw std::invalid_argument(
            "model_type=transformer requires input shape [batch,sequence,feature]");

    resolve_input_shape(input_shape, config);
    resolve_output_size(output_shape, config);
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
    std::ranges::copy(input, input_buffer_.begin());

    const auto tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_buffer_.data(), input_buffer_.size(), input_shape_.data(),
        input_shape_.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    const auto outputs =
        session_.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()
        || outputs[0].GetTensorTypeAndShapeInfo().GetElementType()
               != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != output_size())
        throw std::runtime_error("inference output does not match model dimensions");
    std::copy_n(outputs[0].GetTensorData<float>(), output_size(), output.begin());
}

void OnnxRuntime::resolve_input_shape(
    const std::vector<std::int64_t>& shape, const PolicyConfig& config) {
    const std::size_t expected = config.observation_size;
    if (shape.size() < 3) {
        const auto declared = concrete_dimension(shape.back());
        const auto resolved = declared != 0 ? declared : expected;
        if (resolved == 0)
            throw std::invalid_argument(
                "dynamic rank-" + std::to_string(shape.size()) + " input needs observation_size");
        if (declared != 0 && expected != 0 && declared != expected)
            throw std::invalid_argument(
                "model input '" + input_name_ + "' has " + std::to_string(declared)
                + " observations; vehicle configuration requires " + std::to_string(expected));
        if (shape.size() == 1)
            input_shape_ = {static_cast<std::int64_t>(resolved)};
        else
            input_shape_ = {1, static_cast<std::int64_t>(resolved)};
        input_buffer_.resize(resolved);
        return;
    }

    const auto declared_sequence = concrete_dimension(shape[1]);
    const auto declared_feature = concrete_dimension(shape[2]);
    auto sequence = config.sequence_length != 0 ? config.sequence_length : declared_sequence;
    auto feature = config.feature_size != 0 ? config.feature_size : declared_feature;
    if (sequence == 0 && feature != 0 && expected != 0 && expected % feature == 0)
        sequence = expected / feature;
    if (feature == 0 && sequence != 0 && expected != 0 && expected % sequence == 0)
        feature = expected / sequence;
    if (sequence == 0 || feature == 0)
        throw std::invalid_argument(
            "rank-3 input needs concrete sequence/feature dimensions or "
            "sequence_length/feature_size");
    if ((declared_sequence != 0 && declared_sequence != sequence)
        || (declared_feature != 0 && declared_feature != feature))
        throw std::invalid_argument(
            "model input dims (T=" + std::to_string(declared_sequence)
            + ", F=" + std::to_string(declared_feature) + ") do not match expected (T="
            + std::to_string(sequence) + ", F=" + std::to_string(feature)
            + ") from vehicle configuration (model sequence must equal bridge history_length)");
    if (expected != 0 && sequence * feature != expected)
        throw std::invalid_argument(
            "sequence_length * feature_size does not match observation_size");
    input_shape_ = {1, static_cast<std::int64_t>(sequence), static_cast<std::int64_t>(feature)};
    input_buffer_.resize(sequence * feature);
}

void OnnxRuntime::resolve_output_size(
    const std::vector<std::int64_t>& shape, const PolicyConfig& config) {
    std::size_t size = 1;
    bool dynamic = false;
    for (std::size_t index = shape.size() >= 2 ? 1 : 0; index < shape.size(); ++index) {
        if (shape[index] <= 0)
            dynamic = true;
        else
            size *= static_cast<std::size_t>(shape[index]);
    }
    if (dynamic) {
        if (config.action_size == 0)
            throw std::invalid_argument("dynamic output needs action_size");
        size = config.action_size;
    } else if (config.action_size != 0 && size != config.action_size) {
        throw std::invalid_argument(
            "model output '" + output_name_ + "' has " + std::to_string(size)
            + " actions; vehicle configuration requires " + std::to_string(config.action_size));
    }
    if (size == 0)
        throw std::invalid_argument("model output must contain at least one action");
    output_size_ = size;
}

} // namespace rmcs_rl
