#include "onnx_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <stdexcept>

namespace rmcs_rl {
namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::size_t parse_positive(const std::optional<std::string>& value, const char* key) {
    if (!value || value->empty())
        return 0;
    std::size_t consumed = 0;
    std::size_t result = 0;
    try {
        result = std::stoull(*value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string{key} + " metadata must be a positive integer");
    }
    if (consumed != value->size() || result == 0)
        throw std::invalid_argument(std::string{key} + " metadata must be a positive integer");
    return result;
}

bool batch_dimension_ok(std::int64_t dimension) {
    // ONNX uses -1 for a dynamic dimension. Symbolic dimensions are returned
    // as -1 by the C++ API, so this also covers named batch dimensions.
    return dimension == 1 || dimension == -1;
}

} // namespace

OnnxRuntime::OnnxRuntime(const Config& config)
    : input_name_(config.input_name)
    , output_name_(config.output_name) {
    if (config.model_path.empty())
        throw std::invalid_argument("model path must not be empty");
    const auto requested_type = lower(config.model_type);
    if (requested_type != "auto" && requested_type != "mlp" && requested_type != "transformer"
        && requested_type != "generic")
        throw std::invalid_argument(
            "model_type must be auto, mlp, transformer, or generic (got '" + config.model_type
            + "')");

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    session_ = Ort::Session{env_, config.model_path.c_str(), options};

    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
        throw std::invalid_argument(
            "model must have exactly one runtime input and one output; multi-input models need a future adapter");
    const auto actual_input = session_.GetInputNameAllocated(0, allocator_);
    const auto actual_output = session_.GetOutputNameAllocated(0, allocator_);
    if (actual_input.get() != input_name_ || actual_output.get() != output_name_)
        throw std::invalid_argument(
            "tensor names must be '" + input_name_ + "' / '" + output_name_ + "', got '"
            + actual_input.get() + "' / '" + actual_output.get() + "'");

    // Keep TypeInfo alive while reading the TensorTypeAndShapeInfo. The
    // latter is a view into TypeInfo in ONNX Runtime; chaining a temporary
    // here can read freed type metadata on some runtime builds.
    const auto input_type = session_.GetInputTypeInfo(0);
    const auto output_type = session_.GetOutputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::invalid_argument("tensor element type must be float32");

    const auto raw_input_shape = input_info.GetShape();
    const auto raw_output_shape = output_info.GetShape();
    if (raw_input_shape.empty() || raw_input_shape.size() > 3)
        throw std::invalid_argument(
            "supported policy input ranks are 1 ([N]), 2 ([batch,N]), and 3 ([batch,sequence,feature])");
    if (raw_input_shape.size() >= 2 && !batch_dimension_ok(raw_input_shape[0]))
        throw std::invalid_argument("input batch dimension must be 1 or dynamic");
    if (raw_output_shape.empty() || raw_output_shape.size() > 3)
        throw std::invalid_argument("supported policy output ranks are 1, 2, or 3");
    if (raw_output_shape.size() >= 2 && !batch_dimension_ok(raw_output_shape[0]))
        throw std::invalid_argument("output batch dimension must be 1 or dynamic");

    const auto metadata_type = lower(metadata("rmcs_model_type").value_or(""));
    if (!metadata_type.empty() && metadata_type != "mlp" && metadata_type != "transformer"
        && metadata_type != "generic")
        throw std::invalid_argument("rmcs_model_type metadata must be mlp, transformer, or generic");
    if (requested_type != "auto" && requested_type != "generic" && !metadata_type.empty()
        && requested_type != metadata_type)
        throw std::invalid_argument(
            "model_type parameter '" + requested_type + "' disagrees with rmcs_model_type='"
            + metadata_type + "'");

    std::string resolved_type = requested_type;
    if (resolved_type == "auto")
        resolved_type = metadata_type;
    if (resolved_type.empty()) {
        if (raw_input_shape.size() == 2)
            resolved_type = "mlp";
        else if (raw_input_shape.size() == 3)
            resolved_type = "transformer";
        else
            resolved_type = "generic";
    }
    if (resolved_type == "mlp" && raw_input_shape.size() != 2 && raw_input_shape.size() != 3)
        throw std::invalid_argument(
            "model_type=mlp requires input shape [batch,obs] or [batch,history,feature]");
    if (resolved_type == "transformer" && raw_input_shape.size() != 3)
        throw std::invalid_argument(
            "model_type=transformer requires input shape [batch,sequence,feature]");
    info_.model_type = resolved_type;
    info_.input_rank = raw_input_shape.size();
    info_.output_rank = raw_output_shape.size();

    const auto metadata_sequence = parse_positive(metadata("rmcs_history_length"), "rmcs_history_length");
    const auto metadata_feature = parse_positive(metadata("rmcs_obs_frame_size"), "rmcs_obs_frame_size");
    std::size_t sequence = config.sequence_length != 0 ? config.sequence_length : metadata_sequence;
    std::size_t feature = config.feature_size != 0 ? config.feature_size : metadata_feature;
    const auto expected_observation = config.observation_size;

    input_shape_.resize(raw_input_shape.size());
    if (raw_input_shape.size() == 1) {
        const auto declared = raw_input_shape[0] > 0 ? static_cast<std::size_t>(raw_input_shape[0]) : 0;
        const auto resolved = declared != 0 ? declared : expected_observation;
        if (resolved == 0)
            throw std::invalid_argument("dynamic rank-1 input needs observation_size");
        if (declared != 0 && expected_observation != 0 && declared != expected_observation)
            throw std::invalid_argument("model input size does not match observation_size");
        input_shape_[0] = static_cast<std::int64_t>(resolved);
        info_.feature_size = resolved;
        info_.sequence_length = 1;
    } else if (raw_input_shape.size() == 2) {
        const auto declared = raw_input_shape[1] > 0 ? static_cast<std::size_t>(raw_input_shape[1]) : 0;
        const auto resolved = declared != 0 ? declared : expected_observation;
        if (resolved == 0)
            throw std::invalid_argument("dynamic rank-2 input needs observation_size");
        if (declared != 0 && expected_observation != 0 && declared != expected_observation)
            throw std::invalid_argument("model input feature size does not match observation_size");
        input_shape_ = {1, static_cast<std::int64_t>(resolved)};
        info_.feature_size = resolved;
        info_.sequence_length = 1;
    } else {
        const auto declared_sequence = raw_input_shape[1] > 0
                                            ? static_cast<std::size_t>(raw_input_shape[1])
                                            : 0;
        const auto declared_feature = raw_input_shape[2] > 0
                                          ? static_cast<std::size_t>(raw_input_shape[2])
                                          : 0;
        if (sequence == 0)
            sequence = declared_sequence;
        if (feature == 0)
            feature = declared_feature;
        if (sequence == 0 && feature != 0 && expected_observation != 0
            && expected_observation % feature == 0)
            sequence = expected_observation / feature;
        if (feature == 0 && sequence != 0 && expected_observation != 0
            && expected_observation % sequence == 0)
            feature = expected_observation / sequence;
        if (sequence == 0 || feature == 0)
            throw std::invalid_argument(
                "rank-3 input needs concrete sequence/feature dimensions or sequence_length/feature_size");
        if ((declared_sequence != 0 && declared_sequence != sequence)
            || (declared_feature != 0 && declared_feature != feature))
            throw std::invalid_argument("configured sequence/feature dimensions do not match model");
        if (expected_observation != 0 && sequence * feature != expected_observation)
            throw std::invalid_argument("sequence_length * feature_size does not match observation_size");
        input_shape_ = {1, static_cast<std::int64_t>(sequence), static_cast<std::int64_t>(feature)};
        info_.sequence_length = sequence;
        info_.feature_size = feature;
    }
    input_buffer_.resize(
        std::accumulate(input_shape_.begin(), input_shape_.end(), std::size_t{1}, [](auto value, auto dim) {
            return value * static_cast<std::size_t>(dim);
        }));

    std::size_t output_size = 1;
    bool output_dynamic = false;
    for (std::size_t index = raw_output_shape.size() >= 2 ? 1 : 0; index < raw_output_shape.size(); ++index) {
        if (raw_output_shape[index] <= 0)
            output_dynamic = true;
        else
            output_size *= static_cast<std::size_t>(raw_output_shape[index]);
    }
    if (output_dynamic) {
        if (config.action_size == 0)
            throw std::invalid_argument("dynamic output needs action_size");
        output_size = config.action_size;
    } else if (config.action_size != 0 && output_size != config.action_size) {
        throw std::invalid_argument("model output size does not match action_size");
    }
    if (output_size == 0)
        throw std::invalid_argument("model output must contain at least one action");
    output_size_ = output_size;
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
