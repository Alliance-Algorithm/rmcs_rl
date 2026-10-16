#include "onnx_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

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

struct LayoutDimensions {
    std::size_t history = 0;
    std::size_t feature = 0;
    std::size_t observation = 0;
};

std::optional<LayoutDimensions> layout_dimensions(const std::optional<std::string>& value) {
    if (!value || value->empty())
        return std::nullopt;
    constexpr std::string_view prefix = "v3-history=";
    const std::string_view signature = *value;
    if (!signature.starts_with(prefix))
        return std::nullopt;

    const auto header_end = signature.find('|', prefix.size());
    if (header_end == std::string_view::npos)
        throw std::invalid_argument("rmcs_obs_layout is missing observation entries");
    const auto history_text = signature.substr(prefix.size(), header_end - prefix.size());
    LayoutDimensions result;
    const auto [history_end, history_error] = std::from_chars(
        history_text.data(), history_text.data() + history_text.size(), result.history);
    if (history_error != std::errc{}
        || history_end != history_text.data() + history_text.size() || result.history == 0)
        throw std::invalid_argument("rmcs_obs_layout history must be a positive integer");

    std::size_t begin = header_end + 1;
    while (begin < signature.size()) {
        const auto end = signature.find('|', begin);
        const auto entry = signature.substr(
            begin, end == std::string_view::npos ? std::string_view::npos : end - begin);
        const auto at = entry.rfind('@');
        if (at == std::string_view::npos || at + 1 == entry.size())
            throw std::invalid_argument("rmcs_obs_layout entry is missing @dim");
        std::size_t dimension = 0;
        const auto dim_text = entry.substr(at + 1);
        const auto [dim_end, dim_error] = std::from_chars(
            dim_text.data(), dim_text.data() + dim_text.size(), dimension);
        if (dim_error != std::errc{} || dim_end != dim_text.data() + dim_text.size()
            || dimension == 0)
            throw std::invalid_argument("rmcs_obs_layout entry dimension must be positive");
        if (dimension > std::numeric_limits<std::size_t>::max() - result.feature)
            throw std::invalid_argument("rmcs_obs_layout dimensions overflow");
        result.feature += dimension;
        if (end == std::string_view::npos)
            break;
        begin = end + 1;
    }
    if (result.feature == 0 || result.history > std::numeric_limits<std::size_t>::max() / result.feature)
        throw std::invalid_argument("rmcs_obs_layout dimensions overflow");
    result.observation = result.history * result.feature;
    return result;
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

    if (session_.GetOutputCount() != 1)
        throw std::invalid_argument("model must have exactly one output");
    const std::size_t input_count = session_.GetInputCount();
    if (input_count == 0)
        throw std::invalid_argument("model must have at least one input");

    input_names_.reserve(input_count);
    std::size_t primary_index = input_count; // sentinel: not found
    for (std::size_t index = 0; index < input_count; ++index) {
        const auto actual_input = session_.GetInputNameAllocated(index, allocator_);
        input_names_.emplace_back(actual_input.get());
        if (input_names_.back() == input_name_)
            primary_index = index;
    }
    if (primary_index == input_count) {
        std::string listing;
        for (std::size_t index = 0; index < input_count; ++index)
            listing += (index == 0 ? "" : ", ") + input_names_[index];
        throw std::invalid_argument(
            "primary input '" + input_name_ + "' not found; model inputs: [" + listing + "]");
    }
    const auto actual_output = session_.GetOutputNameAllocated(0, allocator_);
    if (actual_output.get() != output_name_)
        throw std::invalid_argument(
            "output tensor must be named '" + output_name_ + "', got '" + actual_output.get()
            + "'");
    // Keep TypeInfo alive while reading the TensorTypeAndShapeInfo. The
    // latter is a view into TypeInfo in ONNX Runtime; chaining a temporary
    // here can read freed type metadata on some runtime builds.
    const auto input_type = session_.GetInputTypeInfo(primary_index);
    const auto output_type = session_.GetOutputTypeInfo(0);
    if (input_type.GetONNXType() != ONNX_TYPE_TENSOR || output_type.GetONNXType() != ONNX_TYPE_TENSOR)
        throw std::invalid_argument("obs and actions must be tensors");
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::invalid_argument("obs and actions element types must be float32");

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
    const auto layout = layout_dimensions(metadata("rmcs_obs_layout"));
    if (raw_input_shape.size() == 3 && !layout && (config.sequence_length == 0 || config.feature_size == 0))
        throw std::invalid_argument(
            "rank-3 model without metadata requires sequence_length and feature_size");
    if (layout && metadata_sequence != 0 && metadata_sequence != layout->history)
        throw std::invalid_argument("rmcs_history_length does not match rmcs_obs_layout");
    if (layout && metadata_feature != 0 && metadata_feature != layout->feature)
        throw std::invalid_argument("rmcs_obs_frame_size does not match rmcs_obs_layout");
    if (layout && config.sequence_length != 0 && config.sequence_length != layout->history)
        throw std::invalid_argument("sequence_length does not match rmcs_obs_layout");
    if (layout && config.feature_size != 0 && config.feature_size != layout->feature)
        throw std::invalid_argument("feature_size does not match rmcs_obs_layout");
    std::size_t sequence = config.sequence_length != 0
                               ? config.sequence_length
                               : metadata_sequence != 0 ? metadata_sequence
                                                        : layout ? layout->history : 0;
    std::size_t feature = config.feature_size != 0
                              ? config.feature_size
                              : metadata_feature != 0 ? metadata_feature
                                                      : layout ? layout->feature : 0;
    std::size_t expected_observation = config.observation_size;
    if (expected_observation == 0 && layout)
        expected_observation = layout->observation;
    if (layout && expected_observation != layout->observation)
        throw std::invalid_argument("observation_size does not match rmcs_obs_layout dimensions");

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
            throw std::invalid_argument(
                "model input dims (T=" + std::to_string(declared_sequence)
                + ", F=" + std::to_string(declared_feature) + ") do not match expected (T="
                + std::to_string(sequence) + ", F=" + std::to_string(feature)
                + ") from sequence_length/feature_size or rmcs_history_length/rmcs_obs_frame_size "
                  "(model sequence must equal bridge history_length)");
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

    // Extra (non-obs) inputs: every one needs a constant from extra_inputs.*,
    // and every declared constant must match a model input.
    auto remaining = config.extra_input_values;
    constant_slot_.assign(input_count, -1);
    for (std::size_t index = 0; index < input_count; ++index) {
        if (index == primary_index)
            continue;
        const auto found = remaining.find(input_names_[index]);
        if (found == remaining.end())
            throw std::invalid_argument(
                "model input '" + input_names_[index]
                + "' has no constant value; declare it under policy_server extra_inputs."
                   "<name> in the vehicle YAML");
        const auto source = std::move(found->second);
        remaining.erase(found);

        const auto extra_type = session_.GetInputTypeInfo(index);
        if (extra_type.GetONNXType() != ONNX_TYPE_TENSOR)
            throw std::invalid_argument(
                "extra input '" + input_names_[index] + "' must be a tensor");
        const auto extra_info = extra_type.GetTensorTypeAndShapeInfo();
        const auto extra_element_type = extra_info.GetElementType();
        if (extra_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && extra_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            && extra_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
            throw std::invalid_argument(
                "extra input '" + input_names_[index] + "' element type must be float32, int64, or bool");
        const auto raw_shape = extra_info.GetShape();
        if (raw_shape.empty() || raw_shape.size() > 3)
            throw std::invalid_argument(
                "extra input '" + input_names_[index] + "' rank must be 1, 2, or 3");

        ConstantInput constant;
        constant.name = input_names_[index];
        constant.type = extra_element_type;
        std::size_t elements = 1;
        std::size_t first_dim = 0;
        if (raw_shape.size() >= 2) {
            if (!batch_dimension_ok(raw_shape[0]))
                throw std::invalid_argument(
                    "extra input '" + constant.name + "' batch dimension must be 1 or dynamic");
            constant.shape.push_back(1);
            first_dim = 1;
        }
        for (std::size_t dim = first_dim; dim < raw_shape.size(); ++dim) {
            if (raw_shape[dim] <= 0)
                throw std::invalid_argument(
                    "extra input '" + constant.name
                    + "' non-batch dimensions must be concrete (dynamic dim "
                    + std::to_string(dim) + ")");
            constant.shape.push_back(raw_shape[dim]);
            elements *= static_cast<std::size_t>(raw_shape[dim]);
        }
        if (std::visit([](const auto& values) { return values.size(); }, source) != elements)
            throw std::invalid_argument(
                "extra input '" + constant.name + "' needs " + std::to_string(elements)
                + " values, got "
                + std::to_string(std::visit([](const auto& values) { return values.size(); }, source)));
        constant.element_count = elements;
        if (extra_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            constant.float_values.reserve(elements);
            if (const auto* floats = std::get_if<std::vector<double>>(&source)) {
                for (const double value : *floats) {
                    const float converted = static_cast<float>(value);
                    if (!std::isfinite(converted))
                        throw std::invalid_argument(
                            "extra input '" + constant.name + "' contains a non-finite or out-of-range float32 value");
                    constant.float_values.push_back(converted);
                }
            } else if (const auto* integers = std::get_if<std::vector<std::int64_t>>(&source)) {
                for (const auto value : *integers) {
                    const float converted = static_cast<float>(value);
                    if (!std::isfinite(converted))
                        throw std::invalid_argument(
                            "extra input '" + constant.name + "' contains an out-of-range float32 value");
                    constant.float_values.push_back(converted);
                }
            } else {
                throw std::invalid_argument(
                    "extra input '" + constant.name + "' needs a float or integer list");
            }
        } else if (extra_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            if (const auto* integers = std::get_if<std::vector<std::int64_t>>(&source)) {
                constant.int64_values = *integers;
            } else if (const auto* floats = std::get_if<std::vector<double>>(&source)) {
                constant.int64_values.reserve(elements);
                for (const double value : *floats) {
                    const auto promoted = static_cast<long double>(value);
                    if (!std::isfinite(value) || std::trunc(value) != value
                        || promoted < static_cast<long double>(std::numeric_limits<std::int64_t>::min())
                        || promoted > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
                        throw std::invalid_argument(
                            "extra input '" + constant.name + "' contains a non-integral or out-of-range int64 value");
                    constant.int64_values.push_back(static_cast<std::int64_t>(value));
                }
            } else {
                throw std::invalid_argument(
                    "extra input '" + constant.name + "' needs an integer or integral float list for int64 tensor");
            }
        } else {
            const auto* booleans = std::get_if<std::vector<bool>>(&source);
            if (!booleans)
                throw std::invalid_argument(
                    "extra input '" + constant.name + "' needs a bool list for bool tensor");
            constant.bool_values = std::make_unique<bool[]>(elements);
            for (std::size_t value_index = 0; value_index < elements; ++value_index)
                constant.bool_values[value_index] = (*booleans)[value_index];
        }
        constant_slot_[index] = static_cast<std::int64_t>(constants_.size());
        constants_.push_back(std::move(constant));
    }
    if (!remaining.empty()) {
        std::string listing;
        for (const auto& [name, values] : remaining) {
            static_cast<void>(values);
            listing += (listing.empty() ? "" : ", ") + name;
        }
        throw std::invalid_argument(
            "extra_inputs entry(s) [" + listing + "] have no matching model input");
    }

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

    const std::size_t input_count = input_names_.size();
    std::vector<const char*> input_names;
    input_names.reserve(input_count);
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(input_count);
    for (std::size_t index = 0; index < input_count; ++index) {
        input_names.push_back(input_names_[index].c_str());
        if (constant_slot_[index] < 0) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, input_buffer_.data(), input_buffer_.size(), input_shape_.data(),
                input_shape_.size()));
        } else {
            auto& constant = constants_[static_cast<std::size_t>(constant_slot_[index])];
            if (constant.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info_, constant.float_values.data(), constant.element_count,
                    constant.shape.data(), constant.shape.size()));
            } else if (constant.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                input_tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(
                    memory_info_, constant.int64_values.data(), constant.element_count,
                    constant.shape.data(), constant.shape.size()));
            } else {
                input_tensors.push_back(Ort::Value::CreateTensor<bool>(
                    memory_info_, constant.bool_values.get(), constant.element_count,
                    constant.shape.data(), constant.shape.size()));
            }
        }
    }
    const char* output_names[] = {output_name_.c_str()};
    const auto outputs = session_.Run(
        Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_count,
        output_names, 1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()
        || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != output_size())
        throw std::runtime_error("inference output does not match model dimensions");
    std::copy_n(outputs[0].GetTensorData<float>(), output_size(), output.begin());
}

} // namespace rmcs_rl
