#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace rmcs_rl {

class OnnxRuntime {
public:
    using ExtraInputValues = std::variant<
        std::vector<double>, std::vector<std::int64_t>, std::vector<bool>>;

    struct Config {
        std::string model_path;
        std::string input_name = "obs";
        std::string output_name = "actions";
        std::string model_type = "auto";
        std::size_t sequence_length = 0;
        std::size_t feature_size = 0;
        std::size_t observation_size = 0;
        std::size_t action_size = 0;
        // 额外输入张量的常量值（键 = 模型输入名，值 = 展平的元素），
        // 由 policy_server 的 extra_inputs.<name> YAML 参数提供。
        std::map<std::string, ExtraInputValues> extra_input_values;
    };

    struct Info {
        std::string model_type;
        std::size_t input_rank = 0;
        std::size_t output_rank = 0;
        std::size_t sequence_length = 1;
        std::size_t feature_size = 0;
    };

    explicit OnnxRuntime(const Config& config);

    // Kept for small external users of the old deployment API.
    OnnxRuntime(
        const std::string& model_path, const std::string& input_name,
        const std::string& output_name)
        : OnnxRuntime(Config{
              .model_path = model_path,
              .input_name = input_name,
              .output_name = output_name,
              .model_type = "auto",
              .sequence_length = 0,
              .feature_size = 0,
              .observation_size = 0,
              .action_size = 0,
              .extra_input_values = {}}) {}

    [[nodiscard]] std::size_t input_size() const { return input_buffer_.size(); }
    [[nodiscard]] std::size_t output_size() const { return output_size_; }
    [[nodiscard]] const Info& info() const { return info_; }
    [[nodiscard]] std::optional<std::string> metadata(const char* key) const;

    void run(std::span<const float> input, std::span<float> output);

private:
    struct ConstantInput {
        std::string name;
        std::vector<std::int64_t> shape;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        std::size_t element_count = 0;
        std::vector<float> float_values;
        std::vector<std::int64_t> int64_values;
        std::unique_ptr<bool[]> bool_values;
    };

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "rmcs_rl"};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Session session_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::vector<std::string> input_names_; // 会话输入顺序
    std::vector<ConstantInput> constants_;
    // 与 input_names_ 并行：主输入为 -1，其余为 constants_ 下标
    std::vector<std::int64_t> constant_slot_;
    std::vector<std::int64_t> input_shape_;
    std::vector<float> input_buffer_;
    std::size_t output_size_ = 0;
    Info info_;
};

} // namespace rmcs_rl
