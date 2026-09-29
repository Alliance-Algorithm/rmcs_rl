#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace rmcs_rl {

class OnnxRuntime {
public:
    struct Config {
        std::string model_path;
        std::string input_name = "obs";
        std::string output_name = "actions";
        std::string model_type = "auto";
        std::size_t sequence_length = 0;
        std::size_t feature_size = 0;
        std::size_t observation_size = 0;
        std::size_t action_size = 0;
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
        : OnnxRuntime(Config{model_path, input_name, output_name}) {}

    [[nodiscard]] std::size_t input_size() const { return input_buffer_.size(); }
    [[nodiscard]] std::size_t output_size() const { return output_size_; }
    [[nodiscard]] const Info& info() const { return info_; }
    [[nodiscard]] std::optional<std::string> metadata(const char* key) const;

    void run(std::span<const float> input, std::span<float> output);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "rmcs_rl"};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Session session_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::vector<std::int64_t> input_shape_;
    std::vector<float> input_buffer_;
    std::size_t output_size_ = 0;
    Info info_;
};

} // namespace rmcs_rl
