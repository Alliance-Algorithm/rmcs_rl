#pragma once

#include <array>
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
    OnnxRuntime(
        const std::string& model_path, const std::string& input_name,
        const std::string& output_name);

    [[nodiscard]] std::size_t input_size() const { return input_buffer_.size(); }
    [[nodiscard]] std::size_t output_size() const { return output_size_; }
    [[nodiscard]] std::optional<std::string> metadata(const char* key) const;

    void run(std::span<const float> input, std::span<float> output);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "rmcs_rl"};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Session session_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::array<std::int64_t, 2> input_shape_{};
    std::vector<float> input_buffer_;
    std::size_t output_size_ = 0;
};

} // namespace rmcs_rl
