#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "policy/policy_config.hpp"

namespace rmcs_rl {

class OnnxRuntime {
public:
    explicit OnnxRuntime(const PolicyConfig& config);

    [[nodiscard]] std::size_t input_size() const { return input_buffer_.size(); }
    [[nodiscard]] std::size_t output_size() const { return output_size_; }
    [[nodiscard]] std::optional<std::string> metadata(const char* key) const;

    void run(std::span<const float> input, std::span<float> output);

private:
    void resolve_input_shape(const std::vector<std::int64_t>& shape, const PolicyConfig& config);
    void resolve_output_size(const std::vector<std::int64_t>& shape, const PolicyConfig& config);

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "rmcs_rl"};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Session session_{nullptr};

    std::string input_name_;
    std::string output_name_;
    std::vector<std::int64_t> input_shape_;
    std::vector<float> input_buffer_;
    std::size_t output_size_ = 0;
};

} // namespace rmcs_rl
