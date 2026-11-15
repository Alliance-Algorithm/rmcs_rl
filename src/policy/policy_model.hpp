#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace rmcs_rl {

class PolicyModel {
public:
    PolicyModel(
        const std::string& path, std::size_t history_length, std::size_t frame_size,
        std::size_t action_size, const std::map<std::string, std::string>& expected_metadata);

    void run(std::span<const double> observation, std::span<double> action);
    const std::map<std::string, std::string>& metadata() const { return metadata_; }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "rmcs_rl"};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string input_name_, output_name_;
    std::vector<std::int64_t> input_shape_;
    std::vector<float> input_, output_;
    std::vector<double> obs_mean_, obs_std_;
    std::optional<double> obs_clip_, action_clip_;
    std::map<std::string, std::string> metadata_;
};

} // namespace rmcs_rl
