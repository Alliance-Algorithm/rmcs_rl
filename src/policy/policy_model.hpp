#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "onnx_runtime.hpp"
#include <rmcs_rl/policy_config.hpp>

namespace rmcs_rl {

class PolicyModel {
public:
    using Config = PolicyConfig;

    struct Info {
        std::string path;
        std::size_t obs_size = 0;
        std::size_t action_size = 0;
        std::string model_type;
        std::size_t input_rank = 0;
        std::size_t output_rank = 0;
        std::size_t sequence_length = 1;
        std::size_t feature_size = 0;
    };

    explicit PolicyModel(const Config& config);

    [[nodiscard]] const Info& info() const { return info_; }
    bool run(std::span<const double> observation, std::span<double> action, std::string& error);

private:
    void load_normalization(const Config& config);

    OnnxRuntime runtime_;
    Info info_;
    std::vector<double> obs_mean_;
    std::vector<double> obs_std_;
    std::optional<double> obs_clip_;
    std::optional<double> action_clip_;
    std::vector<float> obs_buffer_;
    std::vector<float> action_buffer_;
};

} // namespace rmcs_rl
