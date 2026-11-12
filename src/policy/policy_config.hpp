#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>

namespace rmcs_rl {

struct PolicyConfig {
    std::string path;
    std::string model_type = "auto";

    std::size_t sequence_length = 0;
    std::size_t feature_size = 0;
    std::size_t observation_size = 0;
    std::size_t action_size = 0;

    double expected_policy_rate = 0.0;
    std::string expected_model_spec_sha256;
    std::string expected_observation_contract_sha256;
    std::string expected_observation_version;
    std::string expected_obs_normalization;
    std::map<std::string, std::string> expected_metadata;

    bool normalization_from_metadata = true;
    std::optional<double> obs_clip;
    std::optional<double> action_clip;
};

} // namespace rmcs_rl
