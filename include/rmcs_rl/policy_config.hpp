#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace rmcs_rl {

using PolicyExtraInput =
    std::variant<std::vector<double>, std::vector<std::int64_t>, std::vector<bool>>;

struct PolicyConfig {
    std::string path;
    std::string input_name;
    std::string output_name;
    std::string model_type = "auto";
    std::string expected_model_spec_sha256;
    std::string expected_leg_winding_version;
    std::size_t sequence_length = 0;
    std::size_t feature_size = 0;
    std::size_t observation_size = 0;
    std::size_t action_size = 0;
    bool normalization_from_metadata = true;
    std::optional<double> obs_clip;
    std::optional<double> action_clip;
    std::map<std::string, PolicyExtraInput> extra_input_values;
};

} // namespace rmcs_rl
