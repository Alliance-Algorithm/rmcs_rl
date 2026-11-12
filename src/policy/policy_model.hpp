#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>

#include "policy/policy_config.hpp"

namespace rmcs_rl {

class PolicyModel {
public:
    struct Info {
        std::string path;
        std::string model_spec_sha256;
        std::string observation_contract_sha256;
        std::size_t observation_size = 0;
        std::size_t action_size = 0;
    };

    explicit PolicyModel(const PolicyConfig& config);

    PolicyModel(const PolicyModel&) = delete;
    PolicyModel& operator=(const PolicyModel&) = delete;
    PolicyModel(PolicyModel&&) = delete;
    PolicyModel& operator=(PolicyModel&&) = delete;

    ~PolicyModel();

    const Info& info() const;

    std::expected<void, std::string>
        run(std::span<const double> observation, std::span<double> action);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rmcs_rl
