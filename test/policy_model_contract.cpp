// Loads only policy sessions; no executor, vehicle component or transport.
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <gtest/gtest.h>

#include "policy/policy_model.hpp"

namespace {
rmcs_rl::PolicyConfig configuration() {
    rmcs_rl::PolicyConfig config;
    config.path = ament_index_cpp::get_package_share_directory("rmcs_rl")
                + "/models/wheel_leg/flat_directional200_1346_preview_20261020.onnx";
    config.model_type = "wheel_leg_v7";
    config.expected_model_spec_sha256 =
        "0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997";
    config.expected_leg_winding_version = "paired_branch107_v1";
    config.expected_policy_rate = 200.0;
    config.sequence_length = 1;
    config.feature_size = config.observation_size = 35;
    config.action_size = 6;
    return config;
}

TEST(WheelLegPolicyContract, DirectionalActorRunsWithItsTrainingContract) {
    rmcs_rl::PolicyModel model{configuration()};
    std::array<double, 35> observation{};
    observation[3] = .305 * 5;
    observation[9] = -1;
    observation[28] = 1;
    std::array<double, 6> action{};
    std::string error;
    ASSERT_TRUE(model.run(observation, action, error)) << error;
    for (double value : action)
        EXPECT_TRUE(std::isfinite(value));
}

TEST(WheelLegPolicyContract, RejectsIncorrectPolicyRate) {
    auto config = configuration();
    config.expected_policy_rate = 50.0;
    EXPECT_THROW(rmcs_rl::PolicyModel{config}, std::invalid_argument);
}

TEST(WheelLegPolicyContract, RejectsDifferentMechanicalModel) {
    auto config = configuration();
    config.expected_model_spec_sha256 = std::string(64, '0');
    EXPECT_THROW(rmcs_rl::PolicyModel{config}, std::invalid_argument);
}

TEST(WheelLegPolicyContract, RejectsDifferentTargetWinding) {
    auto config = configuration();
    config.expected_leg_winding_version = "independent_nearest_v1";
    EXPECT_THROW(rmcs_rl::PolicyModel{config}, std::invalid_argument);
}

TEST(WheelLegPolicyContract, RejectsDifferentObservationHistory) {
    auto config = configuration();
    config.sequence_length = 2;
    config.observation_size = 70;
    EXPECT_THROW(rmcs_rl::PolicyModel{config}, std::invalid_argument);
}
} // namespace
