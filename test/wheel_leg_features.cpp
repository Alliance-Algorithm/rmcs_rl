#include <array>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_rl/rl_bridge/config.hpp>
#include <rmcs_rl/rl_bridge/observation.hpp>
#include <rmcs_rl/rl_bridge/utility.hpp>

#include "controller/chassis/wheel_leg/policy/selection.hpp"

namespace {
using namespace rmcs_core::controller::chassis;

TEST(WheelLegFeatures, WheelbipePublishesMechanicalOrderBeforeYamlPermutation) {
    Wheelbipe adapter;
    auto position = kNominalPosition;
    JointVector velocity{1., 2., 3., 4., 5., 6.};
    JointVector previous{.1, .2, .3, .4, .5, .6};
    const auto features = adapter.sample(
        1, position, velocity, {1., 2., 3.}, {0., 0., -1.}, {.2, .3, .4}, .3, previous, true);
    ASSERT_TRUE(features);
    EXPECT_DOUBLE_EQ(features->height, .3);
    EXPECT_DOUBLE_EQ(features->gyro[2], 3.);
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(features->velocity[i], velocity[i]);
        EXPECT_DOUBLE_EQ(features->previous_action[i], previous[i]);
    }
    EXPECT_FLOAT_EQ(features->auxiliary[0], 1.F);
}

TEST(WheelLegFeatures, ProneOdometryRemainsFiniteAcrossResetAndStandingSamples) {
    ProneMotion adapter;
    const JointVector zero{};
    for (int session = 0; session < 2; ++session) {
        adapter.reset();
        for (std::uint64_t tick = 1; tick <= 20; ++tick) {
            const auto features = adapter.sample(
                tick, kNominalPosition, zero, {0., 0., 0.}, {0., 0., -1.}, {0., 0., 0.}, .305, zero,
                true);
            ASSERT_TRUE(features);
            EXPECT_DOUBLE_EQ(features->height, .305);
            for (const auto value : features->position)
                EXPECT_TRUE(std::isfinite(value));
        }
    }
}

TEST(WheelLegFeatures, PublicActionMappingMatchesLegacyPhysicalTargets) {
    std::mt19937 generator{91};
    std::uniform_real_distribution<double> distribution{-20., 20.};
    for (const auto* policy : {&kWheelbipe, &kProneMotion}) {
        for (int sample = 0; sample < 128; ++sample) {
            JointVector raw{}, decoded{};
            for (int i = 0; i < 6; ++i) {
                raw[i] = distribution(generator);
                rmcs_rl::ActionTerm term;
                term.scale = i < 4 ? policy->leg_scale : 10.;
                term.has_clip = i >= 4 || policy == &kProneMotion;
                term.clip_min = i < 4 ? -4. : -90.;
                term.clip_max = -term.clip_min;
                decoded[i] = rmcs_rl::map_action(raw[i], term);
            }
            const auto expected = action_target(*policy, raw, kNominalPosition);
            const auto actual =
                decoded_action_target(decoded, kNominalPosition, policy->winding_bounds);
            EXPECT_EQ(actual, expected);
        }
    }
}

TEST(WheelLegFeatures, RealYamlReproducesTrainingPreprocessingFromPhysicalInputs) {
    rclcpp::init(0, nullptr);
    for (const auto* profile :
         {"wheel-leg-infantry-wheelbipe.yaml", "wheel-leg-infantry-prone-v4.yaml"}) {
        const auto options =
            rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true).arguments(
                {"--ros-args", "--params-file",
                 std::string{RMCS_PROFILE_DIRECTORY} + "/" + profile});
        rclcpp::Node node{"rl_bridge", options};
        auto config = rmcs_rl::load_policy_io_config(node);
        EXPECT_DOUBLE_EQ(config.obs_terms[1].scale, 5.0);
        EXPECT_DOUBLE_EQ(config.obs_terms[2].scale, .5);
        EXPECT_DOUBLE_EQ(config.obs_terms[5].scale, .1);
        std::vector<rmcs_rl::Slot> slots(config.obs_terms.size());
        std::vector<std::vector<double>> sources(slots.size());
        double height = .305;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            config.obs_terms[i].slot = i;
            if (i == 1) {
                slots[i].binding = rmcs_rl::Binding::kDouble;
                slots[i].double_value =
                    std::make_unique<rmcs_executor::Component::InputInterface<double>>();
                slots[i].double_value->bind_directly(height);
            } else {
                sources[i].resize(config.obs_terms[i].source_size);
                slots[i].binding = rmcs_rl::Binding::kDoubleArray;
                slots[i].array_value = std::make_unique<
                    rmcs_executor::Component::InputInterface<std::vector<double>>>();
                slots[i].array_value->bind_directly(sources[i]);
            }
        }
        const auto baseline = [&] {
            std::vector<double> result;
            EXPECT_TRUE(rmcs_rl::build_observation(config.obs_terms, slots, {}, 35, result));
            return result;
        }();
        config.obs_terms[1].scale = 2.0;
        std::vector<double> changed;
        ASSERT_TRUE(rmcs_rl::build_observation(config.obs_terms, slots, {}, 35, changed));
        EXPECT_NE(baseline[3], changed[3]);
        config.obs_terms[1].scale = 5.0;
        std::mt19937 generator{73};
        std::uniform_real_distribution<double> distribution{-300., 300.};
        const bool wheelbipe = std::string{profile}.find("wheelbipe") != std::string::npos;
        for (int sample = 0; sample < 128; ++sample) {
            for (auto& source : sources)
                for (auto& value : source)
                    value = distribution(generator);
            height = distribution(generator);
            std::vector<double> actual;
            ASSERT_TRUE(rmcs_rl::build_observation(config.obs_terms, slots, {}, 35, actual));
            std::size_t cursor = 0;
            for (std::size_t group = 0; group < sources.size(); ++group) {
                const auto& term = config.obs_terms[group];
                for (std::size_t component = 0; component < term.dim; ++component) {
                    const auto index =
                        term.selection.empty() ? component : term.selection[component];
                    if (!term.float32) {
                        EXPECT_DOUBLE_EQ(actual[cursor++], sources[group][index]);
                        continue;
                    }
                    float expected =
                        static_cast<float>(group == 1 ? height : sources[group][index]);
                    if (wheelbipe) {
                        if (group == 1)
                            expected = std::clamp(expected, 0.F, 1.F);
                        if (group == 2)
                            expected = std::clamp(expected, -100.F, 100.F);
                        if (group == 5)
                            expected = std::clamp(expected, -200.F, 200.F);
                    }
                    if (group == 1)
                        expected *= 5.F;
                    if (group == 2)
                        expected *= .5F;
                    if (group == 5)
                        expected *= .1F;
                    if ((group != 4 && group != 7) || (wheelbipe && group == 4))
                        expected = std::clamp(expected, -100.F, 100.F);
                    EXPECT_DOUBLE_EQ(actual[cursor++], static_cast<double>(expected));
                }
            }
        }
    }
    rclcpp::shutdown();
}

} // namespace
