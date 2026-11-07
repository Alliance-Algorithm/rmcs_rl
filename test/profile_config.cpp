#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "policy/policy_model.hpp"
#include <rmcs_rl/rl_bridge/config.hpp>
#include <rmcs_rl/rl_bridge/interface_binding.hpp>
#include <rmcs_rl/rl_bridge/utility.hpp>

namespace {

class ProfileConfig : public testing::Test {
protected:
    static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
    static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST_F(ProfileConfig, InstalledVehicleProfilesHaveConsistentLayouts) {
    for (const auto& [file, size, actions] :
         std::vector<std::tuple<std::string, std::size_t, std::size_t>>{
             {"deformable-infantry-omni-rl.yaml", 256, 4},
             {"wheel-leg-infantry-wheelbipe.yaml", 35, 6},
             {"wheel-leg-infantry-wheelbipe-recovery.yaml", 35, 6},
             {"wheel-leg-infantry-prone-v4.yaml", 35, 6}}) {
        SCOPED_TRACE(file);
        const auto options =
            rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true).arguments(
                {"--ros-args", "--params-file", std::string{RMCS_PROFILE_DIRECTORY} + "/" + file});
        rclcpp::Node node{"rl_bridge", options};
        auto config = rmcs_rl::load_bridge_config(node);
        EXPECT_EQ(config.obs_size, size);
        EXPECT_EQ(config.action_size, actions);
        EXPECT_NO_THROW((rmcs_rl::PolicyModel{config.policy}));
        if (actions == 6) {
            const bool wheelbipe = file.find("wheelbipe") != std::string::npos;
            for (const auto& term : config.action_terms) {
                const bool wheel = term.output.ends_with("wheel");
                EXPECT_DOUBLE_EQ(term.scale, wheel ? 10.0 : wheelbipe ? .5 : .25);
                EXPECT_DOUBLE_EQ(
                    rmcs_rl::map_action(2.0, term), wheel       ? 20.0
                                                    : wheelbipe ? 1.0
                                                                : .5);
                if (wheel) {
                    EXPECT_DOUBLE_EQ(rmcs_rl::map_action(20.0, term), 90.0);
                }
            }
        }
        rmcs_rl::bind_observation_terms(
            config.obs_terms, config.joint_config,
            [](const auto&, const auto&, bool, const char*) { return std::size_t{0}; });
        for (const auto& term : config.obs_terms) {
            if (!term.sources.empty()) {
                EXPECT_EQ(term.joint_slots.size(), term.sources.size());
            }
        }
    }
}

} // namespace
