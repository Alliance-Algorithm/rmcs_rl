#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "policy/policy_model.hpp"
#include "rl_bridge/config.hpp"
#include "rl_bridge/observation_builder.hpp"
#include "rl_bridge/policy_actions.hpp"

namespace rmcs_executor {
class Executor {
public:
    static double read(const Component& component, const std::string& name) {
        for (const auto& output : component.output_list_)
            if (output.name == name && output.type == typeid(double))
                return *static_cast<const double*>(output.binding);
        throw std::runtime_error("Missing profile action: " + name);
    }
};
} // namespace rmcs_executor

namespace {

class ProfileConfig : public testing::Test {
protected:
    static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
    static void TearDownTestSuite() { rclcpp::shutdown(); }
};

class ProfilePorts final : public rmcs_executor::Component {
public:
    explicit ProfilePorts(const rmcs_rl::BridgeConfig& config)
        : actions(*this, config.action_terms, config.invalid_action) {}
    void update() override {}
    rmcs_rl::PolicyActions actions;
};

TEST_F(ProfileConfig, InstalledVehicleProfilesHaveConsistentLayouts) {
    for (const auto& [file, size, actions] :
         std::vector<std::tuple<std::string, std::size_t, std::size_t>>{
             {"wheel-leg-infantry-wheelbipe.yaml", 35, 6},
             {"wheel-leg-infantry-wheelbipe-recovery.yaml", 35, 6},
             {"wheel-leg-infantry-prone-v4.yaml", 35, 6},
             {"wheel-leg-infantry-supermimic.yaml", 500, 6}}) {
        SCOPED_TRACE(file);
        const auto options =
            rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true).arguments(
                {"--ros-args", "--params-file", std::string{RMCS_PROFILE_DIRECTORY} + "/" + file});
        rclcpp::Node node{"rl_bridge", options};
        auto config = rmcs_rl::load_bridge_config(node);
        EXPECT_EQ(config.observation_size, size);
        EXPECT_EQ(config.action_terms.size(), actions);
        if (file.find("supermimic") != std::string::npos) {
            ASSERT_EQ(config.observation_terms.size(), 1u);
            EXPECT_EQ(config.observation_terms.front().source, "/wheel_leg/rl/input/supermimic_frame");
            EXPECT_EQ(config.history_length, 1u);
            EXPECT_EQ(config.policy.expected_obs_normalization, "embedded");
            EXPECT_EQ(config.policy.expected_metadata.at("leg_winding_version"),
                      "paired_full_vehicle120_zero110_v1");
            auto mismatch = config.policy;
            mismatch.expected_metadata["leg_winding_version"] = "wrong-winding";
            EXPECT_THROW((rmcs_rl::PolicyModel{mismatch}), std::invalid_argument);
        }
        EXPECT_NO_THROW((rmcs_rl::PolicyModel{config.policy}));
        rmcs_executor::Component::initializing_component_name = "profile_ports";
        ProfilePorts ports{config};
        if (actions == 6) {
            const bool wheelbipe = file.find("wheelbipe") != std::string::npos
                || file.find("supermimic") != std::string::npos;
            ports.actions.publish(std::vector<double>(actions, 2.0));
            for (const auto& term : config.action_terms) {
                const bool wheel = term.output.ends_with("wheel");
                EXPECT_DOUBLE_EQ(term.scale, wheel ? 10.0 : wheelbipe ? .5 : .25);
                EXPECT_DOUBLE_EQ(
                    rmcs_executor::Executor::read(ports, term.output),
                    wheel ? 20.0 : wheelbipe ? 1.0 : .5);
                EXPECT_DOUBLE_EQ(rmcs_executor::Executor::read(ports, term.output + "/raw"), 2.0);
            }
            ports.actions.publish(std::vector<double>(actions, 20.0));
            for (const auto& term : config.action_terms)
                if (term.output.ends_with("wheel")) {
                    EXPECT_DOUBLE_EQ(rmcs_executor::Executor::read(ports, term.output), 90.0);
                    EXPECT_DOUBLE_EQ(rmcs_executor::Executor::read(ports, term.output + "/raw"), 20.0);
                }
        }
        using Component = rmcs_executor::Component;
        Component::OutputInfoMap outputs;
        const auto add = [&]<class T>(const std::string& source) {
            outputs.emplace(source, Component::OutputInfo{typeid(T), rmcs_executor::InterfaceKind::Normal});
        };
        for (const auto& term : config.observation_terms) {
            using enum rmcs_rl::ObservationType;
            switch (term.type) {
            case kScalar: add.template operator()<double>(term.source); break;
            case kVector3: add.template operator()<Eigen::Vector3d>(term.source); break;
            case kArray: add.template operator()<std::vector<double>>(term.source); break;
            case kProjectedGravity: add.template operator()<Eigen::Quaterniond>(term.source); break;
            case kScalars:
                for (const auto& source : term.sources)
                    add.template operator()<double>(source);
                break;
            case kConstant:
            case kPreviousAction: break;
            }
        }
        rmcs_rl::ObservationBuilder builder{config.observation_terms};
        EXPECT_NO_THROW(builder.bind(ports, outputs));
    }
}

} // namespace
