#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_rl/rl_bridge/config.hpp>
#include <rmcs_rl/rl_bridge/interface_binding.hpp>
#include <rmcs_rl/rl_bridge/observation_history.hpp>

#include "wheel_leg/rl_controller.hpp"

namespace rmcs::rl {

// Exercise the real controller without constructing hardware or enabling drives.
struct WheelLegDeploymentTestAccess {
    static void bind(RlController& c) {
        c.timestamp_.make_and_bind_directly(std::chrono::steady_clock::now());
        c.height_command_.make_and_bind_directly(0.305);
        c.velocity_command_.make_and_bind_directly(
            rmcs_description::BaseLink::DirectionVector{0.2, 0.0, 0.3});
        c.orientation_.make_and_bind_directly(Eigen::Quaterniond::Identity());
        c.gyro_.make_and_bind_directly(Eigen::Vector3d{0.2, -0.4, 0.6});
        c.chassis_mode_.make_and_bind_directly(rmcs_msgs::ChassisMode::AUTO);
        c.jump_apex_command_.make_and_bind_directly(0.0);
        c.state_ = State::kPrepare;
        c.feedback_valid_ = true;
        for (int i = 0; i < 6; ++i) {
            c.q_[i] = c.nominal_[i];
            c.dq_[i] = 0.5 * (i + 1);
            c.previous_action_[i] = 0.1 * (i + 1);
        }
    }
    static auto& q(RlController& c) { return c.q_; }
    static auto& dq(RlController& c) { return c.dq_; }
    static auto& previous(RlController& c) { return c.previous_action_; }
    static const auto& nominal(RlController& c) { return c.nominal_; }
    static const auto& targets(RlController& c) { return c.policy_targets_; }
    static auto& worker(RlController& c) { return *c.policy_worker_; }
    static auto generation(RlController& c) { return c.policy_generation_; }
    static bool targets_valid(RlController& c) { return c.policy_targets_valid_; }
    static bool assemble(RlController& c, bool shadow = false) {
        return c.assemble_observation_(shadow);
    }
    static const auto& observation(RlController& c) { return c.observation_; }
    static const auto& io(RlController& c) { return c.policy_io_; }
    static bool action(RlController& c, const std::vector<double>& values, bool shadow = false) {
        return c.process_action_(values, shadow).has_value();
    }
    static bool evaluate(RlController& c) { return c.evaluate_policy_(0, false); }
    static bool fault(RlController& c) { return c.fault_latched_; }
    static void expire_action(RlController& c) {
        c.accepted_policy_submitted_at_ =
            std::chrono::steady_clock::now()
            - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>{2 * c.inference_timeout_});
    }
    static void accept(RlController& c) { c.accept_policy_result_(); }
    static void reset(RlController& c) { c.enter_(State::kIdle, true); }
    static auto& history(RlController& c) { return *c.observation_history_; }
    static const auto& recovery(RlController& c) { return *c.joint_reference_recovery_; }
    static void seed_outputs(RlController& c) {
        for (auto& output : c.torque_outputs_)
            *output = 1.0;
        for (auto& output : c.action_target_outputs_)
            *output = 1.0;
    }
    static bool outputs_zero(RlController& c) {
        for (const auto& output : c.torque_outputs_)
            if (*output != 0.0)
                return false;
        for (const auto& output : c.action_target_outputs_)
            if (*output != 0.0)
                return false;
        return !*c.enable_request_;
    }
};

namespace {
using Access = WheelLegDeploymentTestAccess;

class WheelLegYaml : public testing::Test {
protected:
    static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
    static void TearDownTestSuite() { rclcpp::shutdown(); }

    void SetUp() override {
        if (!std::filesystem::exists(config_path("wheel-leg-infantry-recovery.yaml")))
            GTEST_SKIP() << "Vehicle integration tests require the adjacent rmcs_bringup checkout";
    }

    static std::string config_path(const std::string& name) {
        return (std::filesystem::path{WHEEL_LEG_BRINGUP_CONFIG} / name).string();
    }
    static rclcpp::NodeOptions options(
        const std::string& profile = "wheel-leg-infantry-recovery.yaml",
        std::vector<rclcpp::Parameter> overrides = {}) {
        auto result = rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true);
        result.enable_rosout(false);
        result.arguments({"--ros-args", "--params-file", config_path(profile)});
        result.parameter_overrides(std::move(overrides));
        return result;
    }
    static std::unique_ptr<RlController> controller(std::vector<rclcpp::Parameter> overrides = {}) {
        rmcs_executor::Component::initializing_component_name = "rl_controller";
        auto c = std::make_unique<RlController>(
            options("wheel-leg-infantry-recovery.yaml", std::move(overrides)));
        c->before_pairing({});
        Access::bind(*c);
        return c;
    }
};

TEST_F(WheelLegYaml, BothVehicleProfilesLoadTheRealActorAndPairInputs) {
    for (const auto* profile : {"wheel-leg-infantry-rl.yaml", "wheel-leg-infantry-recovery.yaml"}) {
        rmcs_executor::Component::initializing_component_name = "rl_controller";
        RlController c{options(profile)};
        EXPECT_NO_THROW(c.before_pairing({}));
        EXPECT_EQ(Access::io(c).obs_size, 35u);
        EXPECT_EQ(Access::worker(c).info().action_size, 6u);
    }
}

TEST_F(WheelLegYaml, ObservationMatchesDeployed35FeaturesAcrossEncoderTurns) {
    auto c = controller();
    std::mt19937 random{107};
    std::uniform_real_distribution<double> values{-30.0, 30.0};
    for (int sample = 0; sample < 128; ++sample) {
        for (int i = 0; i < 6; ++i) {
            Access::q(*c)[i] = values(random);
            Access::dq(*c)[i] = values(random);
        }
        ASSERT_TRUE(Access::assemble(*c));
        const auto& obs = Access::observation(*c);
        ASSERT_EQ(obs.size(), 35u);
        EXPECT_NEAR(obs[0], 0.2, 1e-7);
        EXPECT_DOUBLE_EQ(obs[1], 0.0);
        EXPECT_NEAR(obs[2], 0.3, 1e-7);
        EXPECT_NEAR(obs[3], 1.525, 1e-7);
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(obs[4 + i], (std::array{0.1, -0.2, 0.3}[i]), 1e-7);
            EXPECT_NEAR(obs[7 + i], i == 2 ? -1.0 : 0.0, 1e-7);
        }
        for (int i = 0; i < 4; ++i)
            EXPECT_NEAR(
                obs[10 + i],
                std::remainder(Access::q(*c)[i] - Access::nominal(*c)[i], 2 * std::numbers::pi),
                1e-6);
        EXPECT_DOUBLE_EQ(obs[14], 0.0);
        EXPECT_DOUBLE_EQ(obs[15], 0.0);
        for (int i = 0; i < 6; ++i) {
            EXPECT_NEAR(obs[16 + i], Access::dq(*c)[i] * 0.1, 1e-7);
            EXPECT_NEAR(obs[22 + i], Access::previous(*c)[i], 1e-7);
        }
        EXPECT_DOUBLE_EQ(obs[28], 1.0);
        for (int i = 29; i < 35; ++i)
            EXPECT_DOUBLE_EQ(obs[i], 0.0);
    }
}

TEST_F(WheelLegYaml, ShadowRecoveryKeepsNeutralCommandsAndExecutedHistory) {
    auto c = controller();
    ASSERT_TRUE(Access::assemble(*c, true));
    const auto& obs = Access::observation(*c);
    EXPECT_DOUBLE_EQ(obs[0], 0.0);
    EXPECT_DOUBLE_EQ(obs[2], 0.0);
    EXPECT_NEAR(obs[3], 1.525, 1e-7);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(obs[22 + i], Access::previous(*c)[i], 1e-7);
}

TEST_F(WheelLegYaml, YamlFeatureOrderingAndScalingChangeTheRealObservation) {
    auto base = controller();
    auto specs = base->get_parameter("observation_terms").as_string_array();
    specs[0] = "index=0 path=/wheel_leg/rl/input/command take=z scale=2 name=command/forward";
    specs[2] = "index=2 path=/wheel_leg/rl/input/command take=x scale=3 name=command/yaw";
    auto c = controller({rclcpp::Parameter{"observation_terms", specs}});
    ASSERT_TRUE(Access::assemble(*c));
    EXPECT_NEAR(Access::observation(*c)[0], 0.6, 1e-7);
    EXPECT_NEAR(Access::observation(*c)[2], 0.6, 1e-7);
}

TEST_F(WheelLegYaml, YamlActionPermutationScalesAndLimitsDrivePhysicalTargets) {
    auto base = controller();
    auto specs = base->get_parameter("action_terms").as_string_array();
    specs[0] = "index=0 output=/wheel_leg/rl/target/left_wheel scale=5 clip=-40:40";
    specs[4] = "index=4 output=/wheel_leg/rl/target/left_hip_joint scale=0.125 clip=-0.5:0.5";
    auto c = controller({rclcpp::Parameter{"action_terms", specs}});
    ASSERT_TRUE(Access::action(*c, {20.0, 0.0, 0.0, 0.0, 0.8, -20.0}));
    EXPECT_DOUBLE_EQ(Access::targets(*c)[4], 40.0);
    EXPECT_DOUBLE_EQ(Access::targets(*c)[5], -90.0);
    EXPECT_NEAR(Access::targets(*c)[0], Access::nominal(*c)[0] + 0.1, 1e-7);
    EXPECT_DOUBLE_EQ(Access::previous(*c)[0], 8.0);
    EXPECT_DOUBLE_EQ(Access::previous(*c)[4], 0.8);
    EXPECT_DOUBLE_EQ(Access::previous(*c)[5], -9.0);
}

TEST_F(WheelLegYaml, NonfiniteActionCannotReplaceTheLastTarget) {
    auto c = controller();
    ASSERT_TRUE(Access::action(*c, {0, 0, 0, 0, 1, 2}));
    const auto previous = Access::targets(*c).eval();
    EXPECT_FALSE(Access::action(*c, {0, 0, 0, 0, std::numeric_limits<double>::quiet_NaN(), 0}));
    EXPECT_EQ(Access::targets(*c), previous);
}

TEST_F(WheelLegYaml, StartupRejectsWrongModelShapeAndUnknownActuator) {
    EXPECT_THROW(controller({rclcpp::Parameter{"rl_obs_size", 34}}), std::invalid_argument);
    EXPECT_THROW(controller({rclcpp::Parameter{"rl_feature_size", 34}}), std::invalid_argument);
    EXPECT_THROW(
        controller(
            {rclcpp::Parameter{"rl_obs_size", 36}, rclcpp::Parameter{"rl_feature_size", 36},
             rclcpp::Parameter{
                 "observation_terms",
                 std::vector<std::string>{"index=0 type=constant "
                                          "value=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                                          "0,0,0,0,0,0,0,0,0,0,0 name=wrong_contract"}}}),
        std::invalid_argument);
    auto base = controller();
    auto specs = base->get_parameter("action_terms").as_string_array();
    specs[0] = "index=0 output=/wheel_leg/rl/target/nonexistent scale=0.25 clip=-0.75:0.75";
    EXPECT_THROW(controller({rclcpp::Parameter{"action_terms", specs}}), std::runtime_error);
}

TEST_F(WheelLegYaml, PairingRejectsObservationTypeMismatchAndControlOutputCycles) {
    auto base = controller();
    auto specs = base->get_parameter("observation_terms").as_string_array();
    specs[4] = "index=4 path=/wheel_leg/rl/input/height take=vec3 name=angular_velocity";
    EXPECT_THROW(controller({rclcpp::Parameter{"observation_terms", specs}}), std::runtime_error);
    specs = base->get_parameter("observation_terms").as_string_array();
    specs[0] = "index=0 path=/wheel_leg/rl/state name=command/forward";
    EXPECT_THROW(controller({rclcpp::Parameter{"observation_terms", specs}}), std::runtime_error);
}

TEST_F(WheelLegYaml, ResetClearsEffortActionFeedbackAndObservationHistory) {
    auto c = controller();
    auto& history = Access::history(*c);
    const std::vector<double> old_frame(35, 8.0), new_frame(35, 2.0);
    history.append(old_frame);
    Access::seed_outputs(*c);
    const auto generation = Access::generation(*c);
    Access::reset(*c);
    EXPECT_GT(Access::generation(*c), generation);
    EXPECT_FALSE(Access::targets_valid(*c));
    EXPECT_TRUE(Access::outputs_zero(*c));
    for (const double value : Access::previous(*c))
        EXPECT_DOUBLE_EQ(value, 0.0);
    EXPECT_EQ(history.append(new_frame), new_frame);
}

TEST_F(WheelLegYaml, WorkerPreservesRecoveryContextAndResetGeneration) {
    auto c = controller();
    auto& worker = Access::worker(*c);
    const std::vector<double> obs(35, 0.0);
    ASSERT_TRUE(
        worker.submit(std::span<const double>{obs}, 71, 17, std::chrono::steady_clock::now(), 1));
    rmcs_rl::PolicyWorker::Result result;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while (!worker.try_take(result) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.sequence, 71u);
    EXPECT_EQ(result.generation, 17u);
    EXPECT_EQ(result.context, 1u);
}

TEST_F(WheelLegYaml, PendingFirstInferenceKeepsPreparationAlive) {
    auto c = controller();
    EXPECT_TRUE(Access::evaluate(*c));
    EXPECT_FALSE(Access::fault(*c));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{50};
    while (!Access::targets_valid(*c) && std::chrono::steady_clock::now() < deadline) {
        Access::accept(*c);
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    EXPECT_TRUE(Access::targets_valid(*c));
}

TEST_F(WheelLegYaml, ResetRejectsAnInferenceThatCompletesFromThePreviousSession) {
    auto c = controller();
    EXPECT_TRUE(Access::evaluate(*c));
    Access::reset(*c);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{30};
    while (std::chrono::steady_clock::now() < deadline) {
        Access::accept(*c);
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    EXPECT_FALSE(Access::targets_valid(*c));
    EXPECT_TRUE(Access::outputs_zero(*c));
}

TEST_F(WheelLegYaml, AcceptedActionExpiresWithoutWaitingForAnotherInference) {
    auto c = controller();
    ASSERT_TRUE(Access::action(*c, {0, 0, 0, 0, 1, 2}));
    Access::expire_action(*c);
    Access::accept(*c);
    EXPECT_TRUE(Access::fault(*c));
}

TEST_F(WheelLegYaml, NonfiniteHistoryCannotBeHiddenByObservationClipping) {
    auto c = controller();
    Access::previous(*c)[0] = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(Access::assemble(*c));
    Access::previous(*c)[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(Access::assemble(*c));
}

TEST_F(WheelLegYaml, SharedBridgeStillLoadsTheDeformableHistoryContract) {
    rclcpp::Node node{"rl_bridge", options("deformable-infantry-omni-rl.yaml")};
    const auto config = rmcs_rl::load_bridge_config(node);
    EXPECT_EQ(config.obs_frame_size, 32u);
    EXPECT_EQ(config.history_length, 8u);
    EXPECT_EQ(config.obs_size, 256u);
    EXPECT_EQ(config.action_size, 4u);
    EXPECT_DOUBLE_EQ(config.policy_rate, 200.0);
    EXPECT_TRUE(config.policy_worker->healthy());
}

TEST_F(WheelLegYaml, ReusingAnUpstreamSlotStillChecksTheRequestedType) {
    class InputConsumer : public rmcs_executor::Component {
    public:
        void update() override {}
    } consumer;
    const rmcs_executor::Component::OutputInfoMap outputs{
        {"/test/input", {std::cref(typeid(double)), rmcs_executor::InterfaceKind::Normal}}};
    std::vector<rmcs_rl::Slot> slots;
    EXPECT_EQ(
        rmcs_rl::acquire_slot(
            consumer, "/test/input", {rmcs_rl::Binding::kDouble}, true, outputs, slots, "test"),
        0u);
    EXPECT_THROW(
        rmcs_rl::acquire_slot(
            consumer, "/test/input", {rmcs_rl::Binding::kVector3}, true, outputs, slots, "test"),
        std::runtime_error);
}

TEST_F(WheelLegYaml, ConfiguredHistoryChangesShapeAndResetPrimesWithFreshFrames) {
    rclcpp::Node node{
        "rl_controller",
        options(
            "wheel-leg-infantry-recovery.yaml",
            {rclcpp::Parameter{"history_length", 3}, rclcpp::Parameter{"rl_obs_size", 105}})};
    const auto config = rmcs_rl::load_policy_io_config(node);
    EXPECT_EQ(config.obs_size, 105u);
    rmcs_rl::ObservationHistory history{config.obs_frame_size, config.history_length};
    const std::vector<double> old_frame(35, 1.0), fresh_frame(35, 2.0);
    EXPECT_EQ(history.append(old_frame), std::vector<double>(105, 1.0));
    history.reset();
    EXPECT_EQ(history.append(fresh_frame), std::vector<double>(105, 2.0));
}

TEST_F(WheelLegYaml, RecoveryHistoryUsesConfiguredNominalAndPerMotorScales) {
    auto c = controller();
    auto recovery = Access::recovery(*c);
    const JointReferenceRecoveryVector6 nominal =
        Eigen::Map<const JointReferenceRecoveryVector6>{Access::nominal(*c).data()};
    auto q = nominal;
    q[0] += 0.2;
    q[1] += 0.2;
    recovery.reset(q, true);
    const JointReferenceRecoveryVector6 scales{0.125, 0.25, 0.5, -0.25, 5.0, -10.0};
    const auto result =
        recovery.effective_action_history(JointReferenceRecoveryVector6::Zero(), nominal, scales);
    const auto& command = recovery.command();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(
            result[i],
            (1.0 - command.blend)
                * std::remainder(command.targets[i] - nominal[i], 2 * std::numbers::pi) / scales[i],
            1e-5);
}

} // namespace
} // namespace rmcs::rl
