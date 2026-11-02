#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_rl/rl_bridge/config.hpp>
#include <rmcs_rl/rl_bridge/observation.hpp>
#include <rmcs_rl/rl_bridge/observation_frame_gate.hpp>
#include <rmcs_rl/rl_bridge/term_parser.hpp>

namespace {
using Gate = rmcs_rl::ObservationFrameGate;
using Decision = Gate::Decision;

TEST(ObservationFrameGate, EachProducerFrameIsSubmittedOnce) {
    Gate gate;
    EXPECT_EQ(gate.observe(0), Decision::kWait);
    EXPECT_EQ(gate.observe(1), Decision::kSubmit);
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(gate.observe(1), Decision::kWait);
    EXPECT_EQ(gate.observe(9), Decision::kSubmit);
    EXPECT_EQ(gate.observe(9), Decision::kWait);
    EXPECT_EQ(gate.observe(0), Decision::kReset);
    EXPECT_EQ(gate.observe(0), Decision::kWait);
    EXPECT_EQ(gate.observe(10), Decision::kSubmit);
}

TEST(ObservationFrameGate, RollbackRequiresExplicitReset) {
    Gate gate;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    EXPECT_EQ(gate.observe(maximum - 1), Decision::kSubmit);
    EXPECT_EQ(gate.observe(maximum), Decision::kSubmit);
    EXPECT_EQ(gate.observe(1), Decision::kReset);
    EXPECT_EQ(gate.observe(0), Decision::kWait);
    EXPECT_EQ(gate.observe(maximum), Decision::kWait);
    gate.reset();
    EXPECT_EQ(gate.observe(1), Decision::kSubmit);
}

class ObservationFrame : public testing::Test {
protected:
    static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
    static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST_F(ObservationFrame, RejectsMalformedArraySpecification) {
    rclcpp::Node node{"frame_parser"};
    const rmcs_rl::JointConfig joints;
    const rmcs_rl::TermParseContext context{node, joints, 6};
    for (const auto* text :
         {"path=/frame take=array", "path=/frame take=array dim=0",
          "path=/frame take=array dim=4097", "path=/frame take=array dim=35.0",
          "path=/frame take=array dim=35 type=vec3", "path=/frame take=vec3 dim=35",
          "path=/frame take=array dim=35 default=0", "path=/frame type=double_array"})
        EXPECT_THROW(rmcs_rl::parse_obs_term(text, context), std::invalid_argument) << text;
}

TEST_F(ObservationFrame, CopiesWholeFrameAndRejectsWrongShapeOrNonFiniteInput) {
    rclcpp::Node node{"frame_copy"};
    const rmcs_rl::JointConfig joints;
    auto term = rmcs_rl::parse_obs_term(
        "path=/frame take=array type=double_array dim=35 index=0", {node, joints, 6});
    term.slot = 0;
    std::vector<rmcs_rl::Slot> slots(1);
    auto& slot = slots.front();
    slot.binding = rmcs_rl::Binding::kDoubleArray;
    slot.array_value =
        std::make_unique<rmcs_executor::Component::InputInterface<std::vector<double>>>();
    std::vector<double> frame(35);
    for (std::size_t i = 0; i < frame.size(); ++i)
        frame[i] = static_cast<double>(i) / 7.0;
    slot.array_value->bind_directly(frame);
    std::vector<double> output;
    ASSERT_TRUE(rmcs_rl::build_observation({term}, slots, {}, 35, output));
    EXPECT_EQ(output, frame);
    frame.push_back(0.0);
    EXPECT_FALSE(rmcs_rl::build_observation({term}, slots, {}, 35, output));
    frame.resize(35);
    frame[17] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(rmcs_rl::build_observation({term}, slots, {}, 35, output));
    frame[17] = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(rmcs_rl::build_observation({term}, slots, {}, 35, output));
    frame[17] = 0;
    EXPECT_FALSE(rmcs_rl::build_observation({term}, slots, {}, 34, output));
    term.slot = 1;
    EXPECT_FALSE(rmcs_rl::build_observation({term}, slots, {}, 35, output));
}

TEST_F(ObservationFrame, RejectsExternalFrameHistoryAndSecondTransform) {
    const std::vector<std::vector<rclcpp::Parameter>> invalid_overrides{
        {rclcpp::Parameter{"history_length", 2}},
        {rclcpp::Parameter{"history_rate", 100.0}},
        {rclcpp::Parameter{
            "observation_terms",
            std::vector<std::string>{"path=/frame take=array dim=35 scale=2"}}},
        {rclcpp::Parameter{
            "observation_terms",
            std::vector<std::string>{"path=/frame take=array dim=35 clip=-1:1"}}},
        {rclcpp::Parameter{
            "observation_terms",
            std::vector<std::string>{"path=/frame take=array dim=34", "path=/extra"}}},
    };
    for (const auto& invalid : invalid_overrides) {
        std::vector<rclcpp::Parameter> parameters{
            {"action_terms", std::vector<std::string>{"index=0 output=/target"}},
            {"observation_terms", std::vector<std::string>{"path=/frame take=array dim=35"}},
            {"observation_frame_sequence_interface", "/frame_sequence"},
            {"policy_rate", 200.0},
        };
        for (const auto& replacement : invalid) {
            std::erase_if(parameters, [&](const auto& value) {
                return value.get_name() == replacement.get_name();
            });
            parameters.push_back(replacement);
        }
        auto options = rclcpp::NodeOptions{}
                           .automatically_declare_parameters_from_overrides(true)
                           .parameter_overrides(parameters);
        rclcpp::Node node{"invalid_frame_contract", options};
        try {
            (void)rmcs_rl::load_bridge_config(node);
            FAIL() << "Accepted invalid external frame configuration";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string{error.what()}.find("external frame"), std::string::npos)
                << error.what();
        }
    }
}

TEST_F(ObservationFrame, ConfigurationDoesNotLoadAModelOrStartAnInferenceWorker) {
    const auto options =
        rclcpp::NodeOptions{}
            .automatically_declare_parameters_from_overrides(true)
            .parameter_overrides({
                {"action_terms", std::vector<std::string>{"index=0 output=/target"}},
                {"observation_terms", std::vector<std::string>{"path=/frame take=array dim=35"}},
                {"observation_frame_sequence_interface", "/frame_sequence"},
                {"rl_model_path", "/nonexistent/rmcs-configuration-only.onnx"},
                {"policy_rate", 100.0},
            });
    rclcpp::Node node{"configuration_only", options};
    const auto config = rmcs_rl::load_bridge_config(node);
    EXPECT_EQ(config.obs_size, 35u);
    EXPECT_EQ(config.action_size, 1u);
    EXPECT_EQ(config.policy.path, "/nonexistent/rmcs-configuration-only.onnx");
    EXPECT_EQ(config.policy.expected_policy_rate, 100.);
}
} // namespace
