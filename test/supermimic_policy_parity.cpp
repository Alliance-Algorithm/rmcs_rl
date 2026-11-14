// Production ONNX loader on prerecorded flat frames. No vehicle components.
#include "policy/policy_model.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    try {
        rmcs_rl::PolicyConfig config;
        config.path = argv[1];
        config.model_type = "mlp";
        config.expected_model_spec_sha256 =
            "794232d38b1511d9665cc95fbb2f20cc12bbc6ce0162fd050a9085e1d4620177";
        config.expected_metadata["leg_winding_version"] = "paired_full_vehicle120_zero110_v1";
        config.expected_observation_contract_sha256 = argv[2];
        config.expected_observation_version = "supermimic-v202120-h10-g10-v1";
        config.expected_obs_normalization = "embedded";
        config.expected_policy_rate = 50.;
        config.normalization_from_metadata = false;
        config.sequence_length = 1;
        config.feature_size = config.observation_size = 500;
        config.action_size = 6;
        rmcs_rl::PolicyModel model{config};
        std::array<double, 500> observation{};
        std::array<double, 6> action{};
        std::string token;
        std::cout << std::setprecision(17);
        while (std::cin >> token) {
            observation[0] = std::stod(token);
            for (std::size_t i = 1; i < observation.size(); ++i) {
                if (!(std::cin >> token))
                    return 2;
                observation[i] = std::stod(token);
            }
            if (const auto result = model.run(observation, action); !result) {
                std::cerr << result.error() << '\n';
                return 1;
            }
            for (double value : action)
                std::cout << value << ' ';
            std::cout << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
