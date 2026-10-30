// Offline policy fixture driver. No executor, CAN, or vehicle component.
#include <array>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "policy/policy_model.hpp"

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        rmcs_rl::PolicyConfig config;
        config.path = argv[1];
        config.model_type = "wheel_leg_v202";
        config.expected_model_spec_sha256 =
            "7f62e0b47546a047219c078bdaa44fc2d82553af04b36b72ec1a94be29a22e18";
        config.expected_leg_winding_version = "paired_branch107_v1";
        config.expected_observation_contract_sha256 =
            "6062baf37967d853446a95317aae4a0b315c4267da7387ab7ae020f37a021e35";
        config.expected_observation_version = "wheel-leg-prone-motion-v4";
        config.expected_obs_normalization = "embedded";
        config.expected_policy_rate = 100.;
        config.normalization_from_metadata = false;
        config.obs_clip = 100.;
        config.sequence_length = 1;
        config.feature_size = config.observation_size = 35;
        config.action_size = 6;
        rmcs_rl::PolicyModel model{config};
        std::array<double, 35> observation{};
        std::array<double, 6> action{};
        std::cout << std::setprecision(17);
        std::string token, error;
        while (std::cin >> token) {
            observation[0] = std::strtod(token.c_str(), nullptr);
            for (std::size_t i = 1; i < observation.size(); ++i) {
                if (!(std::cin >> token))
                    return 2;
                observation[i] = std::strtod(token.c_str(), nullptr);
            }
            if (!model.run(observation, action, error)) {
                std::cerr << error << '\n';
                return 1;
            }
            for (double value : action)
                std::cout << value << ' ';
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
