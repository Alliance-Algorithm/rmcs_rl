#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

#include "policy/policy_worker.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument(
                "Usage: wheel_leg_latency_bench MODEL.onnx OUTPUT.csv SECONDS");
        const double seconds = std::stod(argv[3]);
        if (!std::isfinite(seconds) || seconds <= 0 || seconds > 300)
            throw std::invalid_argument("Duration must be in (0,300]");
        rmcs_rl::PolicyConfig config;
        config.path = argv[1];
        config.observation_size = 35;
        config.action_size = 6;
        config.normalization_from_metadata = false;
        rmcs_rl::PolicyWorker worker{config};
        std::array<double, 35> observation{};
        observation[5] = -1.;
        rmcs_rl::PolicyWorker::Result result;
        std::ofstream stream{argv[2]};
        if (!stream)
            throw std::runtime_error("Cannot open benchmark CSV");
        stream << "sequence,source_frame_ns,queued_ns,inference_start_ns,inference_end_ns,result_"
                  "ready_ns,first_take_ns\n";
        const auto ns = [](auto time) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch())
                .count();
        };
        using Clock = rmcs_rl::PolicyWorker::Clock;
        const auto start = Clock::now();
        auto tick = start;
        std::uint64_t sequence = 0, received = 0;
        constexpr double hz = 50.;
        auto next_submit = start;
        while (Clock::now() - start < std::chrono::duration<double>{seconds}) {
            const auto now = Clock::now();
            if (now >= next_submit) {
                if (!worker.submit(observation, ++sequence, 1, now))
                    throw std::runtime_error("Observation rejected during benchmark");
                next_submit += std::chrono::microseconds{static_cast<long>(1e6 / hz)};
            }
            if (worker.try_take(result)) {
                if (!result.success)
                    throw std::runtime_error(result.error);
                stream << result.sequence << ',' << ns(result.submitted_at) << ','
                       << ns(result.queued_at) << ',' << ns(result.inference_started_at) << ','
                       << ns(result.inference_completed_at) << ',' << ns(result.ready_at) << ','
                       << ns(Clock::now()) << '\n';
                ++received;
            }
            tick += std::chrono::milliseconds{1};
            std::this_thread::sleep_until(tick);
        }
        stream.flush();
        if (!stream || !worker.healthy() || received < sequence * .9)
            throw std::runtime_error("Benchmark failed: write, worker health or result coverage");
        std::cerr << "Host-only ONNX worker benchmark, no hardware or PD: submitted=" << sequence
                  << " received=" << received << " rejected=" << worker.submit_rejected()
                  << " pending_overwritten=" << worker.pending_overwritten()
                  << " result_overwritten=" << worker.result_overwritten() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
