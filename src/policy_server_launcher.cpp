#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_rl/parameters.hpp>

namespace rmcs_rl {

class PolicyServerLauncher
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    explicit PolicyServerLauncher()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {
        autostart_ = bool_or(*this, "autostart", true);
        respawn_ = bool_or(*this, "respawn", true);
        respawn_delay_ = number_or(*this, "respawn_delay", 1.0);
        poll_interval_ = number_or(*this, "poll_interval", 0.5);
        params_file_ = string_or(*this, "params_file", "");
        const auto rl_base = string_or(*this, "rl_base", "/chassis/rl");
        register_input(rl_base + "/observation_size", observation_size_);
        register_input(rl_base + "/action_size", action_size_);
        register_input(rl_base + "/history_length", history_length_);
        register_input(rl_base + "/feature_size", feature_size_);
        startup_timeout_ = number_or(*this, "startup_timeout", 30.0);
        if (!std::isfinite(startup_timeout_) || startup_timeout_ <= 0.0)
            throw std::invalid_argument("policy startup_timeout must be finite and positive");

        resolve_paths_();
    }

    void before_updating() override {
        last_poll_time_ = std::chrono::steady_clock::now();
        if (autostart_)
            start_child_();
    }

    void update() override {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_poll_time_ < poll_period_)
            return;
        last_poll_time_ = now;

        if (startup_fd_ >= 0)
            check_child_readiness_(false);

        reap_child_(now);
        if (autostart_ && respawn_ && child_pid_ < 0 && now >= next_start_time_)
            start_child_(false);
    }

    ~PolicyServerLauncher() override { stop_child_(); }

private:
    using SteadyClock = std::chrono::steady_clock;

    void resolve_paths_() {
        try {
            const auto share = ament_index_cpp::get_package_share_directory("rmcs_rl");
            const auto marker = share.rfind("/share/");
            if (marker != std::string::npos)
                executable_ = share.substr(0, marker) + "/lib/rmcs_rl/policy_server";
        } catch (const std::exception& error) {
            RCLCPP_ERROR(get_logger(), "cannot locate rmcs_rl share directory: %s", error.what());
        }

        if (executable_.empty() || ::access(executable_.c_str(), X_OK) != 0) {
            if (autostart_)
                throw std::runtime_error("policy_server executable not found: " + executable_);
        }

        if (!params_file_.empty()) {
            if (params_file_.front() == '/') {
                params_path_ = params_file_;
            } else {
                try {
                    const auto bringup_share = ament_index_cpp::get_package_share_directory(
                        "rmcs_"
                        "bringu"
                        "p");
                    params_path_ = bringup_share + "/config/" + params_file_;
                } catch (const std::exception& error) {
                    RCLCPP_ERROR(
                        get_logger(), "cannot locate rmcs_bringup share directory: %s",
                        error.what());
                }
            }
        }

        if (params_path_.empty() || ::access(params_path_.c_str(), R_OK) != 0) {
            if (autostart_)
                throw std::runtime_error("policy_server params file not found: " + params_file_);
        }

        poll_period_ = std::chrono::duration_cast<SteadyClock::duration>(
            std::chrono::duration<double>(poll_interval_ > 0.0 ? poll_interval_ : 0.5));
        respawn_period_ = std::chrono::duration_cast<SteadyClock::duration>(
            std::chrono::duration<double>(respawn_delay_ >= 0.0 ? respawn_delay_ : 1.0));
    }

    void start_child_(bool wait_for_readiness = true) {
        if (executable_.empty() || params_path_.empty())
            return;

        std::vector<std::string> arguments;
        arguments.push_back(executable_);
        arguments.push_back("--ros-args");
        arguments.push_back("--params-file");
        arguments.push_back(params_path_);
        const auto parameter = [&](const char* name, std::size_t value) {
            arguments.emplace_back("-p");
            arguments.emplace_back(std::string{name} + ":=" + std::to_string(value));
        };
        parameter("rl_obs_size", *observation_size_);
        parameter("rl_action_size", *action_size_);
        parameter("sequence_length", *history_length_);
        parameter("feature_size", *feature_size_);

        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1);
        for (auto& argument : arguments)
            argv.push_back(argument.data());
        argv.push_back(nullptr);

        int startup_pipe[2];
        if (::pipe(startup_pipe) != 0)
            throw std::runtime_error(std::string{"policy startup pipe: "} + std::strerror(errno));
        ::fcntl(startup_pipe[0], F_SETFD, FD_CLOEXEC);
        std::vector<std::string> environment;
        for (char** entry = environ; *entry; ++entry)
            if (!std::string_view{*entry}.starts_with("RMCS_RL_STARTUP_FD="))
                environment.emplace_back(*entry);
        environment.emplace_back("RMCS_RL_STARTUP_FD=" + std::to_string(startup_pipe[1]));
        std::vector<char*> envp;
        for (auto& entry : environment)
            envp.push_back(entry.data());
        envp.push_back(nullptr);

        const auto parent_pid = ::getpid();
        const auto pid = ::fork();
        if (pid < 0) {
            ::close(startup_pipe[0]);
            ::close(startup_pipe[1]);
            throw std::runtime_error(std::string{"policy fork failed: "} + std::strerror(errno));
        }

        if (pid == 0) {
            ::close(startup_pipe[0]);
            // Only async-signal-safe calls are allowed inside the child.
            ::prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (::getppid() != parent_pid)
                ::_exit(EXIT_FAILURE);
            ::execve(argv[0], argv.data(), envp.data());
            ::_exit(EXIT_FAILURE);
        }

        child_pid_ = pid;
        ::close(startup_pipe[1]);
        startup_fd_ = startup_pipe[0];
        startup_deadline_ = SteadyClock::now()
                          + std::chrono::duration_cast<SteadyClock::duration>(
                                std::chrono::duration<double>(startup_timeout_));
        next_start_time_ = SteadyClock::now() + respawn_period_;
        check_child_readiness_(wait_for_readiness);
    }

    void check_child_readiness_(bool wait) {
        pollfd descriptor_poll{startup_fd_, POLLIN, 0};
        int result;
        do {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       startup_deadline_ - SteadyClock::now())
                                       .count();
            result = ::poll(
                &descriptor_poll, 1,
                wait ? static_cast<int>(std::max(std::int64_t{0}, remaining)) : 0);
        } while (result < 0 && errno == EINTR);
        if (result == 0 && !wait && SteadyClock::now() < startup_deadline_)
            return;
        std::array<char, 2049> message{};
        const auto count =
            result > 0 ? ::read(startup_fd_, message.data(), message.size() - 1) : -1;
        ::close(startup_fd_);
        startup_fd_ = -1;
        if (count < 1 || message[0] != 'R') {
            const std::string reason =
                count > 1 && message[0] == 'E'
                    ? std::string{message.data() + 1, static_cast<std::size_t>(count - 1)}
                : result == 0 ? "model initialization timed out"
                              : "policy process exited before readiness";
            respawn_ = false;
            stop_child_();
            if (wait)
                throw std::runtime_error("policy_server initialization failed: " + reason);
            RCLCPP_ERROR(
                get_logger(), "policy_server restart initialization failed: %s", reason.c_str());
            return;
        }
        next_start_time_ = SteadyClock::now() + respawn_period_;
        RCLCPP_INFO(
            get_logger(), "policy_server ready (pid=%d) with params '%s'",
            static_cast<int>(child_pid_), params_path_.c_str());
    }

    void reap_child_(SteadyClock::time_point now) {
        if (child_pid_ <= 0)
            return;

        int status = 0;
        const auto result = ::waitpid(child_pid_, &status, WNOHANG);
        if (result == 0)
            return;

        if (result < 0 && errno != ECHILD) {
            RCLCPP_WARN(
                get_logger(), "waitpid(%d) failed: %s", static_cast<int>(child_pid_),
                std::strerror(errno));
        }

        const auto pid = child_pid_;
        child_pid_ = -1;
        next_start_time_ = now + respawn_period_;

        if (result == pid && WIFEXITED(status))
            RCLCPP_WARN(
                get_logger(), "policy_server (pid=%d) exited with code %d", static_cast<int>(pid),
                WEXITSTATUS(status));
        else if (result == pid && WIFSIGNALED(status))
            RCLCPP_WARN(
                get_logger(), "policy_server (pid=%d) killed by signal %d", static_cast<int>(pid),
                WTERMSIG(status));
        else
            RCLCPP_WARN(get_logger(), "policy_server (pid=%d) is gone", static_cast<int>(pid));
    }

    void stop_child_() {
        if (startup_fd_ >= 0) {
            ::close(startup_fd_);
            startup_fd_ = -1;
        }
        if (child_pid_ <= 0)
            return;

        const auto pid = child_pid_;
        child_pid_ = -1;

        ::kill(pid, SIGTERM);
        for (int attempt = 0; attempt < 40; ++attempt) {
            const auto result = ::waitpid(pid, nullptr, WNOHANG);
            if (result == pid || (result < 0 && errno == ECHILD))
                return;
            ::usleep(50'000);
        }

        ::kill(pid, SIGKILL);
        ::waitpid(pid, nullptr, 0);
        RCLCPP_WARN(
            get_logger(), "policy_server (pid=%d) did not stop, killed", static_cast<int>(pid));
    }

    InputInterface<std::size_t> observation_size_;
    InputInterface<std::size_t> action_size_;
    InputInterface<std::size_t> history_length_;
    InputInterface<std::size_t> feature_size_;
    bool autostart_ = true;
    double startup_timeout_ = 30.0;
    bool respawn_ = true;
    double respawn_delay_ = 1.0;
    double poll_interval_ = 0.5;

    std::string params_file_;
    std::string executable_;
    std::string params_path_;

    SteadyClock::duration poll_period_ = std::chrono::milliseconds(500);
    SteadyClock::duration respawn_period_ = std::chrono::seconds(1);
    SteadyClock::time_point last_poll_time_{};
    SteadyClock::time_point next_start_time_{};
    pid_t child_pid_ = -1;
    int startup_fd_ = -1;
    SteadyClock::time_point startup_deadline_{};
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_rl::PolicyServerLauncher, rmcs_executor::Component)
