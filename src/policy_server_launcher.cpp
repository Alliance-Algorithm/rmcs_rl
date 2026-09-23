
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_executor/component.hpp>

namespace rmcs::rl {

// 随 executor 生命周期拉起独立的 policy_server 子进程：
//   - 组件只存在于需要 RL 的配置里，非 RL 车不受影响；
//   - 子进程设置 PR_SET_PDEATHSIG，executor 结束/崩溃时自动被内核回收，不会留孤儿；
//   - update() 低频 waitpid(WNOHANG) 监管，可选退避重启。
class PolicyServerLauncher
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    PolicyServerLauncher()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {
        autostart_ = bool_or_("autostart", true);
        respawn_ = bool_or_("respawn", true);
        respawn_delay_ = number_or_("respawn_delay", 1.0);
        poll_interval_ = number_or_("poll_interval", 0.5);
        params_file_ = string_or_("params_file", "");

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

        reap_child_(now);
        if (autostart_ && respawn_ && child_pid_ < 0 && now >= next_start_time_)
            start_child_();
    }

    ~PolicyServerLauncher() override { stop_child_(); }

private:
    using SteadyClock = std::chrono::steady_clock;

    std::string string_or_(const std::string& name, const std::string& fallback) const {
        if (!has_parameter(name))
            return fallback;
        const auto parameter = get_parameter(name);
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING)
            return fallback;
        return parameter.as_string();
    }

    double number_or_(const std::string& name, double fallback) const {
        if (!has_parameter(name))
            return fallback;
        const auto parameter = get_parameter(name);
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
            return parameter.as_double();
        if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
            return static_cast<double>(parameter.as_int());
        return fallback;
    }

    bool bool_or_(const std::string& name, bool fallback) const {
        if (!has_parameter(name))
            return fallback;
        const auto parameter = get_parameter(name);
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL)
            return fallback;
        return parameter.as_bool();
    }

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
            RCLCPP_ERROR(get_logger(),
                "policy_server executable not found (looked for '%s'); autostart disabled",
                executable_.c_str());
            autostart_ = false;
        }

        if (!params_file_.empty()) {
            if (params_file_.front() == '/') {
                params_path_ = params_file_;
            } else {
                try {
                    const auto bringup_share =
                        ament_index_cpp::get_package_share_directory("rmcs_bringup");
                    params_path_ = bringup_share + "/config/" + params_file_;
                } catch (const std::exception& error) {
                    RCLCPP_ERROR(get_logger(), "cannot locate rmcs_bringup share directory: %s",
                        error.what());
                }
            }
        }

        if (params_path_.empty() || ::access(params_path_.c_str(), R_OK) != 0) {
            RCLCPP_ERROR(get_logger(),
                "policy_server params file not found (params_file='%s'); autostart disabled",
                params_file_.c_str());
            autostart_ = false;
        }

        poll_period_ = std::chrono::duration_cast<SteadyClock::duration>(
            std::chrono::duration<double>(poll_interval_ > 0.0 ? poll_interval_ : 0.5));
        respawn_period_ = std::chrono::duration_cast<SteadyClock::duration>(
            std::chrono::duration<double>(respawn_delay_ >= 0.0 ? respawn_delay_ : 1.0));
    }

    void start_child_() {
        if (executable_.empty() || params_path_.empty())
            return;

        std::vector<std::string> arguments;
        arguments.push_back(executable_);
        arguments.push_back("--ros-args");
        arguments.push_back("--params-file");
        arguments.push_back(params_path_);

        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1);
        for (auto& argument : arguments)
            argv.push_back(argument.data());
        argv.push_back(nullptr);

        const auto parent_pid = ::getpid();
        const auto pid = ::fork();
        if (pid < 0) {
            RCLCPP_ERROR(get_logger(), "fork() failed: %s; retrying", std::strerror(errno));
            next_start_time_ = SteadyClock::now() + respawn_period_;
            return;
        }

        if (pid == 0) {
            // 子进程内只调用 async-signal-safe 的接口。
            ::prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (::getppid() != parent_pid)
                ::_exit(EXIT_FAILURE);
            ::execv(argv[0], argv.data());
            ::_exit(EXIT_FAILURE);
        }

        child_pid_ = pid;
        next_start_time_ = SteadyClock::now() + respawn_period_;
        RCLCPP_INFO(get_logger(), "started policy_server (pid=%d) with params '%s'",
            static_cast<int>(pid), params_path_.c_str());
    }

    void reap_child_(SteadyClock::time_point now) {
        if (child_pid_ <= 0)
            return;

        int status = 0;
        const auto result = ::waitpid(child_pid_, &status, WNOHANG);
        if (result == 0)
            return;

        if (result < 0 && errno != ECHILD) {
            RCLCPP_WARN(get_logger(), "waitpid(%d) failed: %s", static_cast<int>(child_pid_),
                std::strerror(errno));
        }

        const auto pid = child_pid_;
        child_pid_ = -1;
        next_start_time_ = now + respawn_period_;

        if (result == pid && WIFEXITED(status))
            RCLCPP_WARN(get_logger(), "policy_server (pid=%d) exited with code %d",
                static_cast<int>(pid), WEXITSTATUS(status));
        else if (result == pid && WIFSIGNALED(status))
            RCLCPP_WARN(get_logger(), "policy_server (pid=%d) killed by signal %d",
                static_cast<int>(pid), WTERMSIG(status));
        else
            RCLCPP_WARN(get_logger(), "policy_server (pid=%d) is gone", static_cast<int>(pid));
    }

    void stop_child_() {
        if (child_pid_ <= 0)
            return;

        const auto pid = child_pid_;
        child_pid_ = -1;

        ::kill(pid, SIGTERM);
        for (int attempt = 0; attempt < 40; ++attempt) {
            if (::waitpid(pid, nullptr, WNOHANG) == pid)
                return;
            ::usleep(50'000);
        }

        ::kill(pid, SIGKILL);
        ::waitpid(pid, nullptr, 0);
        RCLCPP_WARN(get_logger(), "policy_server (pid=%d) did not stop, killed", static_cast<int>(pid));
    }

    bool autostart_ = true;
    bool respawn_ = true;
    double respawn_delay_ = 1.0;
    double poll_interval_ = 0.5;

    std::string params_file_;
    std::string executable_;
    std::string params_path_;

    SteadyClock::duration poll_period_ = std::chrono::milliseconds(500);
    SteadyClock::duration respawn_period_ = std::chrono::seconds(1);
    SteadyClock::time_point last_poll_time_ { };
    SteadyClock::time_point next_start_time_ { };
    pid_t child_pid_ = -1;
};

} // namespace rmcs::rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs::rl::PolicyServerLauncher, rmcs_executor::Component)
