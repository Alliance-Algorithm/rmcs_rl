#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_msgs/chassis_mode.hpp>
#include <rmcs_utility/rclcpp/node_mixin.hpp>

namespace rmcs_rl {

class PipeFeedback
    : public rmcs_executor::Component
    , public rclcpp::Node
    , public rmcs_utility::NodeMixin {
public:
    PipeFeedback()
        : Node{get_component_name(), node::options()} {
        node::param("inputs", input_paths_);
        std::vector<std::string> order;
        node::param("outputs.order", order);
        if (order.empty())
            throw std::invalid_argument("PipeFeedback outputs.order must not be empty");
        for (const auto& name : order) {
            const auto prefix = "outputs." + name + ".";
            const auto path = get_parameter(prefix + "path").as_string();
            const auto type = get_parameter(prefix + "type").as_string();
            const auto initial = get_parameter(prefix + "initial");
            if (path.empty() || path.front() != '/' || outputs_.contains(path))
                throw std::invalid_argument("Invalid or duplicate PipeFeedback output: " + path);
            const auto add = [&]<class T>(const T& value) {
                auto output = std::make_unique<Output<T>>();
                register_output(path, output->value, value);
                // One input makes the command partner run after this component even when
                // none of its requested inputs depend on the configured feedback outputs.
                if (outputs_.empty() && std::ranges::find(input_paths_, path) == input_paths_.end())
                    command_->register_input(path, output->ordering);
                outputs_.emplace(path, std::move(output));
            };
            if (type == "double")
                add(initial.as_double());
            else if (type == "array")
                add(initial.as_double_array());
            else if (type == "bool")
                add(initial.as_bool());
            else if (type == "int") {
                const auto value = initial.as_int();
                if (value < std::numeric_limits<int>::min()
                    || value > std::numeric_limits<int>::max())
                    throw std::invalid_argument("PipeFeedback int initial is out of range");
                add(static_cast<int>(value));
            } else if (type == "uint64") {
                const auto value = initial.as_int();
                if (value < 0)
                    throw std::invalid_argument("PipeFeedback uint64 initial must be nonnegative");
                add(static_cast<std::uint64_t>(value));
            } else if (type == "chassis_mode")
                add(static_cast<rmcs_msgs::ChassisMode>(initial.as_int()));
            else if (type == "vector3" || type == "direction_vector") {
                const auto values = initial.as_double_array();
                if (values.size() != 3)
                    throw std::invalid_argument("PipeFeedback vector initial must have 3 values");
                const Eigen::Vector3d value{values[0], values[1], values[2]};
                if (type == "vector3")
                    add(value);
                else {
                    DirectionVector direction;
                    direction.vector = value;
                    add(direction);
                }
            } else if (type == "quaternion_wxyz") {
                const auto values = initial.as_double_array();
                if (values.size() != 4)
                    throw std::invalid_argument(
                        "PipeFeedback quaternion initial must have 4 values");
                add(Eigen::Quaterniond{values[0], values[1], values[2], values[3]});
            } else
                throw std::invalid_argument("Unsupported PipeFeedback output type: " + type);
        }
    }

    void before_pairing(const OutputInfoMap& output_map) override {
        for (const auto& path : input_paths_) {
            if (inputs_.contains(path))
                throw std::invalid_argument("Duplicate PipeFeedback input: " + path);
            const auto found = output_map.find(path);
            if (found == output_map.end())
                throw std::invalid_argument("Missing PipeFeedback input: " + path);
            if (found->second.kind != rmcs_executor::InterfaceKind::Normal)
                throw std::invalid_argument(
                    "PipeFeedback input must be a normal interface: " + path);
            const auto bind = [&]<class T>() {
                auto input = std::make_unique<InputInterface<T>>();
                command_->register_input(path, *input);
                inputs_.emplace(path, std::move(input));
            };
            const auto& type = found->second.type.get();
            if (type == typeid(double))
                bind.template operator()<double>();
            else if (type == typeid(bool))
                bind.template operator()<bool>();
            else if (type == typeid(int))
                bind.template operator()<int>();
            else if (type == typeid(std::uint64_t))
                bind.template operator()<std::uint64_t>();
            else if (type == typeid(Eigen::Vector3d))
                bind.template operator()<Eigen::Vector3d>();
            else if (type == typeid(DirectionVector))
                bind.template operator()<DirectionVector>();
            else if (type == typeid(Eigen::Quaterniond))
                bind.template operator()<Eigen::Quaterniond>();
            else if (type == typeid(rmcs_msgs::ChassisMode))
                bind.template operator()<rmcs_msgs::ChassisMode>();
            else if (type == typeid(std::vector<double>))
                bind.template operator()<std::vector<double>>();
            else
                throw std::invalid_argument("Unsupported PipeFeedback input type: " + path);
        }
    }

    void before_updating() override { std::cout << "{\"ready\":true}\n" << std::flush; }

    void update() override {
        if (remaining_ == 0) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                eof_ = true;
                rclcpp::shutdown();
                return;
            }
            const auto batch = nlohmann::json::parse(line);
            if (!batch.is_object())
                throw std::invalid_argument("PipeFeedback expects a JSON object");
            const auto count = batch.value("ticks", nlohmann::json(1));
            if (!count.is_number_integer() || count.get<std::int64_t>() <= 0)
                throw std::invalid_argument("PipeFeedback ticks must be a positive integer");
            remaining_ = count.get<std::uint64_t>();
            for (const auto& [path, value] : batch.items()) {
                if (path == "ticks")
                    continue;
                const auto found = outputs_.find(path);
                if (found == outputs_.end())
                    throw std::invalid_argument("Unknown PipeFeedback output: " + path);
                std::visit(
                    [&value](auto& output) {
                        using T = std::remove_cvref_t<decltype(*output->value)>;
                        *output->value = decode<T>(value);
                    },
                    found->second);
            }
        }
        --remaining_;
    }

private:
    using DirectionVector = rmcs_description::BaseLink::DirectionVector;

    struct Command : public rmcs_executor::Component {
        explicit Command(std::function<void()> fn)
            : fn{std::move(fn)} {}
        void update() override { fn(); }
        std::function<void()> fn;
    };

    template <class T>
    struct Output {
        OutputInterface<T> value;
        InputInterface<T> ordering;
    };

    template <template <class> class Interface>
    using Slots = std::variant<
        std::unique_ptr<Interface<double>>, std::unique_ptr<Interface<bool>>,
        std::unique_ptr<Interface<int>>, std::unique_ptr<Interface<std::uint64_t>>,
        std::unique_ptr<Interface<Eigen::Vector3d>>, std::unique_ptr<Interface<DirectionVector>>,
        std::unique_ptr<Interface<Eigen::Quaterniond>>,
        std::unique_ptr<Interface<std::vector<double>>>,
        std::unique_ptr<Interface<rmcs_msgs::ChassisMode>>>;

    template <class T>
    static T decode(const nlohmann::json& value) {
        if constexpr (std::is_same_v<T, Eigen::Vector3d> || std::is_same_v<T, DirectionVector>) {
            if (!value.is_array() || value.size() != 3)
                throw std::invalid_argument("PipeFeedback vector must have 3 values");
            const Eigen::Vector3d vector{
                value.at(0).get<double>(), value.at(1).get<double>(), value.at(2).get<double>()};
            if constexpr (std::is_same_v<T, DirectionVector>) {
                T result;
                result.vector = vector;
                return result;
            } else
                return vector;
        } else if constexpr (std::is_same_v<T, Eigen::Quaterniond>) {
            if (!value.is_array() || value.size() != 4)
                throw std::invalid_argument("PipeFeedback quaternion must have 4 values");
            return {
                value.at(0).get<double>(), value.at(1).get<double>(), value.at(2).get<double>(),
                value.at(3).get<double>()};
        } else if constexpr (std::is_same_v<T, std::uint64_t>) {
            if (!value.is_number_unsigned()
                && (!value.is_number_integer() || value.get<std::int64_t>() < 0))
                throw std::invalid_argument("PipeFeedback uint64 must be nonnegative");
            return value.get<T>();
        } else if constexpr (std::is_enum_v<T>)
            return static_cast<T>(decode<int>(value));
        else if constexpr (std::is_same_v<T, int>) {
            if (!value.is_number_integer() || value < std::numeric_limits<int>::min()
                || value > std::numeric_limits<int>::max())
                throw std::invalid_argument("PipeFeedback int must be an integer in range");
            return value.get<int>();
        } else
            return value.get<T>();
    }

    template <class T>
    static nlohmann::json encode(const T& value) {
        if constexpr (std::is_same_v<T, Eigen::Vector3d>)
            return {value.x(), value.y(), value.z()};
        else if constexpr (std::is_same_v<T, DirectionVector>)
            return encode(value.vector);
        else if constexpr (std::is_same_v<T, Eigen::Quaterniond>)
            return {value.w(), value.x(), value.y(), value.z()};
        else if constexpr (std::is_enum_v<T>)
            return static_cast<int>(value);
        else
            return value;
    }

    void publish() {
        if (remaining_ != 0 || eof_)
            return;
        auto batch = nlohmann::json::object();
        for (const auto& [path, input] : inputs_)
            std::visit(
                [&batch, &path](const auto& value) { batch[path] = encode(**value); }, input);
        std::cout << batch.dump() << '\n' << std::flush;
    }

    std::shared_ptr<Command> command_{create_partner_component<Command>(
        get_component_name() + "_command", [this] { publish(); })};
    std::map<std::string, Slots<Output>> outputs_;
    std::map<std::string, Slots<InputInterface>> inputs_;
    std::vector<std::string> input_paths_;
    std::uint64_t remaining_ = 0;
    bool eof_ = false;
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(rmcs_rl::PipeFeedback, rmcs_executor::Component)
