#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <eigen3/Eigen/Geometry>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_msgs/chassis_mode.hpp>
#include <rmcs_msgs/keyboard.hpp>
#include <rmcs_msgs/switch.hpp>
#include <rmcs_utility/rclcpp/node_mixin.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/header.hpp>

namespace rmcs_rl {

class StateMirror
    : public rmcs_executor::Component
    , public rclcpp::Node
    , public rmcs_utility::NodeMixin {
public:
    StateMirror()
        : Node{get_component_name(), node::options()} {
        node::param("topic_prefix", topic_prefix_);
        node::param("frame_id", frame_id_);
        node::param("rate", rate_);
        register_input("/predefined/update_count", update_count_);
        register_input("/predefined/update_rate", update_rate_);

        for (const auto& name : get_parameter_or<std::vector<std::string>>("joints.order", {})) {
            const auto prefix = "joints." + name + ".";
            joints_.push_back(
                {name, reader(prefix + "position"), reader(prefix + "velocity"),
                 reader(prefix + "stamp")});
        }
        if (has_parameter("imu.orientation"))
            imu_ = std::make_unique<Imu>(
                reader("imu.orientation"), reader("imu.angular_velocity"), reader("imu.stamp"));
        for (const auto& name : get_parameter_or<std::vector<std::string>>("vectors.order", {}))
            vectors_.push_back({name, reader("vectors." + name), {}});
        for (const auto& name : list_parameters({"values"}, 0).names)
            values_.push_back({key(name, "values."), reader(name)});
        for (const auto& name : list_parameters({"constants"}, 0).names)
            constants_.emplace(key(name, "constants."), get_parameter(name).as_string());

        const auto qos = rclcpp::QoS{5}.best_effort().durability_volatile();
        if (!joints_.empty())
            joint_publisher_ = create_publisher<sensor_msgs::msg::JointState>(
                topic_prefix_ + "/joint_states", qos);
        if (imu_)
            imu_publisher_ =
                create_publisher<sensor_msgs::msg::Imu>(topic_prefix_ + "/imu_body", qos);
        for (auto& vector : vectors_)
            vector.publisher = create_publisher<geometry_msgs::msg::Vector3Stamped>(
                topic_prefix_ + "/" + vector.name, qos);
        diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            topic_prefix_ + "/diagnostics", qos);
    }

    void before_pairing(const OutputInfoMap& output_map) override {
        for (auto& [path, reader] : readers_)
            reader->bind(*this, path, output_map);
    }

    void before_updating() override {
        if (!std::isfinite(rate_) || rate_ <= 0. || rate_ > *update_rate_)
            throw std::invalid_argument("Mirror rate must be in (0, update_rate]");
        using DirectionVector = rmcs_description::BaseLink::DirectionVector;
        for (const auto& joint : joints_) {
            require<double>(joint.position);
            require<double>(joint.velocity);
            require<std::uint64_t>(joint.stamp);
        }
        if (imu_) {
            require<Eigen::Quaterniond>(imu_->orientation);
            require<Eigen::Vector3d, DirectionVector>(imu_->angular_velocity);
            require<std::uint64_t>(imu_->stamp);
        }
        for (const auto& vector : vectors_)
            require<Eigen::Vector3d, DirectionVector>(vector.value);
        for (const auto& value : values_)
            if (value.value->unsupported())
                throw std::invalid_argument(
                    "Mirror cannot format interface " + value.value->path());
    }

    void update() override {
        const auto divisor = static_cast<std::size_t>(std::lround(*update_rate_ / rate_));
        if (*update_count_ % divisor != 0)
            return;
        std_msgs::msg::Header header;
        header.stamp = get_clock()->now();
        header.frame_id = frame_id_;
        std::string topics = topic_prefix_ + "/diagnostics";
        if (publish_joints(header))
            topics += "," + topic_prefix_ + "/joint_states";
        if (publish_imu(header))
            topics += "," + topic_prefix_ + "/imu_body";
        for (const auto& vector : vectors_) {
            if (!ready(vector.value))
                continue;
            geometry_msgs::msg::Vector3Stamped message;
            message.header = header;
            const auto value = vector.value->as_vector3();
            message.vector.x = value.x();
            message.vector.y = value.y();
            message.vector.z = value.z();
            vector.publisher->publish(message);
            topics += "," + topic_prefix_ + "/" + vector.name;
        }
        publish_diagnostics(header, topics);
    }

private:
    class Reader {
        template <typename T>
        using Interface = InputInterface<T>;
        using DirectionVector = rmcs_description::BaseLink::DirectionVector;
        using Input = std::variant<
            Interface<double>, Interface<bool>, Interface<int>, Interface<std::uint64_t>,
            Interface<std::string>, Interface<rmcs_msgs::Switch>, Interface<rmcs_msgs::ChassisMode>,
            Interface<rmcs_msgs::Keyboard>, Interface<Eigen::Vector2d>, Interface<Eigen::Vector3d>,
            Interface<DirectionVector>, Interface<Eigen::Quaterniond>,
            Interface<std::array<double, 2>>, Interface<std::array<double, 4>>,
            Interface<std::array<double, 6>>, Interface<std::vector<double>>>;

    public:
        void bind(Component& component, const std::string& path, const OutputInfoMap& outputs) {
            const auto found = outputs.find(path);
            if (found == outputs.end())
                return;
            path_ = path;
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                (
                    [&] {
                        using T = std::variant_alternative_t<I, Input>;
                        using Value = std::remove_cvref_t<decltype(*std::declval<T>())>;
                        if (found->second.type.get() == typeid(Value)) {
                            input_ = std::make_unique<Input>(std::in_place_type<T>);
                            component.register_input(path, std::get<T>(*input_), false);
                        }
                    }(),
                    ...);
            }(std::make_index_sequence<std::variant_size_v<Input>>{});
        }

        const std::string& path() const { return path_; }

        // A produced interface whose type the mirror cannot read.
        bool unsupported() const { return !path_.empty() && !input_; }

        template <typename... Types>
        bool holds() const {
            return input_ && (std::holds_alternative<Interface<Types>>(*input_) || ...);
        }

        bool ready() const {
            return input_ && std::visit([](const auto& input) { return input.ready(); }, *input_);
        }

        double as_double() const { return *std::get<Interface<double>>(*input_); }

        Eigen::Vector3d as_vector3() const {
            if (const auto* input = std::get_if<Interface<DirectionVector>>(input_.get()))
                return (*input)->vector;
            return *std::get<Interface<Eigen::Vector3d>>(*input_);
        }

        Eigen::Quaterniond as_quaternion() const {
            return *std::get<Interface<Eigen::Quaterniond>>(*input_);
        }

        double age(std::uint64_t now) const {
            const auto stamp = *std::get<Interface<std::uint64_t>>(*input_);
            return stamp == 0 || stamp > now ? -1. : static_cast<double>(now - stamp) / 1e9;
        }

        void append(
            const std::string& key, std::vector<diagnostic_msgs::msg::KeyValue>& values) const {
            std::visit(
                [&key, &values](const auto& input) {
                    using T = std::remove_cvref_t<decltype(*input)>;
                    const auto add = [&values](const std::string& key, const auto& value) {
                        auto& entry = values.emplace_back();
                        entry.key = key;
                        if constexpr (
                            std::is_same_v<std::remove_cvref_t<decltype(value)>, std::string>)
                            entry.value = value;
                        else if constexpr (
                            std::is_same_v<std::remove_cvref_t<decltype(value)>, bool>)
                            entry.value = value ? "true" : "false";
                        else
                            entry.value = std::format("{}", value);
                    };
                    if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::string>) {
                        add(key, *input);
                    } else if constexpr (std::is_enum_v<T>) {
                        add(key, static_cast<int>(*input));
                    } else if constexpr (std::is_same_v<T, rmcs_msgs::Keyboard>) {
                        add(key, std::bit_cast<std::uint16_t>(*input));
                    } else if constexpr (std::is_same_v<T, Eigen::Quaterniond>) {
                        add(key + "/w", input->w());
                        add(key + "/x", input->x());
                        add(key + "/y", input->y());
                        add(key + "/z", input->z());
                    } else if constexpr (std::is_same_v<T, DirectionVector>) {
                        add(key + "/x", input->vector.x());
                        add(key + "/y", input->vector.y());
                        add(key + "/z", input->vector.z());
                    } else if constexpr (
                        std::is_same_v<T, Eigen::Vector2d> || std::is_same_v<T, Eigen::Vector3d>) {
                        add(key + "/x", input->x());
                        add(key + "/y", input->y());
                        if constexpr (std::is_same_v<T, Eigen::Vector3d>)
                            add(key + "/z", input->z());
                    } else {
                        for (std::size_t i = 0; i < static_cast<std::size_t>(input->size()); ++i)
                            add(key + "/" + std::to_string(i), (*input)[i]);
                    }
                },
                *input_);
        }

    private:
        std::string path_;
        std::unique_ptr<Input> input_;
    };

    struct Joint {
        std::string name;
        Reader *position, *velocity, *stamp;
    };
    struct Imu {
        Reader *orientation, *angular_velocity, *stamp;
    };
    struct Vector {
        std::string name;
        Reader* value;
        rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr publisher;
    };
    struct Value {
        std::string key;
        Reader* value;
    };

    static bool ready(const Reader* reader) { return reader && reader->ready(); }

    // Interfaces without a producer are omitted; a produced one must have the role's type.
    template <typename... Types>
    static void require(const Reader* reader) {
        if (reader && !reader->path().empty() && !reader->holds<Types...>())
            throw std::invalid_argument("Mirror interface has the wrong type: " + reader->path());
    }

    static std::string key(const std::string& parameter, const std::string& prefix) {
        auto result = parameter.substr(prefix.size());
        std::ranges::replace(result, '.', '/');
        return result;
    }

    Reader* reader(const std::string& parameter) {
        if (!has_parameter(parameter))
            return nullptr;
        const auto path = get_parameter(parameter).as_string();
        auto& result = readers_[path];
        if (!result)
            result = std::make_unique<Reader>();
        return result.get();
    }

    bool publish_joints(const std_msgs::msg::Header& header) const {
        if (!joint_publisher_ || !std::ranges::all_of(joints_, [](const auto& joint) {
                return ready(joint.position) && ready(joint.velocity);
            }))
            return false;
        sensor_msgs::msg::JointState message;
        message.header = header;
        for (const auto& joint : joints_) {
            message.name.push_back(joint.name);
            message.position.push_back(joint.position->as_double());
            message.velocity.push_back(joint.velocity->as_double());
        }
        joint_publisher_->publish(message);
        return true;
    }

    bool publish_imu(const std_msgs::msg::Header& header) const {
        if (!imu_ || !ready(imu_->orientation) || !ready(imu_->angular_velocity))
            return false;
        sensor_msgs::msg::Imu message;
        message.header = header;
        const auto q = imu_->orientation->as_quaternion();
        const auto gyro = imu_->angular_velocity->as_vector3();
        message.orientation.w = q.w();
        message.orientation.x = q.x();
        message.orientation.y = q.y();
        message.orientation.z = q.z();
        message.angular_velocity.x = gyro.x();
        message.angular_velocity.y = gyro.y();
        message.angular_velocity.z = gyro.z();
        message.linear_acceleration_covariance[0] = -1.;
        if (!q.coeffs().allFinite() || std::abs(q.squaredNorm() - 1.) > .01)
            message.orientation_covariance[0] = -1.;
        imu_publisher_->publish(message);
        return true;
    }

    void publish_diagnostics(const std_msgs::msg::Header& header, const std::string& topics) const {
        diagnostic_msgs::msg::DiagnosticArray message;
        message.header = header;
        auto& status = message.status.emplace_back();
        status.name = topic_prefix_;
        status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        const auto add = [&status](const std::string& key, const std::string& value) {
            auto& entry = status.values.emplace_back();
            entry.key = key;
            entry.value = value;
        };
        add("schema", "rmcs_state_mirror_v2");
        add("mirror/topics", topics);
        for (const auto& [key, value] : constants_)
            add(key, value);
        for (const auto& value : values_)
            if (ready(value.value))
                value.value->append(value.key, status.values);
        const auto now =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch())
                                           .count());
        for (const auto& joint : joints_)
            if (joint.stamp)
                add("age_s/" + joint.name,
                    std::format("{}", ready(joint.stamp) ? joint.stamp->age(now) : -1.));
        if (imu_ && imu_->stamp)
            add("age_s/imu", std::format("{}", ready(imu_->stamp) ? imu_->stamp->age(now) : -1.));
        diagnostics_publisher_->publish(message);
    }

    InputInterface<std::size_t> update_count_;
    InputInterface<double> update_rate_;
    std::string topic_prefix_, frame_id_;
    double rate_;
    std::map<std::string, std::unique_ptr<Reader>> readers_;
    std::vector<Joint> joints_;
    std::unique_ptr<Imu> imu_;
    std::vector<Vector> vectors_;
    std::vector<Value> values_;
    std::map<std::string, std::string> constants_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;
};

} // namespace rmcs_rl

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rmcs_rl::StateMirror, rmcs_executor::Component)
