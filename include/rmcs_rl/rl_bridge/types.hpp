#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <rclcpp/node.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>

namespace rmcs_rl {

inline constexpr std::size_t kNoSlot = std::numeric_limits<std::size_t>::max();

enum class TermKind { kPath, kJointPos, kJointVel, kJointTorque, kLastAction, kConstant };

enum class Take { kScalar, kComponent, kVector, kGravity };

enum class Binding {
    kDouble,
    kBool,
    kInt,
    kSize,
    kVector3,
    kDirectionVector,
    kQuaternion,
};

enum class InvalidMode { kNaN, kZero, kHold };

struct ObsTerm {
    TermKind kind = TermKind::kPath;
    Take take = Take::kScalar;

    std::string path;
    std::string id;
    std::size_t dim = 1;

    std::size_t index = 0;
    bool has_index = false;
    int component = 0;
    bool has_binding = false;
    Binding binding = Binding::kDouble;

    std::vector<std::size_t> joints;
    std::vector<std::string> joint_names;
    std::vector<std::size_t> joint_slots;
    std::vector<double> joint_defaults;
    bool relative = false;
    bool zero = false;

    std::vector<std::size_t> action_indices;
    std::vector<double> constants;

    double scale = 1.0;
    bool has_clip = false;
    double clip_min = 0.0;
    double clip_max = 0.0;
    bool has_default = false;
    double default_value = 0.0;

    std::size_t slot = kNoSlot;
};

struct ActionTerm {
    std::size_t index = 0;
    std::string output;
    std::string id;
    double scale = 1.0;
    bool has_clip = false;
    double clip_min = 0.0;
    double clip_max = 0.0;
};

struct Slot {
    std::string path;
    Binding binding = Binding::kDouble;
    bool required = true;
    std::unique_ptr<rmcs_executor::Component::InputInterface<double>> double_value;
    std::unique_ptr<rmcs_executor::Component::InputInterface<bool>> bool_value;
    std::unique_ptr<rmcs_executor::Component::InputInterface<int>> int_value;
    std::unique_ptr<rmcs_executor::Component::InputInterface<std::size_t>> size_value;
    std::unique_ptr<rmcs_executor::Component::InputInterface<Eigen::Vector3d>> vector3_value;
    std::unique_ptr<
        rmcs_executor::Component::InputInterface<rmcs_description::BaseLink::DirectionVector>>
        direction_vector_value;
    std::unique_ptr<rmcs_executor::Component::InputInterface<Eigen::Quaterniond>> quaternion_value;
};

struct ActionSnapshot {
    std::uint64_t obs_seq = 0;
    std::uint64_t layout_hash = 0;
    std::uint64_t model_id = 0;
    std::vector<double> action;
    std::chrono::steady_clock::time_point received{};
};

} // namespace rmcs_rl
