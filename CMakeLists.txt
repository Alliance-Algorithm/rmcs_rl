cmake_minimum_required(VERSION 3.24)
project(rmcs_rl)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-packed-bitfield-compat")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O2 -Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/Observation.msg"
    "msg/Action.msg"
    "msg/PolicyStatus.msg"
    DEPENDENCIES std_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

set(ONNXRUNTIME_VERSION 1.20.0)
if(NOT DEFINED ONNXRUNTIME_PLATFORM)
    set(ONNXRUNTIME_PLATFORM linux-x64)
endif()
set(ONNXRUNTIME_ROOT_DIR ${CMAKE_BINARY_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION})
if(NOT EXISTS ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime_cxx_api.h)
    set(ONNXRUNTIME_ARCHIVE ${CMAKE_BINARY_DIR}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION}.tgz)
    if(NOT EXISTS ${ONNXRUNTIME_ARCHIVE})
        file(DOWNLOAD
            https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION}.tgz
            ${ONNXRUNTIME_ARCHIVE}
            EXPECTED_HASH SHA256=aa70d48b22e264b82e83f63245b51ddc9a47ae4a3a66903efaff1ba68b7b5930
            SHOW_PROGRESS
        )
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNXRUNTIME_ARCHIVE}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RESULT_VARIABLE ONNXRUNTIME_EXTRACT_RESULT
    )
    if(NOT ONNXRUNTIME_EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ONNX Runtime archive: ${ONNXRUNTIME_ARCHIVE}")
    endif()
endif()
include_directories(${ONNXRUNTIME_ROOT_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

ament_auto_add_library(rmcs_rl_bridge SHARED
    src/rl_bridge.cpp
    src/rl_bridge/utility.cpp
    src/rl_bridge/parameters.cpp
    src/rl_bridge/joint_config.cpp
    src/rl_bridge/term_parser.cpp
    src/rl_bridge/interface_binding.cpp
    src/rl_bridge/observation.cpp
    src/rl_bridge/action_channel.cpp
    src/policy_server_launcher.cpp
)
target_link_libraries(rmcs_rl_bridge ${cpp_typesupport_target})

ament_auto_add_executable(policy_server
    src/policy_server.cpp
)
target_link_libraries(policy_server ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
target_link_libraries(policy_server ${cpp_typesupport_target})
install(TARGETS policy_server DESTINATION lib/${PROJECT_NAME})

pluginlib_export_plugin_description_file(rmcs_executor plugins.xml)

install(DIRECTORY models DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)

ament_auto_package()
