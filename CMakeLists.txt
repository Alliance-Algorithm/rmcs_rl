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
find_package(rmcs_executor REQUIRED)
find_package(rmcs_description REQUIRED)
find_package(rmcs_msgs REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json REQUIRED)
ament_auto_find_build_dependencies()

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

ament_auto_add_library(rmcs_rl SHARED
    src/policy/configuration.cpp
    src/policy/policy_model.cpp
    src/policy/onnx_runtime.cpp
    src/policy/policy_worker.cpp
    src/parameters.cpp
)
ament_target_dependencies(rmcs_rl rmcs_executor rmcs_description)
target_include_directories(rmcs_rl PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/rmcs_rl>
)
target_include_directories(rmcs_rl PUBLIC ${ONNXRUNTIME_ROOT_DIR}/include)
target_link_libraries(rmcs_rl
    ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so OpenSSL::Crypto nlohmann_json::nlohmann_json)

ament_auto_add_library(rmcs_rl_components SHARED
    src/rl_bridge.cpp
    src/rl_bridge/config.cpp
    src/rl_bridge/observation_history.cpp
    src/rl_bridge/utility.cpp
    src/rl_bridge/joint_config.cpp
    src/rl_bridge/term_parser.cpp
    src/rl_bridge/interface_binding.cpp
    src/rl_bridge/observation.cpp
)
target_include_directories(rmcs_rl_components PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
ament_target_dependencies(rmcs_rl_components rmcs_executor rmcs_description rclcpp)
target_link_libraries(rmcs_rl_components rmcs_rl)

pluginlib_export_plugin_description_file(rmcs_executor plugins.xml)

file(GLOB WHEEL_LEG_SOURCE CONFIGURE_DEPENDS src/wheel_leg/*.cpp)
ament_auto_add_library(rmcs_rl_wheel_leg SHARED ${WHEEL_LEG_SOURCE})
ament_target_dependencies(rmcs_rl_wheel_leg
    rmcs_executor rmcs_description rmcs_msgs rmcs_utility rclcpp)
target_link_libraries(rmcs_rl_wheel_leg rmcs_rl)
set_target_properties(rmcs_rl rmcs_rl_components rmcs_rl_wheel_leg PROPERTIES INSTALL_RPATH "$ORIGIN")
install(DIRECTORY ${ONNXRUNTIME_ROOT_DIR}/lib/ DESTINATION lib FILES_MATCHING PATTERN "*.so*")

install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
