cmake_minimum_required(VERSION 3.24)
project(rmcs_rl)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O2 -Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(nlohmann_json REQUIRED)

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

ament_auto_add_library(${PROJECT_NAME} SHARED
    src/rl_bridge.cpp
    src/mirror/state_mirror.cpp
    src/policy/policy_model.cpp
    src/policy/policy_worker.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE src ${ONNXRUNTIME_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")
install(DIRECTORY ${ONNXRUNTIME_ROOT_DIR}/lib/ DESTINATION lib FILES_MATCHING PATTERN "*.so*")

pluginlib_export_plugin_description_file(rmcs_executor plugins.xml)

ament_auto_add_executable(wheel_leg_policy_host tool/simulation/wheel_leg_policy_host.cpp)
target_link_libraries(wheel_leg_policy_host nlohmann_json::nlohmann_json)
ament_auto_add_executable(wheel_leg_latency_bench tool/simulation/wheel_leg_latency_bench.cpp)
target_include_directories(wheel_leg_latency_bench PRIVATE src ${ONNXRUNTIME_ROOT_DIR}/include)
target_link_libraries(wheel_leg_latency_bench ${PROJECT_NAME})

install(FILES tool/check_policy.py DESTINATION share/${PROJECT_NAME}/tool)

install(DIRECTORY models model_receipts DESTINATION share/${PROJECT_NAME})
install(DIRECTORY tool/mirror tool/visualization DESTINATION share/${PROJECT_NAME}/tool FILES_MATCHING PATTERN "*.py")

ament_auto_package()
