cmake_minimum_required(VERSION 3.24)
project(rmcs_wheel_leg_rl)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-packed-bitfield-compat")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O2 -Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# ---- ONNX Runtime (CPU 推理) ----
# 官方预编译包：https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0
# SHA256 对应 linux-x64 1.20.0；换平台/版本时必须同步更新。
# 交叉编译到其他架构时设置 -DONNXRUNTIME_PLATFORM=<平台>（如 linux-aarch64）。
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

file(GLOB_RECURSE PROJECT_SOURCE CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.c
)

ament_auto_add_library(
    ${PROJECT_NAME} SHARED
    ${PROJECT_SOURCE}
)

include_directories(${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)

pluginlib_export_plugin_description_file(rmcs_executor plugins.xml)

ament_auto_package()
