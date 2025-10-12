# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 Jasper Landa
#


cmake_minimum_required(VERSION 3.28)

project(Sandbox LANGUAGES CXX CUDA VERSION 1.0.0)

find_package(GTest CONFIG REQUIRED)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ===== BUILD RULES =====
add_library(sparkplug STATIC
        src/cuda_fn_error.cu
        src/stream.cu
        src/get_device_properties.cu
)
target_compile_features(sparkplug PUBLIC cxx_std_20)
target_link_libraries(sparkplug PUBLIC
        GTest::gtest
        GTest::gmock
)
target_include_directories(sparkplug PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(sparkplug PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
)

include(GoogleTest)
enable_testing()
add_subdirectory(examples)

# ===== INSTALL RULES =====
include(GNUInstallDirs)

install(TARGETS sparkplug
        EXPORT sparkplug-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT sparkplug-targets
        FILE sparkplug-config.cmake
        NAMESPACE sparkplug::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sparkplug
)