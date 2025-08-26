cmake_minimum_required(VERSION 3.31)

project(Sandbox LANGUAGES CXX CUDA)

find_package(GTest CONFIG REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ===== BUILD RULES =====
add_library(sparkplug STATIC
        src/cuda_error.cu
        src/cuda_stream.cu
        src/device_properties.cu
)
target_compile_features(sparkplug PUBLIC cxx_std_17)
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