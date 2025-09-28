# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 Jasper Landa
#


include(GoogleTest)
enable_testing()

set_property(DIRECTORY PROPERTY ALL_EXAMPLES "")

function(sparkplug_add_example name)
    set(target_name "example_${name}")
    add_executable(${target_name} "${name}.cu")
    set_target_properties(${target_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${target_name} PRIVATE sparkplug GTest::gtest_main GTest::gmock_main)
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_property(DIRECTORY APPEND PROPERTY ALL_EXAMPLES ${target_name})
endfunction()

sparkplug_add_example(single_dependency_host_strict_mock)
sparkplug_add_example(single_dependency_device_stub)
sparkplug_add_example(single_dependency_host_stub)
sparkplug_add_example(multi_dependency_host_strict_mock_device_stub)
sparkplug_add_example(multi_dependency_host_strict_mocks)

get_property(SPARKPLUG_EXAMPLE_TARGETS DIRECTORY PROPERTY ALL_EXAMPLES)
message(STATUS "example targets = ${SPARKPLUG_EXAMPLE_TARGETS}")
add_custom_target(all_examples DEPENDS ${SPARKPLUG_EXAMPLE_TARGETS})