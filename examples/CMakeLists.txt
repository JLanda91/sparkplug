# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 Jasper Landa


include(GoogleTest)
enable_testing()

function(sparkplug_add_example name)
    set(target_name "example_${name}")
    add_executable(${target_name} "${name}.cu")
    target_link_libraries(${target_name} PRIVATE sparkplug GTest::gtest_main GTest::gmock_main)
endfunction()

sparkplug_add_example(single_backend_host_strict_mock)
sparkplug_add_example(single_backend_device_stub)
sparkplug_add_example(single_backend_host_stub)