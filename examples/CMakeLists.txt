# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 Jasper Landa
#


include(GoogleTest)
enable_testing()

#set(SANITIZE_FLAGS "-fno-sanitize=leak" "-fsanitize=address")

#set(CMAKE_CUDA_HOST_FLAGS "${CMAKE_CUDA_HOST_FLAGS} ${SANITIZE_FLAGS}")

function(sparkplug_add_example name)
    set(target_name "example_${name}")
    add_executable(${target_name} "${name}.cu")
    set_target_properties(${target_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${target_name} PRIVATE sparkplug GTest::gtest_main GTest::gmock_main)
#    target_compile_options(${target_name} PRIVATE ${SANITIZE_FLAGS})
#    target_link_options(${target_name} PRIVATE -static-libasan ${SANITIZE_FLAGS})
    add_test(NAME ${target_name} COMMAND ${target_name})
#    set_tests_properties(${target_name} PROPERTIES
#        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0:log_path=asan.log;LSAN_OPTIONS=verbosity=1"
#    )
endfunction()

sparkplug_add_example(single_dependency_host_strict_mock)
sparkplug_add_example(single_dependency_device_stub)
sparkplug_add_example(single_dependency_host_stub)
sparkplug_add_example(multi_dependency_host_strict_mock_device_stub)
sparkplug_add_example(multi_dependency_host_strict_mocks)