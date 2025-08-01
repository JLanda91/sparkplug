add_executable(examples_parameterized_test
    test_functor.cu
)

target_link_libraries(examples_parameterized_test PRIVATE
        sparkplug
        GTest::gtest_main
        GTest::gmock_main
)