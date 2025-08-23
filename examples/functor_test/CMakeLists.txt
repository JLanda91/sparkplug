add_executable(example_functor_test
    functor_test.cu
)

target_link_libraries(example_functor_test PRIVATE
        sparkplug
        GTest::gtest_main
        GTest::gmock_main
)