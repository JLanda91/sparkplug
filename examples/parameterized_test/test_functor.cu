#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <vector>

#include <sparkplug/mocking/mock_signature.hpp>
#include <sparkplug/testing/parameterized_test.hpp>

#include "functor.cuh"

struct FunctorTest : sparkplug::testing::ParameterizedTest<Functor, SPARKPLUG_SIGNATURE(int, int)> {};

TEST_P(FunctorTest, call_operator) {
        RunOnDevice();
        const auto& mock = GetMock();
        const auto retval = GetActualReturnValue();

        ASSERT_THAT(mock, sparkplug::mocking::IsCalledWith(GetExpectedMockInputs()));
        ASSERT_THAT(retval, ::testing::Eq(GetExpectedReturnValue()));
}

static std::vector<FunctorTest::test_case_t> test_cases{{
        { .input = -1, .expected_return = 0, .mock_config = {{}, {}}},
        { .input = -1, .expected_return = 3, .mock_config = {{}, {}}},
        { .input = 0, .expected_return = 1, .mock_config = {{}, {}}},
        { .input = 1, .expected_return = 1, .mock_config = {{0}, {1}}},
        { .input = 5, .expected_return = 120, .mock_config = {{4}, {24}}},
        { .input = 5, .expected_return = 120, .mock_config = {{}, {}}}
}};

INSTANTIATE_TEST_SUITE_P(sparkplug, FunctorTest, ::testing::ValuesIn(test_cases));

