// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sparkplug/testing/backends.hpp>
#include <sparkplug/testing/functor_test.hpp>
#include <sparkplug/di/host_mock.hpp>

#include "factorial_functor.cuh"

using backend_t = SPARKPLUG_STRICT_MOCK_HOST_BACKEND(int, int);
using mock_t = backend_t::type;

namespace {
    using ::testing::_;
    using ::testing::Eq;
    using ::testing::Return;

    struct FactorialTestWithStrictMock : sparkplug::testing::FunctorTest<Factorial, backend_t>{};
}

TEST_F(FactorialTestWithStrictMock, input_le_one) {
    mock_t mock{};
    PrepareBackend(&mock);

    EXPECT_CALL(mock, Call(_)).Times(0);

    ASSERT_THAT(RunOnDevice(-1), Eq(0));
    ASSERT_THAT(RunOnDevice(0), Eq(1));
    ASSERT_THAT(RunOnDevice(1), Eq(1));
}

TEST_F(FactorialTestWithStrictMock, input_gt_one) {
    mock_t mock{};
    PrepareBackend(&mock);

    EXPECT_CALL(mock, Call(4)).WillOnce(Return(24));

    ASSERT_THAT(RunOnDevice(5), ::testing::Eq(120));
}