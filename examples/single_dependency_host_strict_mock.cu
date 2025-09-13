// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sparkplug/di/host_mock_functor.hpp>
#include <sparkplug/testing/functor_test.hpp>

#include "factorial_functor.cuh"

namespace {
    using dependency_t = sparkplug::di::host_strict_gmock_functor_dependency<sparkplug::util::Signature<int, int>>;

    using ::testing::_;
    using ::testing::Eq;
    using ::testing::Return;

    class SingleDepTestWithStrictMock : public sparkplug::testing::FunctorTest<Factorial, dependency_t> {
    public:
        void SetUp() override {
            InjectDependencies(&mock_);
        }

    protected:
        dependency_t::type mock_{};
    };
}

TEST_F(SingleDepTestWithStrictMock, arg_le_one) {
    EXPECT_CALL(mock_, Call(_)).Times(0);

    ConstructArgumentOnDevice(-1);
    ASSERT_THAT(RunOnDevice(), Eq(0));

    ConstructArgumentOnDevice(0);
    ASSERT_THAT(RunOnDevice(), Eq(1));

    ConstructArgumentOnDevice(1);
    ASSERT_THAT(RunOnDevice(), Eq(1));
}

TEST_F(SingleDepTestWithStrictMock, arg_gt_one) {
    EXPECT_CALL(mock_, Call(4)).WillOnce(Return(24));

    ConstructArgumentOnDevice(5);
    ASSERT_THAT(RunOnDevice(), Eq(120));
}