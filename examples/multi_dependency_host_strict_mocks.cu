// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sparkplug/di/host_mock_functor.hpp>
#include <sparkplug/testing/functor_test.hpp>

#include "multi_dependency_functor.cuh"


namespace {
    using dependency1_t = sparkplug::di::host_strict_gmock_functor_dependency<sparkplug::di::Signature<unsigned long, unsigned>>;
    using dependency2_t = sparkplug::di::host_strict_gmock_functor_dependency<sparkplug::di::Signature<unsigned long, unsigned>>;

    using ::testing::_;
    using ::testing::Eq;
    using ::testing::Return;

    class MultiDepTestWithStrictMocks : public sparkplug::testing::FunctorTest<MultiDependencyFunctor, dependency1_t, dependency2_t> {
    public:
        void SetUp() override {
            InjectDependencies(&mock1_, &mock2_);
        }

    protected:
        dependency1_t::type mock1_{};
        dependency2_t::type mock2_{};
    };
}

TEST_F(MultiDepTestWithStrictMocks, arg_zero) {
    EXPECT_CALL(mock1_, Call(_)).Times(0);
    EXPECT_CALL(mock2_, Call(_)).Times(0);

    ConstructArgumentOnDevice(0);
    ASSERT_THAT(RunOnDevice(), Eq(1234u));
}

TEST_F(MultiDepTestWithStrictMocks, only_mock1) {
    EXPECT_CALL(mock1_, Call(3)).WillOnce(Return(321ul));
    EXPECT_CALL(mock2_, Call(_)).Times(0);

    ConstructArgumentOnDevice(3);
    ASSERT_THAT(RunOnDevice(), Eq(322ul));
}

TEST_F(MultiDepTestWithStrictMocks, both_mocks) {
    EXPECT_CALL(mock1_, Call(5)).WillOnce(Return(321ul));
    EXPECT_CALL(mock2_, Call(5)).WillOnce(Return(9));

    ConstructArgumentOnDevice(5);
    ASSERT_THAT(RunOnDevice(), Eq(331ul));
}

TEST_F(MultiDepTestWithStrictMocks, only_mock2) {
    EXPECT_CALL(mock1_, Call(_)).Times(0);
    EXPECT_CALL(mock2_, Call(11)).WillOnce(Return(9));

    ConstructArgumentOnDevice(11);
    ASSERT_THAT(RunOnDevice(), Eq(10ul));
}