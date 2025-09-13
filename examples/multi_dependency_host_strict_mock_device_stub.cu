// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sparkplug/di/host_mock_functor.hpp>
#include <sparkplug/di/stub_functor.cuh>
#include <sparkplug/testing/functor_test.hpp>

#include "multi_dependency_functor.cuh"


namespace {
    using dependency1_t = sparkplug::di::host_strict_gmock_functor_dependency<sparkplug::util::Signature<unsigned long, unsigned>>;
    using dependency2_t = sparkplug::di::device_stub_functor_dependency<sparkplug::util::Signature<unsigned long, unsigned>>;

    using ::testing::_;
    using ::testing::Eq;
    using ::testing::Return;

    class MultiDepTestWithStrictMockAndDeviceStub : public sparkplug::testing::FunctorTest<MultiDependencyFunctor, dependency1_t, dependency2_t> {
    public:
        void SetUp() override {
            InjectDependencies(&mock_, &stub_);
        }

    protected:
        dependency1_t::type mock_{};
        dependency2_t::type stub_{9};
    };
}

TEST_F(MultiDepTestWithStrictMockAndDeviceStub, arg_zero) {
    EXPECT_CALL(mock_, Call(_)).Times(0);

    ConstructArgumentOnDevice(0);
    ASSERT_THAT(RunOnDevice(), Eq(1234u));
}

TEST_F(MultiDepTestWithStrictMockAndDeviceStub, only_mock) {
    EXPECT_CALL(mock_, Call(3)).WillOnce(Return(321ul));

    ConstructArgumentOnDevice(3);
    ASSERT_THAT(RunOnDevice(), Eq(322ul));
}

TEST_F(MultiDepTestWithStrictMockAndDeviceStub, both_mock_and_stub) {
    EXPECT_CALL(mock_, Call(5)).WillOnce(Return(321ul));

    ConstructArgumentOnDevice(5);
    ASSERT_THAT(RunOnDevice(), Eq(331ul));
}

TEST_F(MultiDepTestWithStrictMockAndDeviceStub, only_stub) {
    EXPECT_CALL(mock_, Call(_)).Times(0);

    ConstructArgumentOnDevice(11);
    ASSERT_THAT(RunOnDevice(), Eq(10ul));
}