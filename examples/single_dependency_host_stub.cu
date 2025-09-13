// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sparkplug/di/stub_functor.cuh>
#include <sparkplug/testing/functor_test.hpp>

#include "factorial_functor.cuh"


namespace {
    using dependency_t = sparkplug::di::host_stub_functor_dependency<sparkplug::util::Signature<int, int>>;

    using ::testing::_;
    using ::testing::Eq;
    using ::testing::Return;

    struct SingleDepTestWithHostStub : sparkplug::testing::FunctorTest<Factorial, dependency_t> {
        void SetUp() override {
            InjectDependencies(&stub_);
        }

    protected:
        dependency_t::type stub_{24};
    };
}

TEST_F(SingleDepTestWithHostStub, arg_le_one) {
    ConstructArgumentOnDevice(-1);
    ASSERT_THAT(RunOnDevice(), Eq(0));

    ConstructArgumentOnDevice(0);
    ASSERT_THAT(RunOnDevice(), Eq(1));

    ConstructArgumentOnDevice(1);
    ASSERT_THAT(RunOnDevice(), Eq(1));
}

TEST_F(SingleDepTestWithHostStub, arg_gt_one) {
    ConstructArgumentOnDevice(5);
    ASSERT_THAT(RunOnDevice(), Eq(120));
}