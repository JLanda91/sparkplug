// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sparkplug/testing/backends.hpp>
#include <sparkplug/testing/functor_test.hpp>
#include <sparkplug/di/stub.cuh>

#include "factorial_functor.cuh"

using backend_t = SPARKPLUG_STUB_DEVICE_BACKEND(int, int);
using stub_t = backend_t::type;

namespace {
    using ::testing::_;
    using ::testing::Eq;
    using ::testing::Return;

    struct FactorialTestWithDeviceStub : sparkplug::testing::FunctorTest<Factorial, backend_t>{};
}

TEST_F(FactorialTestWithDeviceStub, input_le_one) {
    stub_t stub{0};
    PrepareBackend(&stub);

    ASSERT_THAT(RunOnDevice(-1), Eq(0));
    ASSERT_THAT(RunOnDevice(0), Eq(1));
    ASSERT_THAT(RunOnDevice(1), Eq(1));
}

TEST_F(FactorialTestWithDeviceStub, input_gt_one) {
    stub_t stub{24};
    PrepareBackend(&stub);

    ASSERT_THAT(RunOnDevice(5), Eq(120));
}