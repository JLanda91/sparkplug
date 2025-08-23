#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sparkplug/testing/backends.hpp>
#include <sparkplug/testing/functor_test.hpp>
#include <sparkplug/mocking/host_mock.hpp>

#include "functor.cuh"

using mock_t = SPARKPLUG_STRICT_HOST_MOCK(int, int);
using backend_t = sparkplug::testing::HostBackend<mock_t>;

namespace {
    using ::testing::_;
    using ::testing::Eq;
    using ::testing::Return;

    struct FunctorTestWithStrictHostMock : sparkplug::testing::FunctorTest<Functor, backend_t>{};
}

TEST_F(FunctorTestWithStrictHostMock, input_le_one) {
    mock_t mock{};
    PrepareBackend(&mock);

    EXPECT_CALL(mock, Call(_)).Times(0);

    ASSERT_THAT(RunOnDevice(-1), Eq(0));
    ASSERT_THAT(RunOnDevice(0), Eq(1));
    ASSERT_THAT(RunOnDevice(1), Eq(1));
}

TEST_F(FunctorTestWithStrictHostMock, input_gt_one) {
    mock_t mock{};
    PrepareBackend(&mock);

    EXPECT_CALL(mock, Call(4)).WillOnce(Return(24));

    ASSERT_THAT(RunOnDevice(5), ::testing::Eq(120));
}