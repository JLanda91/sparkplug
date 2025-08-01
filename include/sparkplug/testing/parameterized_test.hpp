#pragma once

#include <thrust/device_vector.h>

#include <sparkplug/testing/launch_test_kernel.cuh>
#include <sparkplug/mocking/mock_config.hpp>
#include <sparkplug/mocking/mock.hpp>
#include <sparkplug/mocking/detail/device_side_mock.cuh>
#include <sparkplug/testing/detail/deduced_signature.hpp>

namespace sparkplug::testing {

template<template<typename> typename Functor, typename BackendSignature>
struct TestCase {
    using functor_t = Functor<mocking::detail::DeviceSideMock<BackendSignature>>;
    using functor_signature_t = typename detail::DeducedSignature<functor_t>::type;
    static_assert(detail::DeducedSignature<functor_t>::value, "Type under test must be a functor");

    typename functor_signature_t::input_t input;
    typename functor_signature_t::return_t expected_return;
    mocking::MockConfig<BackendSignature> mock_config;
};

template <template <typename> typename Functor, typename BackendSignature>
class ParameterizedTest : public ::testing::TestWithParam<TestCase<Functor, BackendSignature>> {
public:
    using test_case_t = TestCase<Functor, BackendSignature>;
    using functor_t = typename test_case_t::functor_t;
    using functor_signature_t = typename test_case_t::functor_signature_t;

    static_assert(std::is_constructible_v<functor_t, const typename mocking::Mock<BackendSignature>::device_type* >,
        "Functor is not constructible with a const pointer to mock with provided signature");

    static void SetUpTestSuite() {
        d_input.resize(1);
        d_return.resize(1);
    }

    static void TearDownTestSuite() {
        d_input.clear();
        d_return.clear();
    }

    void SetUp() override {
        const auto& test_case = this->GetParam();
        mock = std::make_unique<mocking::Mock<BackendSignature>>(test_case.mock_config);
        d_input[0] = test_case.input;
    }

    void RunOnDevice() {
        launch_test_kernel<functor_t>(
            thrust::raw_pointer_cast(d_input.data()),
            thrust::raw_pointer_cast(d_return.data()),
            mock->device_object()
        );
    }

    [[nodiscard]] auto GetMock() const -> const mocking::Mock<BackendSignature>& {
        return *mock;
    }

    [[nodiscard]] auto& GetExpectedMockInputs() const {
        return this->GetParam().mock_config.expected_inputs();
    }

    [[nodiscard]] auto GetActualReturnValue() const {
        return static_cast<typename functor_signature_t::return_t>(d_return[0]);
    }

    [[nodiscard]] auto GetExpectedReturnValue() const {
        return this->GetParam().expected_return;
    }

protected:

    static thrust::device_vector<typename functor_signature_t::input_t> d_input;
    static thrust::device_vector<typename functor_signature_t::return_t> d_return;
    std::unique_ptr<mocking::Mock<BackendSignature>> mock = nullptr;
};

template <template <typename> typename Functor, typename BackendSignature>
thrust::device_vector<typename ParameterizedTest<Functor, BackendSignature>::functor_signature_t::input_t> ParameterizedTest<Functor, BackendSignature>::d_input = {};

template <template <typename> typename Functor, typename BackendSignature>
thrust::device_vector<typename ParameterizedTest<Functor, BackendSignature>::functor_signature_t::return_t> ParameterizedTest<Functor, BackendSignature>::d_return = {};

// #define SPARKPLUG_FUNCTOR_TEST_P(TestName, FunctorTemplate, MockSignature) \
//     struct TestName : sparkplug::testing::ParameterizedTest<FunctorTemplate, MockSignature> {};   \
//     TEST_P(TestName, call_operator) { Run(GetParam()); }

}