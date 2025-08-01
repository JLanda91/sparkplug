#pragma once

#include <complex>
#include <gmock/gmock-matchers.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <sparkplug/mocking/mock_config.hpp>
#include <sparkplug/mocking/detail/device_side_mock.cuh>

namespace sparkplug::mocking {

template<typename Signature>
class Mock {
public:
    using device_type = detail::DeviceSideMock<Signature>;

    explicit Mock(const MockConfig<Signature>& mock_config)
    : mocked_outputs_(mock_config.mocked_outputs()){}

    [[nodiscard]] auto device_object() const {
        return thrust::raw_pointer_cast(device_object_.data());
    }

    Mock(const Mock&) = delete;
    Mock& operator=(const Mock&) = delete;

    auto call_counter() const {
        return static_cast<std::size_t>(call_counter_[0]);
    }

    auto actual_inputs() const {
        const auto n = std::min(call_counter(), actual_inputs_.size());
        thrust::host_vector<typename Signature::input_t> h_actual_inputs(n);
        thrust::copy_n(actual_inputs_.cbegin(), n, h_actual_inputs.begin());
        return h_actual_inputs;
    }

private:
    thrust::device_vector<typename Signature::return_t> mocked_outputs_;
    std::size_t n_ = mocked_outputs_.size();
    thrust::device_vector<typename Signature::input_t> actual_inputs_ = thrust::device_vector<typename Signature::input_t>(n_);
    thrust::device_vector<std::size_t> call_counter_ = thrust::device_vector{{0uz}};
    thrust::device_vector<device_type> device_object_ = thrust::device_vector{device_type{
        thrust::raw_pointer_cast(mocked_outputs_.data()),
        thrust::raw_pointer_cast(actual_inputs_.data()),
        n_,
        thrust::raw_pointer_cast(call_counter_.data())
    }};
};

template<typename Signature>
void PrintTo(const Mock<Signature>& mock, std::ostream* os) {
    const auto n = mock.call_counter();
    const auto actual_inputs = mock.actual_inputs();
    if (n == 0) {
        *os << "was not called";
    } else if (n > actual_inputs.size()) {
        *os << "was called too many times (" << n << "), the first " << actual_inputs.size() << " inputs were: " << ::testing::PrintToString(actual_inputs);
    } else {
        *os << "was called with " << n << " values: " << ::testing::PrintToString(mock.actual_inputs());
    }
}

MATCHER_P(IsCalledWith, expected_mock_inputs, std::string{"was called with "} + std::to_string(expected_mock_inputs.size()) + " values: " + ::testing::PrintToString(expected_mock_inputs)){
    return ::testing::Value(arg.call_counter(), ::testing::Eq(expected_mock_inputs.size()))
        && ::testing::Value(arg.actual_inputs(), ::testing::ElementsAreArray(expected_mock_inputs));
}

}
