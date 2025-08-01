#pragma once

namespace sparkplug::mocking::detail {

template<typename Signature>
class DeviceSideMock {
public:
    __device__ DeviceSideMock(
        const typename Signature::return_t* mocked_outputs,
        typename Signature::input_t* actual_inputs,
        std::size_t num_expected_inputs,
        std::size_t* call_counter)
        : mocked_outputs_(mocked_outputs)
          , actual_inputs_(actual_inputs)
          , num_expected_inputs_(num_expected_inputs)
          , call_counter_(call_counter){}

    __device__ auto operator()(const typename Signature::input_t& x) const -> typename Signature::return_t {
        const auto call_num = (*call_counter_)++;
        if ( call_num >= num_expected_inputs_) {
            return {};
        }
        actual_inputs_[call_num] = x;
        return mocked_outputs_[call_num];
    }

private:
    const typename Signature::return_t* mocked_outputs_;
    typename Signature::input_t* actual_inputs_;
    std::size_t num_expected_inputs_;
    std::size_t* call_counter_;
};

}
