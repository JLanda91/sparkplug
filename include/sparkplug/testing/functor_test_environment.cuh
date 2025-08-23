#pragma once

#include <thrust/device_vector.h>

#include <sparkplug/util/pinned_host_vector.cuh>
#include <sparkplug/util/cuda_stream.cuh>
#include "detail/deduced_signature.hpp"

namespace sparkplug::testing {
template<typename Functor>
class FunctorTestEnvironment {
    using Signature = typename detail::DeducedSignature<Functor>::type;
public:
    auto GetFunctorPtr() const {
        return thrust::raw_pointer_cast(d_functor_.data());
    }

    void SetFunctor(const Functor& f) {
        h_functor_[0] = f;
        util::cuda_check("FunctorTestEnvironment::SetFunctor H->D copy", cudaMemcpyAsync, thrust::raw_pointer_cast(d_functor_.data()), thrust::raw_pointer_cast(h_functor_.data()), sizeof(Functor), cudaMemcpyHostToDevice, test_driver_stream_);
        util::cuda_check("FunctorTestEnvironment::GetReturnValue stream sync", cudaStreamSynchronize, test_driver_stream_);
    }

    auto GetInputPtr() const {
        return thrust::raw_pointer_cast(d_input_.data());
    }

    void SetInput(const typename Signature::input_t& arg) {
        h_input_[0] = arg;
        util::cuda_check("FunctorTestEnvironment::SetInput H->D copy", cudaMemcpyAsync, thrust::raw_pointer_cast(d_input_.data()), thrust::raw_pointer_cast(h_input_.data()), sizeof(typename Signature::input_t), cudaMemcpyHostToDevice, test_driver_stream_);
        util::cuda_check("FunctorTestEnvironment::GetReturnValue stream sync", cudaStreamSynchronize, test_driver_stream_);
    }

    auto GetReturnPtr() {
        return thrust::raw_pointer_cast(d_return_.data());
    }

    auto GetReturnValue() {
        util::cuda_check("FunctorTestEnvironment::GetReturnValue D->H copy", cudaMemcpyAsync, thrust::raw_pointer_cast(h_return_.data()), thrust::raw_pointer_cast(d_return_.data()), sizeof(typename Signature::return_t), cudaMemcpyDeviceToHost, test_driver_stream_);
        util::cuda_check("FunctorTestEnvironment::GetReturnValue stream sync", cudaStreamSynchronize, test_driver_stream_);
        return h_return_[0];
    }

    auto& TestDriverStream() {
        return test_driver_stream_;
    }

    auto& ProxyStream() {
        return proxy_stream_;
    }

private:
    util::CudaStream test_driver_stream_ = {};
    util::CudaStream proxy_stream_ = {};

    util::pinned_host_vector<Functor> h_functor_ = util::pinned_host_vector<Functor>(1);
    util::pinned_host_vector<typename Signature::input_t> h_input_ = util::pinned_host_vector<typename Signature::input_t>(1);
    util::pinned_host_vector<typename Signature::return_t> h_return_ = util::pinned_host_vector<typename Signature::return_t>(1);

    thrust::device_vector<Functor> d_functor_ = thrust::device_vector<Functor>(1);
    thrust::device_vector<typename Signature::input_t> d_input_ = thrust::device_vector<typename Signature::input_t>(1);
    thrust::device_vector<typename Signature::return_t> d_return_ = thrust::device_vector<typename Signature::return_t>(1);
};

namespace detail {
    template<typename Functor>
    inline FunctorTestEnvironment<Functor>* functor_test_env = nullptr;
}
}