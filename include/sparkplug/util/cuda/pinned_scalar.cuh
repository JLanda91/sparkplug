// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa

#pragma once

#include <cuda_runtime_api.h>

#include "check_cuda_fn_error.cuh"

namespace sparkplug::util::cuda {

template<typename T>
class PinnedScalar {
public:
    PinnedScalar() {
        check_cuda_fn_error("PinnedScalar pinned host alloc", cudaMallocHost<T>, &h_scalar_, sizeof(T), 0);
        check_cuda_fn_error("PinnedScalar device alloc", cudaMalloc<T>, &d_scalar_, sizeof(T));
    }

    template<typename ... Args>
    explicit PinnedScalar(std::in_place_t, Args&& ... args) : PinnedScalar() {
        std::construct_at(h_scalar_, std::forward<Args>(args)...);
        is_emplaced_ = true;
    }

    PinnedScalar& operator=(const T& arg) {
        *h_scalar_ = arg;
        is_emplaced_ = false;
        return *this;
    }

    PinnedScalar(const PinnedScalar& arg) = delete;
    PinnedScalar& operator=(const PinnedScalar& arg) = delete;

    PinnedScalar(PinnedScalar&& arg) = default;
    PinnedScalar& operator=(PinnedScalar&& arg) = default;

    ~PinnedScalar() {
        if (is_emplaced_) {
            std::destroy_at(h_scalar_);
        }
        
        check_cuda_fn_error("PinnedScalar pinned host free", cudaFreeHost, h_scalar_);
        check_cuda_fn_error("PinnedScalar device free", cudaFree, d_scalar_);
    }

    template<typename ... Args>
    void emplace(Args&& ... args) {
        if (is_emplaced_) {
            std::destroy_at(h_scalar_);
        }

        std::construct_at(h_scalar_, std::forward<Args>(args)...);
        is_emplaced_ = true;
    }
    
    [[nodiscard]] auto HostPtr() const {
        return h_scalar_;
    }

    [[nodiscard]] auto DevicePtr() const {
        return d_scalar_;
    }

    void ToHostAsync(const cudaStream_t& stream) const {
        check_cuda_fn_error("PinnedScalar D->H async", cudaMemcpyAsync, h_scalar_, d_scalar_, sizeof(T), cudaMemcpyDeviceToHost, stream);
    }

    void ToDevAsync(const cudaStream_t& stream) const {
        check_cuda_fn_error("PinnedScalar H->D async", cudaMemcpyAsync, d_scalar_, h_scalar_, sizeof(T), cudaMemcpyHostToDevice, stream);
    }

private:
    bool is_emplaced_ = false;
    T* h_scalar_ {};
    T* d_scalar_ {};
};

}