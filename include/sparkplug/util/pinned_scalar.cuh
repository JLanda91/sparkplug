//  SPDX-License-Identifier: Apache-2.0
//
//  Copyright 2025 Jasper Landa
//

#pragma once

#include "cuda_check.cuh"

namespace sparkplug::util {

template<typename T>
class PinnedScalar {
public:
    PinnedScalar() {
        cuda_check("PinnedScalar pinned host alloc", cudaMallocHost<T>, &h_scalar_, sizeof(T), 0);
        cuda_check("PinnedScalar device alloc", cudaMalloc<T>, &d_scalar_, sizeof(T));
    }

    explicit PinnedScalar(const T& arg) : PinnedScalar() {
        *h_scalar_ = arg;
    }

    PinnedScalar& operator=(const T& arg) {
        *h_scalar_ = arg;
        return *this;
    }

    PinnedScalar(const PinnedScalar& arg) = delete;
    PinnedScalar& operator=(const PinnedScalar& arg) = delete;

    PinnedScalar(PinnedScalar&& arg) = default;
    PinnedScalar& operator=(PinnedScalar&& arg) = default;

    ~PinnedScalar() {
        cuda_check("PinnedScalar pinned host free", cudaFreeHost, h_scalar_);
        cuda_check("PinnedScalar device free", cudaFree, d_scalar_);
    }

    [[nodiscard]] auto HostPtr() const {
        return h_scalar_;
    }

    [[nodiscard]] auto DevicePtr() const {
        return d_scalar_;
    }

    void ToHostAsync(const cudaStream_t& stream) const {
        cuda_check("PinnedScalar D->H async", cudaMemcpyAsync, h_scalar_, d_scalar_, sizeof(T), cudaMemcpyDeviceToHost, stream);
    }

    void ToDevAsync(const cudaStream_t& stream) const {
        cuda_check("PinnedScalar H->D async", cudaMemcpyAsync, d_scalar_, h_scalar_, sizeof(T), cudaMemcpyHostToDevice, stream);
    }

private:
    T* h_scalar_ {};
    T* d_scalar_ {};
};

}