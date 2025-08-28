// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#include <sparkplug/util/cuda_stream.cuh>
#include <sparkplug/util/cuda_check.cuh>

namespace sparkplug::util {
    CudaStream::CudaStream() {
        cuda_check("failed creating CUDA Stream", cudaStreamCreate, &cuda_stream_);
    }

    CudaStream::~CudaStream() {
        cuda_check("failed destroying CUDA Stream", cudaStreamDestroy, cuda_stream_);
    }

    CudaStream::operator cudaStream_t() const {
        return cuda_stream_;
    }

    void CudaStream::Synchronize() {
        cuda_check("CudaStream sync", cudaStreamSynchronize, cuda_stream_);
    }
}