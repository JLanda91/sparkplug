// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#include <sparkplug/util/cuda/stream.cuh>
#include <sparkplug/util/cuda/check_cuda_fn_error.cuh>

namespace sparkplug::util::cuda {
    Stream::Stream() {
        check_cuda_fn_error("failed creating CUDA Stream", cudaStreamCreate, &cuda_stream_);
    }

    Stream::~Stream() {
        check_cuda_fn_error("failed destroying CUDA Stream", cudaStreamDestroy, cuda_stream_);
    }

    Stream::operator cudaStream_t() const {
        return cuda_stream_;
    }

    void Stream::Synchronize() {
        check_cuda_fn_error("Stream sync", cudaStreamSynchronize, cuda_stream_);
    }
}