// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

namespace sparkplug::util {

class CudaStream {
public:
    CudaStream();

    ~CudaStream();

    operator cudaStream_t() const;

    void Synchronize();

private:
    cudaStream_t cuda_stream_{};
};

}