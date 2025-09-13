// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

#include <cuda_runtime_api.h>

namespace sparkplug::util::cuda {

class Stream {
public:
    Stream();

    ~Stream();

    operator cudaStream_t() const;

    void Synchronize();

private:
    cudaStream_t cuda_stream_{};
};

}