#pragma once

namespace sparkplug::util {

class CudaStream {
public:
    CudaStream();

    ~CudaStream();

    operator cudaStream_t() const;

private:
    cudaStream_t cuda_stream_{};
};

}