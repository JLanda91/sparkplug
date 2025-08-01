#pragma once

template<typename Backend>
class Functor {
    const Backend* d_backend_;

public:
    explicit __device__ Functor(const Backend* backend) : d_backend_(backend) {}

    __device__ int operator()(int x) const {
        if (x < 0) {
            return 0;
        }
        if (x <= 1) {
            return 1;
        }
        return x * (*d_backend_)(x-1);
    }
};
