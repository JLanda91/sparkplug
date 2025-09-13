// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once

template<typename Dep1, typename Dep2>
struct MultiDependencyFunctor {
    const Dep1* dep1_ = nullptr;
    const Dep2* dep2_ = nullptr;

    __device__ unsigned long operator()(unsigned n) const {
        if (n == 0) return 1234u;

        unsigned long result = 1;
        if (n < 7) result += (*dep1_)(n);
        if (n > 3) result += (*dep2_)(n);
        return result;
    }
};
