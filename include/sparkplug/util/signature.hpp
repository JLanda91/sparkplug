// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Jasper Landa


#pragma once


#include <sparkplug/util/concepts/callable.hpp>

namespace sparkplug::util {

template<typename ReturnT, typename ArgT>
struct Signature {
    using return_type = ReturnT;
    using arg_type = ArgT;
};

template<concepts::Callable F>
class deduced_signature {
    template<typename ClassT, typename ReturnT, typename ArgT>
    static std::tuple<ReturnT, ArgT> deduce(ReturnT(ClassT::*)(ArgT) const);

    using result = decltype(deduce(decltype(&F::operator()){}));

    using arg_type  = std::remove_cvref_t<std::tuple_element_t<1, result>>;
    using return_type = std::remove_cvref_t<std::tuple_element_t<0, result>>;

public:
    using type = Signature<return_type, arg_type>;
};

template<concepts::Callable F>
using deduced_signature_t = deduced_signature<F>::type;

}