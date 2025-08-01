#pragma once

#include <gtest/gtest.h>

#include <sparkplug/testing/detail/deduced_signature.hpp>

namespace sparkplug::testing {
template <template <typename> typename Functor, typename Backend>
class FunctorTest : public ::testing::Test {
public:
    using functor_type = Functor<Backend>;
    using backend_signature = typename detail::DeducedSignature<functor_type>::type;
protected:

private:

};
}