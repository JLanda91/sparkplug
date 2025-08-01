#pragma once

#include <sparkplug/mocking/mock_signature.hpp>

namespace sparkplug::testing::detail {

    template<typename, typename = void>
    struct DeducedSignature {
        static constexpr bool value = false;
    };

    // Specialization for callable types
    template<typename F>
    struct DeducedSignature<F, std::void_t<decltype(&F::operator())>>
    {
    private:

        template<typename ClassT, typename ReturnT, typename ArgT>
        static std::tuple<ReturnT, ArgT> deduce(ReturnT(ClassT::*)(ArgT) const);

        using result = decltype(deduce(decltype(&F::operator()){}));

        using input_t  = std::remove_cv_t<std::remove_reference_t<std::tuple_element_t<1, result>>>;
        using return_t = std::remove_reference_t<std::tuple_element_t<0, result>>;

    public:
        static constexpr bool value = true;
        using type = mocking::MockSignature<input_t, return_t>;
    };

}