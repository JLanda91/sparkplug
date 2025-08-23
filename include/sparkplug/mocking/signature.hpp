#pragma once

namespace sparkplug::mocking {
    template<typename InputT, typename ReturnT>
    struct Signature {
        using input_t = InputT;
        using return_t = ReturnT;
    };

}