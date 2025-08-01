#pragma once

namespace sparkplug::mocking {
    template<typename InputT, typename ReturnT>
    struct MockSignature {
        using input_t = InputT;
        using return_t = ReturnT;
    };

#define SPARKPLUG_SIGNATURE(InputT, ReturnT) \
    sparkplug::mocking::MockSignature<InputT, ReturnT>

}