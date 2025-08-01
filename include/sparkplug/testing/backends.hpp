#pragma once

namespace sparkplug::testing {
    template<class T, bool IsMock>
    struct Backend {
        using type = T;
        static constexpr bool is_mock = IsMock;
    };

    template<class T>
    using RealBackend = Backend<T, false>;

    template<class T>
    using MockBackend = Backend<T, true>;

    template<typename T>
    class Widget {

    };


}
