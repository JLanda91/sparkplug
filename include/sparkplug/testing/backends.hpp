#pragma once

namespace sparkplug::testing {
    template<class T, bool IsDeviceSide>
    struct Backend {
        using type = T;
        static constexpr bool is_device_side = IsDeviceSide;
    };

    template<class T>
    using HostBackend = Backend<T, false>;

    template<class T>
    using DeviceBackend = Backend<T, true>;

}
