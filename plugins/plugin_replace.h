/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for replace */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //   extract_container_element type trait implementation
    template <typename T>
    struct extract_container_element
    {
        using type = TinyDIP::get_deep_scalar_t<T>;
    };

    template <typename ElemT>
    struct extract_container_element<TinyDIP::Image<ElemT>>
    {
        using type = ElemT;
    };

    template <std::ranges::input_range RangeT>
    requires (not TinyDIP::is_Image<RangeT>::value)
    struct extract_container_element<RangeT>
    {
        using type = std::ranges::range_value_t<RangeT>;
    };

    template <typename T>
    using extract_container_element_t = typename extract_container_element<std::remove_cvref_t<T>>::type;
}