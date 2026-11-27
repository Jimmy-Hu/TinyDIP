/* Developed by Jimmy Hu */

#ifndef TINYDIP_BASE_TYPES_H  // base_types.h header guard, follow the suggestion from https://codereview.stackexchange.com/a/293832/231235
#define TINYDIP_BASE_TYPES_H

#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#ifndef __POLYGEIST__
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <ostream>
#include <string>
#endif
#include <tuple>
#include <type_traits>
#include <utility>

namespace TinyDIP
{
    /**
     * @brief A tag dispatch type to select execution policy automatically at runtime.
     */
    struct auto_execution_policy {};

    // Create a convenient instance for users to import and use.
    inline constexpr auto_execution_policy auto_exec;

    namespace execution
    {
        enum class hardware_parallel_unroll_policy : int { tag };
        enum class hardware_pipelined_policy : int { tag };

        inline constexpr hardware_parallel_unroll_policy hw_unroll = hardware_parallel_unroll_policy::tag;
        inline constexpr hardware_pipelined_policy hw_pipe = hardware_pipelined_policy::tag;
    }

    enum BoundaryCondition {
        constant,
        mirror,
        replicate
    };

    template<typename T>
    concept image_element_standard_floating_point_type =
        std::same_as<double, T>
        or std::same_as<float, T>
        or std::same_as<long double, T>
        ;

    /**
     * get_underlying_real_type struct implementation
     * @brief A type trait to get the underlying real type.
     * For non-complex types, it's the type itself. For std::complex<T>, it's T.
     */
    template <typename T>
    struct get_underlying_real_type
    {
        using type = T;
    };

    template <typename T>
    struct get_underlying_real_type<std::complex<T>>
    {
        using type = T;
    };

    template<typename T>
    using get_underlying_real_type_t = typename get_underlying_real_type<T>::type;

    namespace base_types_detail
    {
        template<typename PixelT, std::size_t N, typename BinaryOp>
        [[nodiscard]] constexpr PixelT apply_channel3_op(
            const PixelT& lhs,
            const PixelT& rhs,
            const BinaryOp op
        ) noexcept
        {
            static_assert(N == 3);
            using ChanT = std::remove_cvref_t<decltype(lhs.channels[0])>;
            return PixelT{
                static_cast<ChanT>(op(lhs.channels[0], rhs.channels[0])),
                static_cast<ChanT>(op(lhs.channels[1], rhs.channels[1])),
                static_cast<ChanT>(op(lhs.channels[2], rhs.channels[2]))
            };
        }

        template<typename T, std::size_t N, typename BinaryOp>
        [[nodiscard]] constexpr std::array<T, N> apply_array_op(
            const std::array<T, N>& lhs,
            const std::array<T, N>& rhs,
            const BinaryOp op
        ) noexcept(noexcept(static_cast<T>(op(std::declval<const T&>(), std::declval<const T&>()))))
        {
            std::array<T, N> output{};
            for (std::size_t i = 0; i < N; ++i)
            {
                output[i] = static_cast<T>(op(lhs[i], rhs[i]));
            }
            return output;
        }

        #ifndef __POLYGEIST__
        template<typename OStreamT, typename ElemRange>
        constexpr std::ostream& print_braced_3(OStreamT& out, const ElemRange& elems)
        {
            out << '{' << +elems[0] << ", " << +elems[1] << ", " << +elems[2] << '}';
            return out;
        }

        template<typename OStreamT, typename ElemRange>
        constexpr std::ostream& print_braced_n(OStreamT& out, const ElemRange& elems, const std::size_t count)
        {
            out << '{';
            for (std::size_t i = 0; i < count; ++i)
            {
                out << +elems[i] << ", ";
            }
            out << '}';
            return out;
        }
        #endif
    }

    //  RGB struct implementation
    struct RGB
    {
        std::uint8_t channels[3];

        [[nodiscard]] constexpr RGB operator+(const RGB& input) const noexcept
        {
            return base_types_detail::apply_channel3_op<RGB, 3>(input, *this, std::plus<>{});
        }

        [[nodiscard]] constexpr RGB operator-(const RGB& input) const noexcept
        {
            return base_types_detail::apply_channel3_op<RGB, 3>(*this, input, std::minus<>{});
        }

        #ifndef __POLYGEIST__
        template<typename OStreamT = std::ostream>
        requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
        friend std::ostream& operator<<(OStreamT& out, const RGB& _myStruct)
        {
            return base_types_detail::print_braced_3(out, _myStruct.channels);
        }
        #endif

        constexpr bool operator==(const RGB& other) const noexcept = default;
    };

    //  RGB_DOUBLE struct implementation
    struct RGB_DOUBLE
    {
        double channels[3];

        [[nodiscard]] constexpr RGB_DOUBLE operator+(const RGB_DOUBLE& input) const noexcept
        {
            return base_types_detail::apply_channel3_op<RGB_DOUBLE, 3>(input, *this, std::plus<>{});
        }

        [[nodiscard]] constexpr RGB_DOUBLE operator-(const RGB_DOUBLE& input) const noexcept
        {
            return base_types_detail::apply_channel3_op<RGB_DOUBLE, 3>(*this, input, std::minus<>{});
        }

        #ifndef __POLYGEIST__
        template<typename OStreamT = std::ostream>
        requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
        friend std::ostream& operator<<(OStreamT& out, const RGB_DOUBLE& _myStruct)
        {
            return base_types_detail::print_braced_3(out, _myStruct.channels);
        }
        #endif

        constexpr bool operator==(const RGB_DOUBLE& other) const noexcept = default;
    };

    using GrayScale = std::uint8_t;

    //  HSV struct implementation
    struct HSV
    {
        double channels[3];    //  Range: 0 <= H < 360, 0 <= S <= 1, 0 <= V <= 255

        [[nodiscard]] constexpr HSV operator+(const HSV& input) const noexcept
        {
            return base_types_detail::apply_channel3_op<HSV, 3>(input, *this, std::plus<>{});
        }

        [[nodiscard]] constexpr HSV operator-(const HSV& input) const noexcept
        {
            return base_types_detail::apply_channel3_op<HSV, 3>(*this, input, std::minus<>{});
        }

        #ifndef __POLYGEIST__
        template<typename OStreamT = std::ostream>
        requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
        friend std::ostream& operator<<(OStreamT& out, const HSV& _myStruct)
        {
            return base_types_detail::print_braced_3(out, _myStruct.channels);
        }
        #endif

        //  For applying as the key to std::map
        [[nodiscard]] constexpr bool operator<(const HSV& other) const noexcept
        {
            return std::tie(channels[0], channels[1], channels[2])
                < std::tie(other.channels[0], other.channels[1], other.channels[2]);
        }

        constexpr bool operator==(const HSV& other) const noexcept = default;
    };

    //  MultiChannel struct implementation
    template<class ElementT, std::size_t channel_count = 3>
    struct MultiChannel
    {
        std::array<ElementT, channel_count> channels;

        [[nodiscard]] constexpr MultiChannel operator+(const MultiChannel& input) const
        {
            return MultiChannel{ base_types_detail::apply_array_op(channels, input.channels, std::plus<>{}) };
        }

        [[nodiscard]] constexpr MultiChannel operator-(const MultiChannel& input) const
        {
            return MultiChannel{ base_types_detail::apply_array_op(channels, input.channels, std::minus<>{}) };
        }

        #ifndef __POLYGEIST__
        template<typename OStreamT = std::ostream>
        requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
        friend std::ostream& operator<<(OStreamT& out, const MultiChannel& _myStruct)
        {
            return base_types_detail::print_braced_n(out, _myStruct.channels, channel_count);
        }
        #endif

        constexpr bool operator==(const MultiChannel& other) const = default;
    };

    //  Point struct implementation
    template<std::size_t dimension = 2>
    struct Point
    {
        std::array<std::size_t, dimension> p;

        [[nodiscard]] constexpr Point operator+(const Point& input) const noexcept
        {
            return Point{ base_types_detail::apply_array_op(p, input.p, std::plus<>{}) };
        }

        [[nodiscard]] constexpr Point operator-(const Point& input) const noexcept
        {
            return Point{ base_types_detail::apply_array_op(p, input.p, std::minus<>{}) };
        }

        #ifndef __POLYGEIST__
        template<typename OStreamT = std::ostream>
        requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
        friend std::ostream& operator<<(OStreamT& out, const Point& _myStruct)
        {
            return base_types_detail::print_braced_n(out, _myStruct.p, dimension);
        }
        #endif

        constexpr bool operator==(const Point& other) const noexcept = default;
    };

    #ifndef __POLYGEIST__
    struct BMPIMAGE
    {
        std::filesystem::path FILENAME;

        unsigned int XSIZE;
        unsigned int YSIZE;
        std::uint8_t FILLINGBYTE;
        std::uint8_t* IMAGE_DATA;
    };
    #endif
}
#endif
