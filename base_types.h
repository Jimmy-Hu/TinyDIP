/* Developed by Jimmy Hu */

#ifndef TINYDIP_BASE_TYPES_H  // base_types.h header guard, follow the suggestion from https://codereview.stackexchange.com/a/293832/231235
#define TINYDIP_BASE_TYPES_H

#include <cstdint>
#include <filesystem>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

namespace TinyDIP
{
    /**
     * @brief A tag dispatch type to select execution policy automatically at runtime.
     */
    struct auto_execution_policy {};

    // Create a convenient instance for users to import and use.
    inline constexpr auto_execution_policy auto_exec;

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

    //  RGB struct implementation
    struct RGB
    {
        std::uint8_t channels[3];

        inline RGB operator+(const RGB& input) const
        {
            return RGB{
                static_cast<std::uint8_t>(input.channels[0] + channels[0]),
                static_cast<std::uint8_t>(input.channels[1] + channels[1]),
                static_cast<std::uint8_t>(input.channels[2] + channels[2]) };
        }

        inline RGB operator-(const RGB& input) const
        {
            return RGB{
                static_cast<std::uint8_t>(channels[0] - input.channels[0]),
                static_cast<std::uint8_t>(channels[1] - input.channels[1]),
                static_cast<std::uint8_t>(channels[2] - input.channels[2]) };
        }

        friend std::ostream& operator<<(std::ostream& out, const RGB& _myStruct)
        {
            out << '{' << +_myStruct.channels[0] << ", " << +_myStruct.channels[1] << ", " << +_myStruct.channels[2] << '}';
            return out;
        }

        bool operator==(const RGB& other) const = default;
    };

    //  RGB_DOUBLE struct implementation
    struct RGB_DOUBLE
    {
        double channels[3];

        inline RGB_DOUBLE operator+(const RGB_DOUBLE& input) const
        {
            return RGB_DOUBLE{
                input.channels[0] + channels[0],
                input.channels[1] + channels[1],
                input.channels[2] + channels[2] };
        }

        inline RGB_DOUBLE operator-(const RGB_DOUBLE& input) const
        {
            return RGB_DOUBLE{
                channels[0] - input.channels[0],
                channels[1] - input.channels[1],
                channels[2] - input.channels[2] };
        }

        friend std::ostream& operator<<(std::ostream& out, const RGB_DOUBLE& _myStruct)
        {
            out << '{' << +_myStruct.channels[0] << ", " << +_myStruct.channels[1] << ", " << +_myStruct.channels[2] << '}';
            return out;
        }
    };

    using GrayScale = std::uint8_t;

    //  HSV struct implementation
    struct HSV
    {
        double channels[3];    //  Range: 0 <= H < 360, 0 <= S <= 1, 0 <= V <= 255

        inline HSV operator+(const HSV& input) const
        {
            return HSV{
                input.channels[0] + channels[0],
                input.channels[1] + channels[1],
                input.channels[2] + channels[2] };
        }

        inline HSV operator-(const HSV& input) const
        {
            return HSV{
                channels[0] - input.channels[0],
                channels[1] - input.channels[1],
                channels[2] - input.channels[2] };
        }

        friend std::ostream& operator<<(std::ostream& out, const HSV& _myStruct)
        {
            out << '{' << +_myStruct.channels[0] << ", " << +_myStruct.channels[1] << ", " << +_myStruct.channels[2] << '}';
            return out;
        }

        //  For applying as the key to std::map
        bool operator<(const HSV& other) const
        {
            // Compare channels lexicographically
            return std::tie(channels[0], channels[1], channels[2])
                < std::tie(other.channels[0], other.channels[1], other.channels[2]);
        }

        bool operator==(const HSV& other) const = default;
    };

    //  MultiChannel struct implementation
    template<class ElementT, std::size_t channel_count = 3>
    struct MultiChannel
    {
        std::array<ElementT, channel_count> channels;

        inline MultiChannel operator+(const MultiChannel& input) const
        {
            std::array<ElementT, channel_count> channels_output;
            for(std::size_t i = 0; i < channels.size(); ++i)
            {
                channels_output[i] = channels[i] + input.channels[i];
            }
            return MultiChannel{channels_output};
        }

        inline MultiChannel operator-(const MultiChannel& input) const
        {
            std::array<ElementT, channel_count> channels_output;
            for(std::size_t i = 0; i < channels.size(); ++i)
            {
                channels_output[i] = channels[i] - input.channels[i];
            }
            return MultiChannel{channels_output};
        }

        friend std::ostream& operator<<(std::ostream& out, const MultiChannel& _myStruct)
        {
            out << '{';
            for(std::size_t i = 0; i < channel_count; ++i)
            {
                out << +_myStruct.channels[i] << ", ";
            }
            out << '}';
            return out;
        }

        bool operator==(const MultiChannel& other) const = default;
    };

    //  Point struct implementation
    template<std::size_t dimension = 2>
    struct Point
    {
        std::array<std::size_t, dimension> p;

        inline Point operator+(const Point& input) const
        {
            std::array<std::size_t, dimension> new_array;
            for (std::size_t i = 0; i < p.size(); ++i)
            {
                new_array[i] = p[i] + input.p[i];
            }
            return Point{ new_array };
        }

        inline Point operator-(const Point& input) const
        {
            std::array<std::size_t, dimension> new_array;
            for (std::size_t i = 0; i < p.size(); ++i)
            {
                new_array[i] = p[i] - input.p[i];
            }
            return Point{ new_array };
        }

        friend std::ostream& operator<<(std::ostream& out, const Point& _myStruct)
        {
            out << '{';
            for (std::size_t i = 0; i < dimension; ++i)
            {
                out << +_myStruct.p[i] << ", ";
            }
            out << '}';
            return out;
        }
    };

    struct BMPIMAGE
    {
        std::filesystem::path FILENAME;

        unsigned int XSIZE;
        unsigned int YSIZE;
        std::uint8_t FILLINGBYTE;
        std::uint8_t* IMAGE_DATA;
    };
}
#endif
