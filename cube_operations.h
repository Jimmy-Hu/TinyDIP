/* Developed by Jimmy Hu */

#ifndef CubeOperations_H
#define CubeOperations_H

#include <concepts>
#include <execution>
#include <fstream>
#include <numbers>
#include <string>
#include "base_types.h"
#include "basic_functions.h"
#include "cube.h"
#include "image.h"

namespace TinyDIP
{
    template<typename ElementT>
    constexpr bool is_width_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        return x.getSizeX() == y.getSizeX();
    }

    template<typename ElementT>
    constexpr bool is_width_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        return is_width_same(x, y) && is_width_same(y, z);
    }

    template<typename ElementT>
    constexpr bool is_height_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        return x.getSizeY() == y.getSizeY();
    }
    
    template<typename ElementT>
    constexpr bool is_height_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        return is_height_same(x, y) && is_height_same(y, z);
    }

    template<typename ElementT>
    constexpr bool is_depth_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        return x.getSizeZ() == y.getSizeZ();
    }

    template<typename ElementT>
    constexpr bool is_depth_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        return is_depth_same(x, y) && is_depth_same(y, z);
    }

    template<typename ElementT>
    constexpr bool is_size_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        return is_width_same(x, y) && is_height_same(x, y) && is_depth_same(x, y);
    }

    template<typename ElementT>
    constexpr bool is_size_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        return is_size_same(x, y) && is_size_same(y, z);
    }

    template<typename ElementT>
    constexpr void assert_width_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        assert(is_width_same(x, y));
    }

    template<typename ElementT>
    constexpr void assert_width_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        assert(is_width_same(x, y, z));
    }

    template<typename ElementT>
    constexpr void assert_height_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        assert(is_height_same(x, y));
    }

    template<typename ElementT>
    constexpr void assert_height_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        assert(is_height_same(x, y, z));
    }

    template<typename ElementT>
    constexpr void assert_depth_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        assert(is_depth_same(x, y));
    }

    template<typename ElementT>
    constexpr void assert_depth_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        assert(is_depth_same(x, y));
        assert(is_depth_same(y, z));
    }

    template<typename ElementT>
    constexpr void assert_size_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        assert_width_same(x, y);
        assert_height_same(x, y);
        assert_depth_same(x, y);
    }

    template<typename ElementT>
    constexpr void assert_size_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        assert_size_same(x, y);
        assert_size_same(y, z);
    }

    template<typename ElementT>
    constexpr void check_width_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        if (!is_width_same(x, y))
            throw std::runtime_error("Width mismatched!");
    }

    template<typename ElementT>
    constexpr void check_height_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        if (!is_height_same(x, y))
            throw std::runtime_error("Height mismatched!");
    }

    template<typename ElementT>
    constexpr void check_depth_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        if (!is_depth_same(x, y))
            throw std::runtime_error("Depth mismatched!");
    }

    template<typename ElementT>
    constexpr void check_size_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        check_width_same(x, y);
        check_height_same(x, y);
        check_depth_same(x, y);
    }

    //  voxelwiseOperation template function implementation
    template<std::size_t unwrap_level = 1, class... Args>
    constexpr static auto voxelwiseOperation(auto op, const Args&... inputs)
    {
        auto output = Cube(
            recursive_transform<unwrap_level>(
                [&](auto&& element1, auto&&... elements) 
                    {
                        return op(element1, elements...);
                    },
                inputs.getData()...),
            first_of(inputs...).getWidth(),
            first_of(inputs...).getHeight(),
            first_of(inputs...).getDepth());
        return output;
    }

    //  plus template function implementation
    template<class InputT>
    constexpr static Cube<InputT> plus(const Cube<InputT>& input1)
    {
        return input1;
    }
}

#endif
