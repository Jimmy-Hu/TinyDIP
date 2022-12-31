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

    
}

#endif
