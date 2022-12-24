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
        return x.getWidth() == y.getWidth();
    }

    template<typename ElementT>
    constexpr bool is_width_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        return is_width_same(x, y) && is_width_same(y, z);
    }

    template<typename ElementT>
    constexpr bool is_height_same(const Cube<ElementT>& x, const Cube<ElementT>& y)
    {
        return x.getHeight() == y.getHeight();
    }
    
    template<typename ElementT>
    constexpr bool is_height_same(const Cube<ElementT>& x, const Cube<ElementT>& y, const Cube<ElementT>& z)
    {
        return is_height_same(x, y) && is_height_same(y, z);
    }
}

#endif
