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
}

#endif
