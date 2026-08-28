/* Developed by Jimmy Hu */

#ifndef TINYDIP_HW_ALGORITHMS_H
#define TINYDIP_HW_ALGORITHMS_H

#include <concepts>
#include <execution>
#include <fstream>
#include <future>
#include <numbers>
#include <string>
#include <thread>
#include "base_types.h"
#include "basic_functions.h"
#include "image.h"
#include "histogram.h"
#include "linear_algebra.h"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace std
{
    // Register custom types as valid execution policies via trait specialization
    template <>
    struct is_execution_policy<TinyDIP::execution::hardware_parallel_unroll_policy> : std::true_type {};

    template <>
    struct is_execution_policy<TinyDIP::execution::hardware_pipelined_policy> : std::true_type {};

    // Intercept std::transform for hardware parallel unroll
    template <class InputIt, class OutputIt, class UnaryOperation>
    constexpr OutputIt transform(
        TinyDIP::execution::hardware_parallel_unroll_policy,
        InputIt first1,
        InputIt last1,
        OutputIt d_first,
        UnaryOperation unary_op)
    {
        // Pragma to instruct MLIR affine/scf generation for full unrolling
        #pragma clang loop unroll(full)
        while (first1 != last1)
        {
            *d_first = unary_op(*first1);
            ++first1;
            ++d_first;
        }
        return d_first;
    }
}

namespace TinyDIP
{
    
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
    
}

#endif