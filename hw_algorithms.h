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