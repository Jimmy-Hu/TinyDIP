#include <algorithm>
#include <array>
#include <concepts>
#include <cstdlib>
#include <execution>
#include <type_traits>

// ---------------------------------------------------------
// Hide Software-Only Headers from Polygeist
// ---------------------------------------------------------
#ifndef __POLYGEIST__
#include <iostream>
#include <vector>
#endif

#include "../../base_types.h"
#include "../../basic_functions.h"
#include "../../hw_algorithms.h"
#include "../../image.h"
#include "../../timer.h"

constexpr double default_pixel_processor(const double pixel)
{
    return pixel * 1.5234567890123456789;
}

// ---------------------------------------------------------
// Algorithm Kernel (Struct-Free & Pointer-Free)
// ---------------------------------------------------------
template <
    typename ExecutionPolicy,
    TinyDIP::image_element_standard_floating_point_type T,
    std::size_t Depth>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
[[gnu::always_inline]]
inline void process_and_stream_image(
    const ExecutionPolicy policy,
    const T* const input_data,
    T* const processed_data,
    const std::size_t size,
    T* const output_stream_buffer,
    std::size_t& stream_count)
{
    // Abandon std::transform at the hardware boundary.
    // Direct C++ loop ensures NO function pointers or structs are passed as arguments.
    // Seamlessly hybridize OpenMP for CPU and clang unroll for HW.
#ifndef __POLYGEIST__
    #pragma omp parallel for
#else
    #pragma clang loop unroll(full)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        // Direct function call synthesizes perfectly into a hardwired multiplier.
        processed_data[i] = default_pixel_processor(input_data[i]);
    }

    constexpr std::size_t MAX_STREAM_ITERATIONS = 16384;
    std::size_t current_index{};
    
    std::size_t tail{};

    while (current_index < size && current_index < MAX_STREAM_ITERATIONS)
    {
        if (stream_count < Depth)
        {
            output_stream_buffer[tail] = processed_data[current_index];
            tail = (tail + 1) % Depth;
            stream_count++;
        }
        current_index++;
    }
}

// ---------------------------------------------------------
// Top-Level Hardware Entry Point (Synthesizable)
// ---------------------------------------------------------
#ifdef __POLYGEIST__

extern "C" void hw_top_level_kernel(
    const double* const input_data,
    double* const processed_data,
    double* const output_stream_buffer,
    std::size_t* const out_count,
    const std::size_t size)
{
    constexpr auto hw_policy = TinyDIP::execution::hardware_parallel_unroll_policy::tag;
    
    std::size_t internal_count{};

    process_and_stream_image<decltype(hw_policy), double, 16>(
        hw_policy,
        input_data,
        processed_data,
        size,
        output_stream_buffer,
        internal_count);
        
    *out_count = internal_count;
}

#endif // __POLYGEIST__


// ---------------------------------------------------------
// Unit Test and Verification
// ---------------------------------------------------------

#ifndef __POLYGEIST__

void test_process_and_stream_image()
{
    std::cout << "--- Starting test_process_and_stream_image ---\n";

    constexpr std::size_t test_size = 5;
    std::vector<double> input_data;
    input_data.reserve(test_size);

    for (std::size_t i{}; i < test_size; ++i)
    {
        input_data.emplace_back(static_cast<double>(i + 1));
    }

    std::array<double, test_size> processed_data{};
    
    std::array<double, 16> pixel_stream_buffer{};
    std::size_t stream_count{};

    process_and_stream_image<decltype(std::execution::par_unseq), double, 16>(
        std::execution::par_unseq,
        input_data.data(),
        processed_data.data(),
        test_size,
        pixel_stream_buffer.data(),
        stream_count);

    std::cout << "Stream outputs:\n";
    for(std::size_t i = 0; i < stream_count; ++i)
    {
        std::cout << pixel_stream_buffer[i] << '\n';
    }

    // Expected Output:
    // 1.52346
    // 3.04691
    // 4.57037
    // 6.09383
    // 7.61728

    std::cout << "--- Unit test finished ---\n";
}

int main()
{
    test_process_and_stream_image();
    return EXIT_SUCCESS;
}

#endif // __POLYGEIST__