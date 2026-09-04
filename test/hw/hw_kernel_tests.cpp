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

// Define the fractional precision (e.g., 16 bits for the fractional part)
constexpr std::uint32_t FRACTIONAL_BITS = 16;

// Convert 1.5234567890123456 to a fixed-point integer
// 1.5234567890123456 * (1 << 16) = 99841.28 -> 99841
constexpr std::uint64_t MULTIPLIER_FIXED = 99841;

// Use std::uint64_t instead of double for hardware synthesis
constexpr std::uint64_t default_pixel_processor(const std::uint64_t pixel)
{
    // Perform fixed-point multiplication and shift back to correct the scale
    return (pixel * MULTIPLIER_FIXED) >> FRACTIONAL_BITS;
}

//  process_and_stream_image template function implementation
// ---------------------------------------------------------
// Algorithm Kernel (Struct-Free & Pointer-Free)
// ---------------------------------------------------------
template <
    typename ExecutionPolicy,
    std::unsigned_integral T,
    std::size_t Depth>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
[[gnu::always_inline]]
inline void process_and_stream_image(
    [[maybe_unused]] const ExecutionPolicy policy,
    const T* const input_data,
    T* const processed_data,
    const std::size_t size,
    T* const output_stream_buffer,
    std::size_t& stream_count)
{
    #ifndef __POLYGEIST__
    #pragma omp parallel for
    for (std::size_t i = 0; i < size; ++i)
    {
        processed_data[i] = default_pixel_processor(input_data[i]);
    }
#else
    std::size_t i = 0;
    constexpr std::size_t MAX_SIZE = 16384;
    while ((i < size) & (i < MAX_SIZE))
    {
        processed_data[i] = default_pixel_processor(input_data[i]);
        i++;
    }
#endif

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
    const std::uint64_t* const input_data,
    std::uint64_t* const processed_data,
    std::uint64_t* const output_stream_buffer,
    std::size_t* const out_count,
    const std::size_t size)
{
    constexpr auto hw_policy = TinyDIP::execution::hardware_parallel_unroll_policy::tag;
    
    std::size_t internal_count{};

    process_and_stream_image<decltype(hw_policy), std::uint64_t, 16>(
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
template <typename T = std::uint64_t>
void test_process_and_stream_image()
{
    std::cout << "--- Starting test_process_and_stream_image ---\n";

    constexpr std::size_t test_size = 5;
    std::vector<T> input_data;
    input_data.reserve(test_size);

    for (std::size_t i{}; i < test_size; ++i)
    {
        input_data.emplace_back(static_cast<T>(i + 1));
    }

    std::array<T, test_size> processed_data{};
    
    std::array<T, 16> pixel_stream_buffer{};
    std::size_t stream_count{};

    process_and_stream_image<decltype(std::execution::par_unseq), T, 16>(
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

    std::cout << "--- Unit test finished ---\n";
}

int main()
{
    test_process_and_stream_image();
    return EXIT_SUCCESS;
}

#endif // __POLYGEIST__