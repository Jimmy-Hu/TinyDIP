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

// ---------------------------------------------------------
// Default Processor
// ---------------------------------------------------------

struct DefaultPixelProcessor
{
    template <TinyDIP::image_element_standard_floating_point_type T>
    constexpr T operator()(const T pixel) const
    {
        // Achieving maximum precision by enforcing long double arithmetic
        return static_cast<T>(pixel * 1.5234567890123456789L);
    }
};

// ---------------------------------------------------------
// Hardware-Friendly FIFO Stream Abstraction
// ---------------------------------------------------------

// Strictly avoids dynamic memory allocation by using std::array and a compile-time depth constraint.
template <TinyDIP::image_element_standard_floating_point_type T, std::size_t Depth>
class FIFO
{
private:
    std::array<T, Depth> buffer{};
    std::size_t head{};
    std::size_t tail{};
    std::size_t count{};

public:
    constexpr FIFO() = default;

    constexpr void push(const T item)
    {
        if (count < Depth)
        {
            buffer[tail] = item;
            tail = (tail + 1) % Depth;
            count++;
        }
    }

    constexpr T pop()
    {
        T item{};
        if (count > 0)
        {
            item = buffer[head];
            head = (head + 1) % Depth;
            count--;
        }
        return item;
    }

    constexpr bool empty() const
    {
        return count == 0;
    }
    
    constexpr bool full() const
    {
        return count == Depth;
    }
};

// ---------------------------------------------------------
// Algorithm Kernel
// ---------------------------------------------------------

// A generalized hardware-friendly kernel that processes an image array
// and pushes the results into the TinyDIP::FIFO stream.
template <
    typename ExecutionPolicy,
    TinyDIP::image_element_standard_floating_point_type T,
    std::size_t Depth,
    std::regular_invocable<T> PixelOp = DefaultPixelProcessor>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
void process_and_stream_image(
    ExecutionPolicy&& policy,
    const T* const input_data,
    T* const processed_data,
    const std::size_t size,
    FIFO<T, Depth>& output_stream,
    const PixelOp operation = PixelOp{})
{
    // OpenMP combined with execution policies to maximize multi-threading capability.
    // CIRCT will ignore these pragmas during hardware synthesis, but they provide
    // extreme performance during software-in-the-loop (SIL) testing.
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            std::transform(
                std::forward<ExecutionPolicy>(policy),
                input_data,
                input_data + size,
                processed_data,
                operation);
        }
    }

    // Explicit maximum iteration bound to ensure CIRCT can determine static loop bounds.
    constexpr std::size_t MAX_STREAM_ITERATIONS = 16384;
    std::size_t current_index{};

    // Data-dependent while loop with an explicit maximum iteration timeout counter
    // Avoids recursion and safely pushes data into the FIFO abstraction.
    while (current_index < size && current_index < MAX_STREAM_ITERATIONS)
    {
        output_stream.push(processed_data[current_index]);
        current_index++;
    }
}

// ---------------------------------------------------------
// Top-Level Hardware Entry Point (Synthesizable)
// ---------------------------------------------------------
#ifdef __POLYGEIST__

// This function forces Polygeist to instantiate the template so it generates actual MLIR hardware logic.
// We use a basic double type and sequential execution policy for the baseline hardware test.
extern "C" void hw_top_level_kernel(
    const double* const input_data,
    double* const processed_data,
    const std::size_t size,
    FIFO<double, 16>& output_stream)
{
    process_and_stream_image(
        std::execution::seq,
        input_data,
        processed_data,
        size,
        output_stream);
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

    // Static arrays avoid dynamic allocation during hardware kernel execution
    std::array<double, test_size> processed_data{};
    
    // Instantiate our custom FIFO with a depth sufficient for the test
    FIFO<double, 16> pixel_stream{};

    process_and_stream_image(
        std::execution::par_unseq,
        input_data.data(),
        processed_data.data(),
        test_size,
        pixel_stream);

    std::cout << "Stream outputs:\n";
    
    constexpr std::size_t MAX_READ_ITERATIONS = 1000;
    std::size_t read_count{};

    // Read out the FIFO stream
    while (!pixel_stream.empty() && read_count < MAX_READ_ITERATIONS)
    {
        std::cout << pixel_stream.pop() << '\n';
        read_count++;
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