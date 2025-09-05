/* Developed by Jimmy Hu */

#include <chrono>
#include <execution>
#include <map>
#include <omp.h>
#include <span>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

/**
    * @brief Maps the number of iterations from a fractal calculation to an RGB color.
    * @param iterations The number of iterations completed.
    * @param max_iterations The maximum number of iterations allowed.
    * @return A TinyDIP::RGB struct representing the calculated color.
    */
[[nodiscard]] constexpr TinyDIP::RGB map_iterations_to_color(const std::size_t iterations, const std::size_t max_iterations) noexcept
{
    if (iterations >= max_iterations)
    {
        return TinyDIP::RGB{ 0, 0, 0 }; // Black for points inside the set
    }

    // Use a smooth coloring formula based on sine waves for a psychedelic effect.
    // These constants can be tweaked to produce different color palettes.
    constexpr double freq_r = 0.1;
    constexpr double freq_g = 0.15;
    constexpr double freq_b = 0.2;
    constexpr double phase_r = 3.0;
    constexpr double phase_g = 2.5;
    constexpr double phase_b = 1.0;

    const double t = static_cast<double>(iterations);

    const auto r = static_cast<std::uint8_t>(sin(freq_r * t + phase_r) * 127.5 + 127.5);
    const auto g = static_cast<std::uint8_t>(sin(freq_g * t + phase_g) * 127.5 + 127.5);
    const auto b = static_cast<std::uint8_t>(sin(freq_b * t + phase_b) * 127.5 + 127.5);

    return TinyDIP::RGB{ r, g, b };
}

/**
    * @brief Generates a Mandelbrot set fractal image.
    *
    * @tparam ExecutionPolicy The execution policy (e.g., std::execution::seq, std::execution::par).
    * @tparam FloatingPoint The floating-point type for calculations (e.g., double, long double).
    * @param policy The execution policy instance.
    * @param image_width The width of the output image in pixels.
    * @param image_height The height of the output image in pixels.
    * @param x_min The minimum value of the real component (complex plane).
    * @param x_max The maximum value of the real component (complex plane).
    * @param y_min The minimum value of the imaginary component (complex plane).
    * @param y_max The maximum value of the imaginary component (complex plane).
    * @param max_iterations The maximum number of iterations for each point.
    * @return An Image<TinyDIP::RGB> containing the Mandelbrot set.
    */
template<class ExecutionPolicy, std::floating_point FloatingPoint = double>
requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
[[nodiscard]] TinyDIP::Image<TinyDIP::RGB> generate_mandelbrot(
    ExecutionPolicy&& policy,
    const std::size_t image_width,
    const std::size_t image_height,
    const FloatingPoint x_min,
    const FloatingPoint x_max,
    const FloatingPoint y_min,
    const FloatingPoint y_max,
    const std::size_t max_iterations)
{
    TinyDIP::Image<TinyDIP::RGB> image(image_width, image_height);
    const std::size_t total_pixels = image.count();

    // Create a view of all pixel indices from 0 to total_pixels - 1.
    auto pixel_indices = std::ranges::views::iota(std::size_t{0}, total_pixels);

    // Process all pixels, potentially in parallel.
    std::for_each(
        std::forward<ExecutionPolicy>(policy),
        std::ranges::begin(pixel_indices),
        std::ranges::end(pixel_indices),
        [&](const std::size_t pixel_index)
        {
            // 1. Map 1D pixel index to 2D image coordinates
            const std::size_t px = pixel_index % image_width;
            const std::size_t py = pixel_index / image_width;

            // 2. Map image coordinates to a point in the complex plane
            const auto x0 = static_cast<FloatingPoint>(px) / (image_width - 1) * (x_max - x_min) + x_min;
            const auto y0 = static_cast<FloatingPoint>(py) / (image_height - 1) * (y_max - y_min) + y_min;

            // 3. Perform Mandelbrot iteration
            auto x = static_cast<FloatingPoint>(0);
            auto y = static_cast<FloatingPoint>(0);
            std::size_t iteration = 0;

            while (x * x + y * y <= static_cast<FloatingPoint>(4) && iteration < max_iterations)
            {
                const auto xtemp = x * x - y * y + x0;
                y = static_cast<FloatingPoint>(2) * x * y + y0;
                x = xtemp;
                iteration++;
            }

            // 4. Map the result to a color and set the pixel
            image.set(pixel_index) = map_iterations_to_color(iteration, max_iterations);
        }
    );

    return image;
}

//  remove_extension Function Implementation
//  Copy from: https://stackoverflow.com/a/6417908/6667035
std::string remove_extension(const std::string& filename)
{
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';
    // --- Fractal Parameters ---
    constexpr std::size_t width = 1200;
    constexpr std::size_t height = 800;
    constexpr std::size_t max_iterations = 255;

    // Region of the complex plane to render
    constexpr double x_min = -2.0;
    constexpr double x_max = 1.0;
    constexpr double y_min = -1.0;
    constexpr double y_max = 1.0;

    // --- Sequential Execution ---
    std::cout << "Generating Mandelbrot set (Sequential)...\n";
    auto start_seq = std::chrono::high_resolution_clock::now();
    
    auto mandelbrot_image_seq = generate_mandelbrot(
        std::execution::seq,
        width, height,
        x_min, x_max,
        y_min, y_max,
        max_iterations
    );
    
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_seq = end_seq - start_seq;
    std::cout << "Sequential generation took: " << diff_seq.count() << " s\n";

    TinyDIP::bmp_write("mandelbrot_sequential", mandelbrot_image_seq);

    // --- Parallel Execution ---
    std::cout << "Generating Mandelbrot set (Parallel)...\n";
    auto start_par = std::chrono::high_resolution_clock::now();
    
    auto mandelbrot_image_par = generate_mandelbrot(
        std::execution::par,
        width, height,
        x_min, x_max,
        y_min, y_max,
        max_iterations
    );

    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_par = end_par - start_par;
    std::cout << "Parallel generation took: " << diff_par.count() << " s\n";

    TinyDIP::bmp_write("mandelbrot_parallel", mandelbrot_image_par);

    std::cout << "Performance improvement: " << (diff_seq.count() / diff_par.count()) << "x\n";

    return EXIT_SUCCESS;
}