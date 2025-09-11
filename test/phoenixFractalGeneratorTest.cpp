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
    * @brief Generates a pre-computed color map for fractal rendering.
    * @return An std::array of 256 RGB colors.
    */
[[nodiscard]] constexpr std::array<TinyDIP::RGB, 256> generate_fractal_color_map() noexcept
{
    std::array<TinyDIP::RGB, 256> color_map{};
    for (std::size_t i = 0; i < 256; ++i)
    {
        // Use a smooth coloring formula based on sine waves for a psychedelic effect.
        constexpr double freq_r = 0.1;
        constexpr double freq_g = 0.15;
        constexpr double freq_b = 0.2;
        constexpr double phase_r = 3.0;
        constexpr double phase_g = 2.5;
        constexpr double phase_b = 1.0;

        const double t = static_cast<double>(i);

        const auto r = static_cast<std::uint8_t>(std::sin(freq_r * t + phase_r) * 127.5 + 127.5);
        const auto g = static_cast<std::uint8_t>(std::sin(freq_g * t + phase_g) * 127.5 + 127.5);
        const auto b = static_cast<std::uint8_t>(std::sin(freq_b * t + phase_b) * 127.5 + 127.5);

        color_map[i] = TinyDIP::RGB{ r, g, b };
    }
    return color_map;
}

/**
    * @brief Maps the number of iterations to an RGB color using a pre-computed color map.
    * @param iterations The number of iterations completed.
    * @param max_iterations The maximum number of iterations allowed.
    * @param color_map A pre-computed table of colors.
    * @return A TinyDIP::RGB struct representing the calculated color.
    */
[[nodiscard]] constexpr TinyDIP::RGB map_iterations_to_color(const std::size_t iterations, const std::size_t max_iterations, const std::array<TinyDIP::RGB, 256>& color_map) noexcept
{
    if (iterations >= max_iterations)
    {
        return TinyDIP::RGB{ 0, 0, 0 }; // Black for points inside the set
    }

    // Look up the color from the pre-computed map.
    // The modulo wraps the iteration count to the size of the map.
    return color_map[iterations % 256];
}

template<class ExecutionPolicy, std::floating_point FloatingPoint = double>
requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
[[nodiscard]] TinyDIP::Image<TinyDIP::RGB> generate_phoenix(
    ExecutionPolicy&& policy,
    const std::size_t image_width,
    const std::size_t image_height,
    const FloatingPoint x_min,
    const FloatingPoint x_max,
    const FloatingPoint y_min,
    const FloatingPoint y_max,
    const FloatingPoint p, // Phoenix constant P
    const FloatingPoint q, // Phoenix constant Q
    const std::size_t max_iterations)
{
    TinyDIP::Image<TinyDIP::RGB> image(image_width, image_height);
    const FloatingPoint width_float = image_width > 1 ? static_cast<FloatingPoint>(image_width - 1) : 1.0;
    const FloatingPoint height_float = image_height > 1 ? static_cast<FloatingPoint>(image_height - 1) : 1.0;
    const FloatingPoint x_range = x_max - x_min;
    const FloatingPoint y_range = y_max - y_min;

    static const auto color_map = generate_fractal_color_map();
    auto proxy = image.pixels_with_coordinates();

    std::for_each(
        std::forward<ExecutionPolicy>(policy),
        proxy.begin(),
        proxy.end(),
        [&](auto&& pixel_tuple)
        {
            auto& [pixel_value, px, py] = pixel_tuple;

            auto z_real = x_range * static_cast<FloatingPoint>(px) / width_float + x_min;
            auto z_imag = y_range * static_cast<FloatingPoint>(py) / height_float + y_min;

            auto z_prev_real = static_cast<FloatingPoint>(0);
            auto z_prev_imag = static_cast<FloatingPoint>(0);

            std::size_t iteration = 0;
            while (z_real * z_real + z_imag * z_imag <= static_cast<FloatingPoint>(4) && iteration < max_iterations)
            {
                const auto z_n_real = z_real;
                const auto z_n_imag = z_imag;

                // z_{n+1} = z_n^2 + p + q*z_{n-1}
                z_real = z_n_real * z_n_real - z_n_imag * z_n_imag + p + q * z_prev_real;
                z_imag = static_cast<FloatingPoint>(2) * z_n_real * z_n_imag + q * z_prev_imag;
                    
                z_prev_real = z_n_real;
                z_prev_imag = z_n_imag;

                iteration++;
            }
                
            pixel_value = map_iterations_to_color(iteration, max_iterations, color_map);
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

void generate_phoenix_set()
{
    // --- Fractal Parameters ---
    constexpr std::size_t width = 1200;
    constexpr std::size_t height = 800;
    constexpr std::size_t max_iterations = 255;
    constexpr double x_min = -1.2;
    constexpr double x_max = 1.2;
    constexpr double y_min = -1.2;
    constexpr double y_max = 1.2;
    // Classic constants for the Phoenix fractal
    constexpr double p = 0.5667;
    constexpr double q = -0.5;

    std::cout << "--- Generating Phoenix Set (p = " << p << ", q = " << q << ") ---\n";

    // --- Sequential Execution ---
    std::cout << "Executing sequentially...\n";
    auto start_seq = std::chrono::high_resolution_clock::now();
    auto phoenix_image_seq = generate_phoenix(
        std::execution::seq, width, height, x_min, x_max, y_min, y_max, p, q, max_iterations
    );
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_seq = end_seq - start_seq;
    std::cout << "Sequential generation took: " << diff_seq.count() << " s\n";
    TinyDIP::bmp_write("phoenix_sequential", phoenix_image_seq);

    // --- Parallel Execution ---
    std::cout << "Executing in parallel...\n";
    auto start_par = std::chrono::high_resolution_clock::now();
    auto phoenix_image_par = generate_phoenix(
        std::execution::par_unseq, width, height, x_min, x_max, y_min, y_max, p, q, max_iterations
    );
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_par = end_par - start_par;
    std::cout << "Parallel generation took: " << diff_par.count() << " s\n";
    TinyDIP::bmp_write("phoenix_parallel", phoenix_image_par);

    std::cout << "Phoenix performance improvement: " << (diff_seq.count() / diff_par.count()) << "x\n";
    std::cout << "-------------------------------------------\n\n";
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';
    generate_phoenix_set();
    return EXIT_SUCCESS;
}