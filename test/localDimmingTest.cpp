//  Developed by Jimmy Hu

#include <algorithm>
#include <cassert>
#include <execution>
#include <filesystem>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h"

//  RGB_max Function Implementation
static auto RGB_max(const TinyDIP::Image<TinyDIP::RGB>& input_image)
{
    return TinyDIP::pixelwise_transform([](auto&& each_pixel)
            {
                auto max_value = std::ranges::max(each_pixel.channels);
                TinyDIP::RGB new_pixel{ max_value, max_value, max_value };
                return new_pixel;
            }, input_image);
}

//  RGB_max_parallel Function Implementation
static auto RGB_max_parallel(const TinyDIP::Image<TinyDIP::RGB>& input_image)
{
    return TinyDIP::pixelwise_transform(std::execution::par_unseq, [](auto&& each_pixel)
            {
                auto max_value = std::ranges::max(each_pixel.channels);
                TinyDIP::RGB new_pixel{ max_value, max_value, max_value };
                return new_pixel;
            }, input_image);
}

//  belongs_bin_index Function Implementation
template<std::ranges::random_access_range RangeT, class ElementT>
requires(std::equality_comparable<std::ranges::range_value_t<RangeT>> and
         std::equality_comparable<ElementT> and
         std::convertible_to<std::ranges::range_value_t<RangeT>, ElementT>)
static auto belongs_bin_index(const RangeT& thresholds, const ElementT& value)
{
    auto it = std::ranges::lower_bound(thresholds, value);
    return static_cast<int>(std::distance(std::ranges::begin(thresholds), it)) - 1;
}

//  gray2gamma_single_pixel Template Function Implementation
template<
    TinyDIP::arithmetic PixelType = int,
    std::ranges::random_access_range GammaRange1,
    std::ranges::random_access_range GammaRange2>
requires(std::equality_comparable<std::ranges::range_value_t<GammaRange1>> and
         std::equality_comparable<std::ranges::range_value_t<GammaRange2>>)
static auto gray2gamma_single_pixel(
    const PixelType& pixel_value,
    const GammaRange1& gamma_range1,
    const GammaRange2& gamma_range2
)
{
    auto bin_index = belongs_bin_index(gamma_range1, pixel_value);
    return std::clamp(
            gamma_range2[bin_index] + (((gamma_range2[bin_index + 1] - gamma_range2[bin_index]) * (pixel_value - gamma_range1[bin_index])) >> (static_cast<int>(std::log2(gamma_range1[bin_index + 1] - gamma_range1[bin_index])))),
            0, static_cast<int>(std::pow(2, 12) - 1)
    );
}

//  gray2gamma Template Function Implementation
//  Output is TinyDIP::Image<TinyDIP::RGB_DOUBLE>, 12 bits
template<
    class ExecutionPolicy,
    std::ranges::random_access_range GammaRange1,
    std::ranges::random_access_range GammaRange2>
requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> and
         std::equality_comparable<std::ranges::range_value_t<GammaRange1>> and
         std::equality_comparable<std::ranges::range_value_t<GammaRange2>>)
static auto gray2gamma(
    ExecutionPolicy&& policy,
    const TinyDIP::Image<TinyDIP::RGB>& input_image,
    const GammaRange1& gamma_range1,
    const GammaRange2& gamma_range2
)
{
    return TinyDIP::pixelwise_transform(std::forward<ExecutionPolicy>(policy), [&](auto&& each_pixel)
            {
                auto pixel_value = each_pixel.channels[0];
                auto final_pixel_value = gray2gamma_single_pixel(pixel_value, gamma_range1, gamma_range2);
                //final_pixel_value = final_pixel_value >> 4;         //  12 bits to 8 bits
                TinyDIP::RGB_DOUBLE new_pixel{ static_cast<double>(final_pixel_value), static_cast<double>(final_pixel_value), static_cast<double>(final_pixel_value) };
                return new_pixel;
            }, input_image);
}

//  clamp12bit Template Function Implementation
template<TinyDIP::arithmetic ElementT>
constexpr static auto clamp12bit(const ElementT input)
{
    return std::clamp(input, static_cast<ElementT>(0), static_cast<ElementT>(std::pow(2, 12) - 1));
}

//  gamma_table_generator Template Function Implementation
template<TinyDIP::arithmetic FloatingType = double>
static auto gamma_table_generator(
    const FloatingType gamma = 2.2,
    const int input_maximum = 255,
    const int output_bits = 12
)
{
    const int NUM_NODES = 33; // 0 to 256, step of 8, total 33 nodes
    const int STEP = 8;
    const FloatingType output_max = std::pow(static_cast<FloatingType>(2.0), static_cast<FloatingType>(output_bits)) - 1.0;

    std::vector<int> nodes_x(NUM_NODES);
    std::vector<int> baseline_y(NUM_NODES);
    
    // Pre-calculate the perfect, lossless ideal curve as the absolute reference for Minimax evaluation.
    std::vector<FloatingType> ideal_y(input_maximum + 1);
    for (int x = 0; x <= input_maximum; ++x)
    {
        FloatingType normalized_x = static_cast<FloatingType>(x) / static_cast<FloatingType>(input_maximum);
        ideal_y[x] = std::pow(normalized_x, gamma) * output_max;
    }

    // Generate baseline nodes using standard mathematical calculation and rounding.
    for (int i = 0; i < NUM_NODES; ++i)
    {
        int x = i * STEP;
        nodes_x[i] = x;
        
        // Note: When x=256, normalized_x > 1.0. This generates a mathematically extrapolated value (> 4095).
        // This is a strictly required technique to ensure the hardware's piece-wise linear (PWL) 
        // interpolation works correctly without dropping values for the final segment (x=248~255).
        FloatingType normalized_x = static_cast<FloatingType>(x) / static_cast<FloatingType>(input_maximum);
        baseline_y[i] = static_cast<int>(std::round(std::pow(normalized_x, gamma) * output_max));
    }

    std::vector<int> optimized_y = baseline_y;

    // The boundary nodes (x=0, x=256) are strictly fixed to ensure boundary safety and continuity.
    // We only apply the Minimax tuning optimization to the intermediate nodes.
    for (int i = 1; i < NUM_NODES - 1; ++i)
    {
        FloatingType best_error = std::numeric_limits<FloatingType>::max();
        int best_val = baseline_y[i];

        int x_start = nodes_x[i-1];
        int x_mid   = nodes_x[i];
        int x_end   = nodes_x[i+1];

        int y_start = optimized_y[i-1]; // Left node is already optimized in the previous iteration
        int y_end   = baseline_y[i+1];  // Right node is still the initial baseline value

        // Test tweaking the current anchor point by -3 to +3 to minimize the chord error
        for (int tweak = -3; tweak <= 3; ++tweak)
        {
            int test_y = baseline_y[i] + tweak;
            if (test_y < 0) continue; // Basic boundary protection
            
            FloatingType max_err = 0.0;
            
            for (int x = x_start; x <= x_mid; ++x)
            {
                if (x > input_maximum) break; // Prevent out-of-bounds access for ideal_y array
                FloatingType y_interp = y_start + static_cast<FloatingType>(test_y - y_start) * (x - x_start) / (x_mid - x_start);
                FloatingType err = std::abs(y_interp - ideal_y[x]);
                if (err > max_err) max_err = err;
            }
            
            for (int x = x_mid; x <= x_end; ++x)
            {
                if (x > input_maximum) break;
                FloatingType y_interp = test_y + static_cast<FloatingType>(y_end - test_y) * (x - x_mid) / (x_end - x_mid);
                FloatingType err = std::abs(y_interp - ideal_y[x]);
                if (err > max_err) max_err = err;
            }

            // Minimax core logic: If this tweak reduces the maximum chord error in the affected segments, 
            // record it as the current optimal value.
            if (max_err < best_error)
            {
                best_error = max_err;
                best_val = test_y;
            }
        }
        
        // Commit the best value found by the Minimax optimizer to the final output array
        optimized_y[i] = best_val;
    }

    return std::make_pair(nodes_x, optimized_y);
}

//  calculate_block_count Function Implementation
static auto calculate_block_count(
    const std::size_t block_size_x = 60,
    const std::size_t block_size_y = 60,
    const std::size_t x_extension_pixel_count = 30,
    const std::size_t y_extension_pixel_count = 30
)
{
    auto width = block_size_x + x_extension_pixel_count * 2;
    auto height = block_size_y + y_extension_pixel_count * 2;
    return width * height;
}

//  calculate_reg_avg_div_inv Template Function Implementation
template<TinyDIP::arithmetic FloatingType = double>
static auto calculate_block_count_inv(
    const std::size_t block_size_x = 60,
    const std::size_t block_size_y = 60,
    const std::size_t x_extension_pixel_count = 30,
    const std::size_t y_extension_pixel_count = 30,
    const std::size_t representation_bit_count = 19
)
{
    auto block_count = calculate_block_count(block_size_x, block_size_y, x_extension_pixel_count, y_extension_pixel_count);
    auto block_count_inv = static_cast<FloatingType>(1) / static_cast<FloatingType>(block_count);
    return static_cast<int>(std::round(std::pow(2, representation_bit_count) * block_count_inv));
}

//  weight_of_sum_of_histogram Template Function Implementation
template<TinyDIP::arithmetic FloatingType = double>
static auto weight_of_sum_of_histogram(
    const std::size_t block_size_x = 60,
    const std::size_t block_size_y = 60,
    const std::size_t x_extension_pixel_count = 30,
    const std::size_t y_extension_pixel_count = 30,
    const std::size_t representation_bit_count = 12
)
{
    auto block_count = calculate_block_count(block_size_x, block_size_y, x_extension_pixel_count, y_extension_pixel_count);
    auto half_block_count = static_cast<FloatingType>(block_count) / static_cast<FloatingType>(2);
    return static_cast<int>(std::round(
        (static_cast<FloatingType>((std::pow(2, representation_bit_count) - 1)) / half_block_count) * std::pow(2, representation_bit_count)
    ));
}

//  get_real_size_PWM_image Template Function Implementation
template<
    class ExecutionPolicy,
    class ElementT,
    std::ranges::random_access_range HistogramWeightRange,
    TinyDIP::arithmetic FloatingType = double
>
requires((std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>) and
         (TinyDIP::arithmetic<std::ranges::range_value_t<HistogramWeightRange>>))
static auto get_real_size_PWM_image(
    ExecutionPolicy&& policy,
    const TinyDIP::Image<ElementT>& input_img,
    const FloatingType gamma = 1.0,
    const std::size_t light_bead_width = 22,
    const std::size_t light_bead_height = 8,
    const std::size_t x_extension_pixel_count = 41,
    const std::size_t y_extension_pixel_count = 45,
    const int estimated_average_offset = 20,
    const bool adp_adj_hist_weight_en = true,       //  adaptive adjustment histogram weight
    const bool local_dimming_en = true,
    const std::size_t output_scale_x = 1,
    const std::size_t output_scale_y = 1,
    const HistogramWeightRange& histogram_weight = {},
    const std::string_view local_dimming_mode = "adaptive_blending",
    std::ostream& os = std::cout
)
{
    auto RGB_max_result = RGB_max(input_img);
    auto gamma_table = gamma_table_generator<FloatingType>(gamma);
    auto gray2gamma_12bits = gray2gamma(std::forward<ExecutionPolicy>(policy), RGB_max_result, gamma_table.first, gamma_table.second);
    auto split_overlap_output = TinyDIP::split_overlap(
        std::forward<ExecutionPolicy>(policy),
        gray2gamma_12bits,
        light_bead_width,
        light_bead_height,
        x_extension_pixel_count,
        y_extension_pixel_count
    );
    auto split_overlap_max = TinyDIP::recursive_transform<2>(
        std::forward<ExecutionPolicy>(policy),
        [&](const auto& each_block)
        {
            return static_cast<int>(TinyDIP::max(TinyDIP::getRplane(each_block)));
        }, split_overlap_output);
    const int representation_bit_count = 19;
    auto split_overlap_estimated_average = TinyDIP::recursive_transform<2>(
        std::forward<ExecutionPolicy>(policy),
        [&](const auto& each_block)
        {
            return static_cast<int>((((
                        static_cast<unsigned long long>(TinyDIP::sum(TinyDIP::getRplane(each_block))) *
                        static_cast<unsigned long long>(calculate_block_count_inv(
                            input_img.getWidth() / light_bead_width,
                            input_img.getHeight() / light_bead_height,
                            x_extension_pixel_count,
                            y_extension_pixel_count,
                            representation_bit_count
                        ))
                     ) >> (representation_bit_count - 1)) + 1) >> 1);
        }, split_overlap_output);
    auto split_overlap_histogram = TinyDIP::recursive_transform<2>(
        std::forward<ExecutionPolicy>(policy),
        [&](const auto& each_block)
        {
            auto each_block_r = TinyDIP::getRplane(each_block);
            std::array<int, 32> histogram_output{};
            for (std::size_t y = 0; y < each_block_r.getHeight(); ++y)
            {
                for (std::size_t x = 0; x < each_block_r.getWidth(); ++x)
                {
                    ++histogram_output[static_cast<int>(each_block_r.at_without_boundary_check(x, y)) >> 7];
                }
            }
            return histogram_output;
        }, split_overlap_output);
    if (false)
    {
        //  Print Value for Debugging
        os << "split_overlap_max[0][0] = " << +split_overlap_max[0][0] << '\n';
        os << "split_overlap_estimated_average[0][0] = " << +split_overlap_estimated_average[0][0] << '\n';
        TinyDIP::recursive_print(split_overlap_histogram[0][0]);
    }

    auto output_image = TinyDIP::concat(TinyDIP::recursive_transform<2>(
        [&](const auto& local_maximum, const auto& local_estimated_average, const auto& local_histogram)
        {
            auto sum_of_histogram = 0;
            if (adp_adj_hist_weight_en)
            {
                for (int index = (local_estimated_average >> 7 + 1); index < std::ranges::size(local_histogram); ++index)
                {
                    sum_of_histogram += std::min(
                        (local_histogram[index] * 
                        (histogram_weight[
                            std::clamp(index - std::max(local_estimated_average >> 7, 24), 0, 7)
                        ] + 8)) >> 3
                        , (1 << 18) - 1);
                }
            }
            else
            {
                auto selected_local_histogram { std::span{local_histogram}.subspan((local_estimated_average >> 7 + 1), local_histogram.size() - (local_estimated_average >> 7 + 1)) };
                sum_of_histogram = std::reduce(
                    std::forward<ExecutionPolicy>(policy),
                    std::ranges::cbegin(selected_local_histogram),
                    std::ranges::cend(selected_local_histogram),
                    0
                );
            }
            
            const int representation_bit_count2 = 12;
            int final_adptive_weight = clamp12bit(clamp12bit((
                    static_cast<int>((sum_of_histogram * weight_of_sum_of_histogram(
                            input_img.getWidth() / light_bead_width,
                            input_img.getHeight() / light_bead_height,
                            x_extension_pixel_count,
                            y_extension_pixel_count,
                            representation_bit_count2
                        ) + std::pow(2, representation_bit_count2 - 1))) >> representation_bit_count2
                )) + std::invoke(
                [](const int maximum, const int average, const bool light_spot_protect_en = false)
                {
                    if (!light_spot_protect_en)
                        return 0;
                    return std::max((maximum >> 1) - ((average >> 1) + 0), 0);
                },
                local_maximum, local_estimated_average
            ));
            
            std::map<std::string, int> local_dimming_modes;
            local_dimming_modes.insert(std::make_pair("local_maximum", local_maximum));
            local_dimming_modes.insert(std::make_pair("estimated_average", local_estimated_average));
            local_dimming_modes.insert(std::make_pair("adaptive_blending", 
                clamp12bit(
                    (static_cast<int>((std::min(local_estimated_average + estimated_average_offset, local_maximum)
                    * (std::pow(2, 12) - final_adptive_weight) + local_maximum * final_adptive_weight + std::pow(2, 11))) >> 12)
                )
            ));
            
            TinyDIP::Image<TinyDIP::RGB> output_subimage(std::size_t{ 1 }, std::size_t{ 1 });
            auto pixel_value = static_cast<std::uint8_t>((
                (!local_dimming_en)?
                (static_cast<int>(std::pow(2.0, 12.0)) - 1):
                local_dimming_modes[std::string(local_dimming_mode)]
            ) >> 4); //  Make pixel_value 8 bits
            TinyDIP::RGB output_pixel{ pixel_value, pixel_value, pixel_value };
            output_subimage.at_without_boundary_check(0, 0) = output_pixel;
            return output_subimage;
        },
        split_overlap_max, split_overlap_estimated_average, split_overlap_histogram
    ));
    output_image = TinyDIP::resize_nearest_neighbor(output_image, output_image.getWidth() * output_scale_x, output_image.getHeight() * output_scale_y);
    return output_image;
}

//  localDimmingTest Template Function Implementation
template<
    class ExecutionPolicy,
    TinyDIP::arithmetic FloatingType = double
>
requires std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>
static auto localDimmingTest(
    ExecutionPolicy&& policy,
    const std::filesystem::path& input_path,
    const std::string_view output_path,
    const FloatingType gamma,
    const int output_scale_x,
    const int output_scale_y,
    const std::string_view local_dimming_mode = "adaptive_blending"
)
{
    TinyDIP::Image<TinyDIP::RGB> input_img(0, 0);
    if (input_path.extension() == ".bmp")
    {
        input_img = TinyDIP::bmp_read(input_path.string().c_str(), true);
    }
    else
    {
        input_img = TinyDIP::pnm::read(std::forward<ExecutionPolicy>(policy), input_path.string().c_str());
    }
    
    const std::array<int, 8> histogram_weight = {0, 8, 16, 24, 32, 40, 48, 56};
    auto real_size_PWM_image = get_real_size_PWM_image(
        std::forward<ExecutionPolicy>(policy),
        input_img,
        gamma,
        18,
        32,
        static_cast<std::size_t>(0),
        static_cast<std::size_t>(0),
        0,
        true,       //  adaptive adjustment histogram weight
        true,
        output_scale_x,
        output_scale_y,
        histogram_weight,
        local_dimming_mode,
        std::cout
    );
    TinyDIP::bmp_write(std::string(output_path).c_str(), real_size_PWM_image);
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc = " << std::to_string(argc) << '\n';
    if(argc == 2)
    {
        std::filesystem::path input_path = std::string(argv[1]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << '\n';
            return EXIT_SUCCESS;
        }
        localDimmingTest(std::execution::par_unseq, input_path, std::string("localDimmingTest"), 2.2, 1, 1);
    }
    else if  (argc == 3)
    {
        std::filesystem::path input_path = std::string(argv[1]);
        std::filesystem::path output_path = std::string(argv[2]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << '\n';
            return EXIT_SUCCESS;
        }
        std::filesystem::path path_without_extension = output_path.parent_path() / output_path.stem();
        localDimmingTest(std::execution::par_unseq, input_path, path_without_extension.string(), 2.2, 1, 1);
    }
    else if  (argc == 6)
    {
        std::filesystem::path input_path = std::string(argv[1]);
        std::filesystem::path output_path = std::string(argv[2]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << '\n';
            return EXIT_SUCCESS;
        }
        auto gamma = std::stod(argv[3]);
        auto output_scale_x = std::atoi(argv[4]);
        auto output_scale_y = std::atoi(argv[5]);
        std::filesystem::path path_without_extension = output_path.parent_path() / output_path.stem();
        localDimmingTest(std::execution::par_unseq, input_path, path_without_extension.string(), gamma, output_scale_x, output_scale_y);
    }
    else if  (argc == 7)
    {
        std::filesystem::path input_path = std::string(argv[1]);
        std::filesystem::path output_path = std::string(argv[2]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << '\n';
            return EXIT_SUCCESS;
        }
        auto gamma = std::stod(argv[3]);
        auto output_scale_x = std::atoi(argv[4]);
        auto output_scale_y = std::atoi(argv[5]);
        auto local_dimming_mode = std::string(argv[6]);
        std::filesystem::path path_without_extension = output_path.parent_path() / output_path.stem();
        localDimmingTest(std::execution::par_unseq, input_path, path_without_extension.string(), gamma, output_scale_x, output_scale_y, local_dimming_mode);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " <input_image_path> [output_image_path] [gamma] [output_scale_x] [output_scale_y] [\"local_maximum\", \"estimated_average\", \"adaptive_blending\"]\n";
    }
    return EXIT_SUCCESS;
}



