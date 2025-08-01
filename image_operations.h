/* Developed by Jimmy Hu */

#ifndef TINYDIP_IMAGE_OPERATIONS_H
#define TINYDIP_IMAGE_OPERATIONS_H

#include <concepts>
#include <execution>
#include <fstream>
#include <numbers>
#include <string>
#include "base_types.h"
#include "basic_functions.h"
#include "image.h"
#include "histogram.h"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace TinyDIP
{
    template<typename T>
    concept image_element_standard_floating_point_type =
        std::same_as<double, T>
        or std::same_as<float, T>
        or std::same_as<long double, T>
        ;

    //  all_of template function implementation
    template<typename ElementT, class UnaryPredicate>
    constexpr auto all_of(const Image<ElementT>& input, UnaryPredicate p)
    {
        return std::ranges::all_of(std::ranges::begin(input.getImageData()), std::ranges::end(input.getImageData()), p);
    }

    //  all_positive template function implementation
    template<typename ElementT>
    constexpr auto all_positive(const Image<ElementT>& input)
    {
        return all_of(input, [](const auto& element) { return element > 0; });
    }

    template<typename ElementT>
    constexpr bool is_width_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        return x.getWidth() == y.getWidth();
    }

    template<typename ElementT>
    constexpr bool is_width_same(const Image<ElementT>& x, const Image<ElementT>& y, const Image<ElementT>& z)
    {
        return is_width_same(x, y) && is_width_same(y, z);
    }

    template<typename ElementT>
    constexpr bool is_height_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        return x.getHeight() == y.getHeight();
    }

    template<typename ElementT>
    constexpr bool is_height_same(const Image<ElementT>& x, const Image<ElementT>& y, const Image<ElementT>& z)
    {
        return is_height_same(x, y) && is_height_same(y, z);
    }
    
    template<typename ElementT>
    constexpr bool is_size_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        return is_width_same(x, y) && is_height_same(x, y);
    }

    template<typename ElementT>
    constexpr bool is_size_same(const Image<ElementT>& x, const Image<ElementT>& y, const Image<ElementT>& z)
    {
        return is_size_same(x, y) && is_size_same(y, z);
    }

    template<typename ElementT>
    constexpr void assert_width_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        assert(is_width_same(x, y));
    }

    template<typename ElementT>
    constexpr void assert_width_same(const Image<ElementT>& x, const Image<ElementT>& y, const Image<ElementT>& z)
    {
        assert(is_width_same(x, y, z));
    }

    template<typename ElementT>
    constexpr void assert_height_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        assert(is_height_same(x, y));
    }

    template<typename ElementT>
    constexpr void assert_height_same(const Image<ElementT>& x, const Image<ElementT>& y, const Image<ElementT>& z)
    {
        assert(is_height_same(x, y, z));
    }

    template<typename ElementT>
    constexpr void assert_size_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        assert_width_same(x, y);
        assert_height_same(x, y);
    }

    template<typename ElementT>
    constexpr void assert_size_same(const Image<ElementT>& x, const Image<ElementT>& y, const Image<ElementT>& z)
    {
        assert_size_same(x, y);
        assert_size_same(y, z);
    }

    template<typename ElementT>
    constexpr void check_width_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        if (!is_width_same(x, y))
            throw std::runtime_error("Width mismatched!");
    }

    template<typename ElementT>
    constexpr void check_height_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        if (!is_height_same(x, y))
            throw std::runtime_error("Height mismatched!");
    }

    //  check_size_same template function implementation
    template<typename ElementT>
    constexpr void check_size_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        if(x.getSize() != y.getSize())
            throw std::runtime_error("Size mismatched!");
    }

    //  zeros template function implementation
    template<typename ElementT, std::same_as<std::size_t>... Sizes>
    constexpr static auto zeros(Sizes... sizes)
    {
        auto output = Image<ElementT>(sizes...);
        return output;
    }

    //  zeros template function implementation
    template<typename ElementT, std::ranges::input_range Sizes>
    requires (std::same_as<std::ranges::range_value_t<Sizes>, std::size_t>) or
             (std::same_as<std::ranges::range_value_t<Sizes>, int>)
    constexpr static auto zeros(const Sizes& input)
    {
        return Image<ElementT>(input);
    }

    //  ones template function implementation
    template<typename ElementT, std::same_as<std::size_t>... Sizes>
    constexpr static auto ones(Sizes... sizes)
    {
        auto output = zeros<ElementT>(sizes...);
        output.setAllValue(1);
        return output;
    }

    //  nan template function implementation
    template<typename ElementT = double, std::same_as<std::size_t>... Sizes>
    constexpr static auto nan(Sizes... sizes)
    {
        auto output = zeros<ElementT>(sizes...);
        output.setAllValue(std::numeric_limits<double>::quiet_NaN());
        return output;
    }

    //  linear_index_to_indices template function implementation
    //  Helper function to convert linear index to N-dimensional indices
    template<std::ranges::input_range Sizes>
    requires(std::same_as<std::ranges::range_value_t<Sizes>, std::size_t>)
    constexpr static std::vector<std::size_t> linear_index_to_indices(
        const std::size_t linearIndex,
        const Sizes& strides,
        const Sizes& sizes)
    {
        const std::size_t dim = std::ranges::size(sizes);
        std::vector<std::size_t> indices(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            indices[i] = (linearIndex / strides[i]) % sizes[i];
        }
        return indices;
    }

    //  linear_index_to_indices template function implementation
    template<std::ranges::input_range Sizes>
    requires(std::same_as<std::ranges::range_value_t<Sizes>, std::size_t>)
    constexpr static std::vector<std::size_t> linear_index_to_indices(
        std::size_t linear_idx,
        const Sizes& sizes
    )
    {
        std::vector<std::size_t> indices;
        std::size_t stride = 1;
        for (auto& s : sizes) {
            indices.emplace_back((linear_idx / stride) % s);
            stride *= s;
        }
        return indices;
    }

    //  generate template function implementation
    template<std::ranges::input_range Sizes, typename F>
    requires(((std::same_as<std::ranges::range_value_t<Sizes>, std::size_t>) or
              (std::same_as<std::ranges::range_value_t<Sizes>, int>)) and
             (std::invocable<F&>))
    constexpr static auto generate(F gen, const Sizes& sizes)
    {
        using ElementT = std::invoke_result_t<F>;
        auto count = std::reduce(std::ranges::cbegin(sizes), std::ranges::cend(sizes), 1, std::multiplies());
        std::vector<ElementT> element_vector(count);
        std::ranges::generate(element_vector, gen);
        Image<ElementT> image(element_vector, sizes);
        return image;
    }

    //  generate template function implementation
    //  https://codereview.stackexchange.com/a/295600/231235
    template<typename F, class... Sizes>
    requires((std::invocable<F&>) and
             ((std::same_as<Sizes, std::size_t>&&...) or
              (std::same_as<Sizes, int>&&...)))
    constexpr static auto generate(F gen, Sizes... sizes)
    {
        return generate(gen, std::array<std::size_t, sizeof...(Sizes)>{ static_cast<std::size_t>(sizes)...});
    }

    //  rand template function implementation
    template<image_element_standard_floating_point_type ElementT = double, typename Urbg, class... Sizes>
    requires(std::uniform_random_bit_generator<std::remove_reference_t<Urbg>> and
             ((std::same_as<Sizes, std::size_t>&&...) or
              (std::same_as<Sizes, int>&&...)))
    constexpr static auto rand(Urbg&& urbg, Sizes... sizes)
    {
        //  Reference: https://stackoverflow.com/a/23143753/6667035
        //  Reference: https://codereview.stackexchange.com/a/294739/231235
        auto dist = std::uniform_real_distribution<ElementT>{};
        return generate([&dist, &urbg]() { return dist(urbg); }, sizes...);
    }

    //  rand template function implementation
    template<image_element_standard_floating_point_type ElementT = double, class... Size>
    requires((std::same_as<Size, std::size_t>&&...) or
             (std::same_as<Size, int>&&...))
    inline auto rand(Size... size)
    {
        return rand<ElementT>(std::mt19937{std::random_device{}()}, size...);
    }

    //  rand template function implementation
    template<image_element_standard_floating_point_type ElementT = double, typename Urbg>
    requires std::uniform_random_bit_generator<std::remove_reference_t<Urbg>>
    constexpr auto rand(Urbg&& urbg)
    {
        auto dist = std::uniform_real_distribution<ElementT>{};
        return Image<ElementT>(std::vector{ dist(urbg) }, 1, 1);
    }

    //  rand template function implementation
    template<image_element_standard_floating_point_type ElementT = double>
    inline auto rand()
    {
        return rand<ElementT>(std::mt19937{std::random_device{}()});
    }

    //  randi template function implementation
    //  function that can handle everything, this one calls `generate()`
    template<std::integral ElementT = int, typename Urbg, class... Sizes>
    requires((std::uniform_random_bit_generator<std::remove_reference_t<Urbg>>) and
             ((std::same_as<Sizes, std::size_t>&&...) or
              (std::same_as<Sizes, int>&&...)))
    constexpr static auto randi(Urbg&& urbg, std::pair<ElementT, ElementT> min_and_max, Sizes... sizes)
    {
        auto dist = std::uniform_int_distribution<ElementT>{ min_and_max.first, min_and_max.second };
        if constexpr (sizeof...(Sizes) == 0)
        {
            
            return generate([&dist, &urbg]() { return dist(urbg); }, std::size_t{ 1 });
        }
        else
        {
            return generate([&dist, &urbg]() { return dist(urbg); }, sizes...);
        }
    }

    // randi template function implementation
    template<std::integral ElementT = int, class... Size>
    requires((std::same_as<Size, std::size_t>&&...) or
             (std::same_as<Size, int>&&...))
    inline auto randi(std::pair<ElementT, ElementT> min_and_max, Size... size)
    {
        return randi<ElementT>(std::mt19937{std::random_device{}()}, min_and_max, size...);
    }

    // randi template function implementation
    template<std::integral ElementT = int, class... Size>
    requires((std::same_as<Size, std::size_t>&&...) or
             (std::same_as<Size, int>&&...))
    inline auto randi(ElementT max, Size... size)
    {
        return randi<ElementT>(std::mt19937{ std::random_device{}() }, std::pair<ElementT, ElementT>{static_cast<ElementT>(1), max}, size...);
    }

    //  randi template function implementation
    template<std::integral ElementT = int, typename Urbg>
    requires std::uniform_random_bit_generator<std::remove_reference_t<Urbg>>
    constexpr auto randi(Urbg&& urbg, ElementT max)
    {
        return randi<ElementT>(std::forward<Urbg>(urbg), std::pair<ElementT, ElementT>{static_cast<ElementT>(1), max});
    }

    // randi template function implementation
    template<std::integral ElementT = int>
    inline auto randi(ElementT max)
    {
        return randi<ElementT>(std::mt19937{std::random_device{}()}, max);
    }

    //  generate_complex_image template function implementation
    template<class ElementT>
    requires(std::floating_point<ElementT> || std::integral<ElementT>)
    constexpr static auto generate_complex_image(const Image<ElementT>& real, const Image<ElementT>& imaginary)
    {
        if (real.getSize() != imaginary.getSize())
        {
            throw std::runtime_error("Size mismatched!");
        }
        return pixelwiseOperation(
            [&](auto&& real_element, auto&& imaginary_element)
            {
                return std::complex{ real_element, imaginary_element };
            },
            real,
            imaginary
        );
    }

    //  conv2 template function implementation
    template<typename ElementT>
    requires(std::floating_point<ElementT> || std::integral<ElementT> || is_complex<ElementT>::value)
    constexpr static auto conv2(const Image<ElementT>& x, const Image<ElementT>& y, bool is_size_same = false)
    {
        Image<ElementT> output(x.getWidth() + y.getWidth() - 1, x.getHeight() + y.getHeight() - 1);
        for (std::size_t y1 = 0; y1 < x.getHeight(); ++y1) {
            auto* x_row = &(x.at(std::size_t{ 0 }, y1));
            for (std::size_t y2 = 0; y2 < y.getHeight(); ++y2) {
                auto* y_row = &(y.at(std::size_t{ 0 }, y2));
                auto* out_row = &(output.at(std::size_t{ 0 }, y1 + y2));
                for (std::size_t x1 = 0; x1 < x.getWidth(); ++x1) {
                    for (std::size_t x2 = 0; x2 < y.getWidth(); ++x2) {
                        out_row[x1 + x2] += x_row[x1] * y_row[x2];
                    }
                }
            }
        }
        if(is_size_same)
        {
            output = subimage(output, x.getWidth(), x.getHeight(), static_cast<double>(output.getWidth()) / 2.0, static_cast<double>(output.getHeight()) / 2.0);
        }
        return output;
    }

    //  conv2 template function implementation
    template<typename ElementT, typename ElementT2>
    requires (((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>)) &&
              (std::floating_point<ElementT2> || std::integral<ElementT2> || is_complex<ElementT2>::value))
    constexpr static auto conv2(const Image<ElementT>& input1, const Image<ElementT2>& input2, bool is_size_same = false)
    {
        return apply_each(input1, [&](auto&& planes) { return conv2(planes, input2, is_size_same); });
    }

    namespace impl {
        //  convolution_detail template function implementation
        //  This function is a recursive implementation of convolution.
        //  It performs convolution on the input image and kernel, and stores the result in the output image.
        //  Test: https://godbolt.org/z/ohe1EcWz5
        template<class ExecutionPolicy, typename ImageT, typename KernelT,
                 typename F = std::multiplies<std::common_type_t<ImageT, KernelT>>>
        requires((std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
                 && std::regular_invocable<F, ImageT, KernelT>)
        constexpr static void convolution_detail(
                    ExecutionPolicy&& execution_policy,
                    const Image<ImageT>& image,
                    const Image<KernelT>& kernel,
                    Image<ImageT>& output,
                    std::size_t level = 0,
                    std::size_t output_index = 0,
                    std::size_t index2 = 0,
                    std::size_t index3 = 0,
                    F f = {})
        {
            auto kernel_size = kernel.getSize(level);
            auto image_size = image.getSize(level);
            //  Cannot use #pragma omp parallel for collapse(2) here because the output_index, index2, and index3 are not thread-safe.
            for (std::size_t i = 0; i < kernel_size; ++i)
            {
                for (std::size_t j = 0; j < image_size; ++j)
                {
                    output_index += (i + j) * output.getStride(level);
                    index2 += j * image.getStride(level);
                    index3 += i * kernel.getStride(level);
                    if(level == 0)
                    {
                        output.set(output_index) = 
                                output.get(output_index) +
                                std::invoke(f, image.get(index2), kernel.get(index3));
                    }
                    else
                    {
                        convolution_detail(std::forward<ExecutionPolicy>(execution_policy), image, kernel, output, level - 1, output_index, index2, index3, f);
                    }
                    output_index -= (i + j) * output.getStride(level);
                    index2 -= j * image.getStride(level);
                    index3 -= i * kernel.getStride(level);
                }
            }
        }
    }

    //  convolution template function implementation
    template<typename ElementT>
    requires(std::floating_point<ElementT> || std::integral<ElementT> || is_complex<ElementT>::value)
    constexpr static auto convolution(const Image<ElementT>& image, const Image<ElementT>& kernel)
    {
        return convolution(std::execution::seq, image, kernel);
    }

    //  convolution template function implementation (with Execution Policy)
    template<class ExecutionPolicy, typename ElementT>
    requires((std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>) &&
             (std::floating_point<ElementT> || std::integral<ElementT> || is_complex<ElementT>::value))
    constexpr static auto convolution(ExecutionPolicy&& execution_policy, const Image<ElementT>& image, const Image<ElementT>& kernel)
    {
        /*  ranges::to support list: https://stackoverflow.com/a/74662256/6667035
        auto output_size =
            std::views::zip_transform(
                [](auto lhs, auto rhs){ return lhs + rhs - 1; },
                image.getSize(),
                kernel.getSize()
                ) | std::ranges::to<std::vector>();
        */
        std::vector<std::size_t> output_size;
        std::ranges::transform(
            image.getSize(),
            kernel.getSize(),
            std::back_inserter(output_size),
            [](auto lhs, auto rhs){ return lhs + rhs - 1; }
        );
        
        Image<ElementT> output(output_size);
        impl::convolution_detail(std::forward<ExecutionPolicy>(execution_policy), image, kernel, output, image.getSize().size() - 1);
        return output;
    }

    //  two dimensional discrete fourier transform template function implementation
    //  https://codereview.stackexchange.com/q/292276/231235
    template<typename ElementT, typename ComplexType = std::complex<long double>>
    requires(std::floating_point<ElementT> || std::integral<ElementT>)
    constexpr static auto dft2(const Image<ElementT>& input)
    {
        Image<ComplexType> output(input.getWidth(), input.getHeight());
        auto normalization_factor = std::sqrt(1.0 / static_cast<long double>(input.getWidth() * input.getHeight()));
        for (std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.getWidth(); ++x)
            {
                long double sum_real = 0.0;
                long double sum_imag = 0.0; 
                for (std::size_t n = 0; n < input.getHeight(); ++n)
                {
                    for (std::size_t m = 0; m < input.getWidth(); ++m)
                    {
                        sum_real += input.at_without_boundary_check(m, n) * 
                            std::cos(2 * std::numbers::pi_v<long double> * (x * m / static_cast<long double>(input.getWidth()) + y * n / static_cast<long double>(input.getHeight())));
                        sum_imag += -input.at_without_boundary_check(m, n) * 
                            std::sin(2 * std::numbers::pi_v<long double> * (x * m / static_cast<long double>(input.getWidth()) + y * n / static_cast<long double>(input.getHeight())));
                    }
                }
                output.at_without_boundary_check(x, y).real(normalization_factor * sum_real);
                output.at_without_boundary_check(x, y).imag(normalization_factor * sum_imag);
            }
        }
        return output;
    }

    //  two dimensional inverse discrete fourier transform template function implementation
    template<typename ElementT, typename ComplexType = std::complex<long double>>
    constexpr auto idft2(const Image<ElementT>& input)
    {
        Image<ComplexType> output(input.getWidth(), input.getHeight());
        auto normalization_factor = std::sqrt(1.0 / static_cast<long double>(input.getWidth() * input.getHeight()));
        for (std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.getWidth(); ++x)
            {
                std::complex<long double> sum = 0.0;
                std::complex<long double> i (0.0,1.0);
                for (std::size_t n = 0; n < input.getHeight(); ++n)
                {
                    for (std::size_t m = 0; m < input.getWidth(); ++m)
                    {
                        sum += input.at_without_boundary_check(m, n) * 
                            (std::cos(2 * std::numbers::pi_v<long double> * (x * m / static_cast<long double>(input.getWidth()) + y * n / static_cast<long double>(input.getHeight()))) +
                            i * std::sin(2 * std::numbers::pi_v<long double> * (x * m / static_cast<long double>(input.getWidth()) + y * n / static_cast<long double>(input.getHeight()))));
                    }
                }
                output.at_without_boundary_check(x, y) = normalization_factor * sum;
            }
        }
        return output;
    }
    
    #ifdef USE_OPENCV
    //  to_cv_mat function implementation
    constexpr auto to_cv_mat(const Image<RGB>& input)
    {
        cv::Mat output = cv::Mat::zeros(cv::Size(input.getWidth(), input.getHeight()), CV_8UC3);
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < output.rows; ++y)
        {
            for (int x = 0; x < output.cols; ++x)
            {
                output.at<cv::Vec3b>(output.rows - y - 1, x)[0] = input.at(x, y).channels[2];
                output.at<cv::Vec3b>(output.rows - y - 1, x)[1] = input.at(x, y).channels[1];
                output.at<cv::Vec3b>(output.rows - y - 1, x)[2] = input.at(x, y).channels[0];
            }
        }
        return output;
    }

    //  to_color_image function implementation
    constexpr auto to_color_image(const cv::Mat input)
    {
        auto output = Image<RGB>(input.cols, input.rows);
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < input.rows; ++y)
        {
            for (int x = 0; x < input.cols; ++x)
            {
                output.at(x, y).channels[0] = input.at<cv::Vec3b>(input.rows - y - 1, x)[2];
                output.at(x, y).channels[1] = input.at<cv::Vec3b>(input.rows - y - 1, x)[1];
                output.at(x, y).channels[2] = input.at<cv::Vec3b>(input.rows - y - 1, x)[0];
            }
        }
        return output;
    }
    #endif

    //  rgb2hsv function implementation
    static auto rgb2hsv(RGB input)
    {
        HSV output{};
        std::uint8_t Red = input.channels[0], Green = input.channels[1], Blue = input.channels[2];
        std::vector<std::uint8_t> v{ Red, Green, Blue };
        std::ranges::sort(v);
        std::uint8_t Max = v[2], Mid = v[1], Min = v[0];

        auto H1 = std::acos(0.5 * ((Red - Green) + (Red - Blue)) /
            std::sqrt(((std::pow((Red - Green), 2.0)) +
                (Red - Blue) * (Green - Blue)))) * (180.0 / std::numbers::pi);
        if (Max == Min)
        {
            output.channels[0] = 0.0;
        }
        else if (Blue <= Green)
        {
            output.channels[0] = H1;
        }
        else
        {
            output.channels[0] = 360.0 - H1;
        }
        if (Max == 0)
        {
            output.channels[1] = 0.0;
        }
        else
        {
            output.channels[1] = 1.0 - (static_cast<double>(Min) / static_cast<double>(Max));
        }
        output.channels[2] = Max;
        return output;
    }

    //  rgb2hsv function implementation
    static auto rgb2hsv(RGB_DOUBLE input)
    {
        RGB rgb{static_cast<std::uint8_t>(input.channels[0]),
                static_cast<std::uint8_t>(input.channels[1]),
                static_cast<std::uint8_t>(input.channels[2])};
        return rgb2hsv(rgb);
    }

    //  hsv2rgb function implementation
    static auto hsv2rgb(HSV input)
    {
        RGB output{};
        long double H = input.channels[0], S = input.channels[1], Max = input.channels[2];
        std::uint8_t hi = static_cast<std::uint8_t>(floor(H / 60.0));
        long double f = (H / 60.0) - hi;
        long double Min, q, t;
        Min = Max * (1.0 - S);
        q = Max * (1.0 - f * S);
        t = Max * (1.0 - (1.0 - f) * S);
        if (hi == 0)
        {
            output.channels[0] = static_cast<std::uint8_t>(Max);
            output.channels[1] = static_cast<std::uint8_t>(t);
            output.channels[2] = static_cast<std::uint8_t>(Min);
        }
        else if (hi == 1)
        {
            output.channels[0] = static_cast<std::uint8_t>(q);
            output.channels[1] = static_cast<std::uint8_t>(Max);
            output.channels[2] = static_cast<std::uint8_t>(Min);
        }
        else if (hi == 2)
        {
            output.channels[0] = static_cast<std::uint8_t>(Min);
            output.channels[1] = static_cast<std::uint8_t>(Max);
            output.channels[2] = static_cast<std::uint8_t>(t);
        }
        else if (hi == 3)
        {
            output.channels[0] = static_cast<std::uint8_t>(Min);
            output.channels[1] = static_cast<std::uint8_t>(q);
            output.channels[2] = static_cast<std::uint8_t>(Max);
        }
        else if (hi == 4)
        {
            output.channels[0] = static_cast<std::uint8_t>(t);
            output.channels[1] = static_cast<std::uint8_t>(Min);
            output.channels[2] = static_cast<std::uint8_t>(Max);
        }
        else if (hi == 5)
        {
            output.channels[0] = static_cast<std::uint8_t>(Max);
            output.channels[1] = static_cast<std::uint8_t>(Min);
            output.channels[2] = static_cast<std::uint8_t>(q);
        }
        return output;
    }

    //  Grayscale2RGB function implementation
    //  Grayscale2RGB function returns RGB pixel which represents GrayScale input in hue color scale. 
    static auto Grayscale2RGB(GrayScale input)
    {
        HSV hsv;
        hsv.channels[0] = static_cast<double>(input) / 256.0 * 360;
        hsv.channels[1] = 1.0;
        hsv.channels[2] = 255.0;
        return hsv2rgb(hsv);
    }

    //  Grayscale2RGB function implementation
    static auto Grayscale2RGB(const Image<GrayScale>& input)
    {
        auto input_data = input.getImageData();
        auto output_data = TinyDIP::recursive_transform([](auto&& input) { return Grayscale2RGB(input); }, input_data);
        Image<RGB> output(output_data, input.getSize());
        return output;
    }

    //  constructRGB template function implementation
    template<typename OutputT = RGB>
    constexpr static auto constructRGB(const Image<GrayScale>& r, const Image<GrayScale>& g, const Image<GrayScale>& b)
    {
        check_size_same(r, g);
        check_size_same(g, b);
        auto image_data_r = r.getImageData();
        auto image_data_g = g.getImageData();
        auto image_data_b = b.getImageData();
        std::vector<OutputT> new_data;
        new_data.resize(r.count());
        #pragma omp parallel for
        for (std::size_t index = 0; index < r.count(); ++index)
        {
            OutputT rgb {   image_data_r[index],
                        image_data_g[index],
                        image_data_b[index]};
            new_data[index] = rgb;
        }
        Image<OutputT> output(new_data, r.getSize());
        return output;
    }

    //  constructRGBDOUBLE template function implementation
    template<typename OutputT = RGB_DOUBLE>
    constexpr static auto constructRGBDOUBLE(const Image<double>& r, const Image<double>& g, const Image<double>& b)
    {
        check_size_same(r, g);
        check_size_same(g, b);
        auto image_data_r = r.getImageData();
        auto image_data_g = g.getImageData();
        auto image_data_b = b.getImageData();
        std::vector<OutputT> new_data;
        new_data.resize(r.count());
        #pragma omp parallel for
        for (std::size_t index = 0; index < r.count(); ++index)
        {
            OutputT rgb_double { image_data_r[index],
                                    image_data_g[index],
                                    image_data_b[index]};
            new_data[index] = rgb_double;
        }
        Image<OutputT> output(new_data, r.getSize());
        return output;
    }

    //  constructHSV template function implementation
    template<typename OutputT = HSV>
    constexpr static auto constructHSV(const Image<double>& h, const Image<double>& s, const Image<double>& v)
    {
        check_size_same(h, s);
        check_size_same(s, v);
        auto image_data_h = h.getImageData();
        auto image_data_s = s.getImageData();
        auto image_data_v = v.getImageData();
        std::vector<OutputT> new_data;
        new_data.resize(h.count());
        #pragma omp parallel for
        for (std::size_t index = 0; index < h.count(); ++index)
        {
            OutputT hsv {   image_data_h[index],
                        image_data_s[index],
                        image_data_v[index]};
            new_data[index] = hsv;
        }
        Image<OutputT> output(new_data, h.getSize());
        return output;
    }

    //  addChannel template function implementation
    template<class ElementT, std::size_t channel_count>
    constexpr static auto addChannel(const MultiChannel<ElementT, channel_count>& multichannel, const ElementT& input)
    {
        MultiChannel<ElementT, channel_count + 1> output{ append(multichannel.channels, input)};
        return output;
    }

    //  addChannel template function implementation
    template<class ElementT, std::size_t channel_count>
    constexpr static auto addChannel(const std::vector<MultiChannel<ElementT, channel_count>>& multichannel, std::vector<ElementT> inputs)
    {
        if (multichannel.size() != inputs.size())
        {
            throw std::runtime_error("Size mismatched!");
        }
        std::vector<MultiChannel<ElementT, channel_count + 1>> output(multichannel.size());
        for (std::size_t i = 0; i < multichannel.size(); i++)
        {
            output[i] = addChannel(multichannel[i], inputs[i]);
        }
        return output;
    }

    //  addChannel template function implementation
    template<class ElementT, std::size_t channel_count>
    constexpr static auto addChannel(const Image<MultiChannel<ElementT, channel_count>>& input, const Image<ElementT>& image_plane)
    {
        if (input.getSize() != image_plane.getSize())
        {
            throw std::runtime_error("Size mismatched!");
        }
        auto image_data1 = input.getImageData();
        auto image_data2 = image_plane.getImageData();
        return Image<MultiChannel<ElementT, channel_count + 1>>(addChannel(image_data1, image_data2), input.getSize());
    }

    //  addChannel template function implementation
    template<class ElementT, std::size_t channel_count, typename... Args>
    requires(is_Image<Args>::value && ...)
    constexpr static auto addChannel(const Image<MultiChannel<ElementT, channel_count>>& input, const Image<ElementT>& image_plane, const Args... images)
    {
        return addChannel(addChannel(input, image_plane), images...);
    }

    //  constructMultiChannel template function implementation
    template<typename ElementT, std::size_t channel_count = 1>
    constexpr static auto constructMultiChannel(const Image<ElementT>& input1)
    {
        auto image_data1 = input1.getImageData();
        std::vector<MultiChannel<ElementT, channel_count>> new_data;
        new_data.resize(input1.count());
        for (std::size_t index = 0; index < input1.count(); ++index)
        {
            new_data[index] = MultiChannel<ElementT, channel_count>{ image_data1[index] };
        }
        Image<MultiChannel<ElementT, channel_count>> output(new_data, input1.getSize());
        return output;
    }

    //  constructMultiChannel template function implementation
    template<typename ElementT, std::size_t channel_count = 2>
    constexpr static auto constructMultiChannel(const Image<ElementT>& input1, const Image<ElementT>& input2)
    {
        check_size_same(input1, input2);
        auto image_data1 = input1.getImageData();
        auto image_data2 = input2.getImageData();
        std::vector<MultiChannel<ElementT, channel_count>> new_data;
        new_data.resize(input1.count());
        #pragma omp parallel for
        for (std::size_t index = 0; index < input1.count(); ++index)
        {
            new_data[index] = MultiChannel<ElementT, channel_count>{ image_data1[index], image_data2[index] };
        }
        Image<MultiChannel<ElementT, channel_count>> output(new_data, input1.getSize());
        return output;
    }

    //  constructMultiChannel template function implementation
    template<typename ElementT, std::size_t channel_count = 3>
    constexpr static auto constructMultiChannel(const Image<ElementT>& input1, const Image<ElementT>& input2, const Image<ElementT>& input3)
    {
        check_size_same(input1, input2);
        check_size_same(input2, input3);
        auto image_data1 = input1.getImageData();
        auto image_data2 = input2.getImageData();
        auto image_data3 = input3.getImageData();
        std::vector<MultiChannel<ElementT, channel_count>> new_data;
        new_data.resize(input1.count());
        #pragma omp parallel for
        for (std::size_t index = 0; index < input1.count(); ++index)
        {
            MultiChannel<ElementT, channel_count> output_element 
                    {   image_data1[index],
                        image_data2[index],
                        image_data3[index]};
            new_data[index] = output_element;
        }
        Image<MultiChannel<ElementT, channel_count>> output(new_data, input1.getSize());
        return output;
    }

    //  constructMultiChannel template function implementation
    template<typename ElementT, typename... Args>
    requires(is_Image<Args>::value && ...)
    constexpr static auto constructMultiChannel(const Image<ElementT>& input1, const Image<ElementT>& input2, const Image<ElementT>& input3, const Args... images)
    {
        return addChannel(constructMultiChannel(input1, input2, input3), images...);
    }

    //  convert_image template function implementation
    //  Reference: https://codereview.stackexchange.com/a/292847/231235
    template<typename DstT, typename SrcT>
    requires(std::same_as<DstT, RGB_DOUBLE> or std::same_as<DstT, HSV>)
    constexpr static auto convert_image(Image<SrcT> input)
    {
        auto image_data = input.getImageData();
        std::vector<DstT> new_data;
        new_data.resize(input.count());
        #pragma omp parallel for
        for (std::size_t index = 0; index < input.count(); ++index)
        {
            DstT dst { static_cast<double>(image_data[index].channels[0]),
                       static_cast<double>(image_data[index].channels[1]),
                       static_cast<double>(image_data[index].channels[2])};
            new_data[index] = dst;
        }
        Image<DstT> output(new_data, input.getSize());
        return output;
    }

    //  convert_image template function implementation
    //  Reference: https://codereview.stackexchange.com/a/292847/231235
    template<typename DstT, typename SrcT>
    requires(std::same_as<DstT, RGB>)
    constexpr static auto convert_image(Image<SrcT> input)
    {
        auto image_data = input.getImageData();
        std::vector<DstT> new_data;
        new_data.resize(input.count());
        #pragma omp parallel for
        for (std::size_t index = 0; index < input.count(); ++index)
        {
            DstT dst { static_cast<GrayScale>(image_data[index].channels[0]),
                       static_cast<GrayScale>(image_data[index].channels[1]),
                       static_cast<GrayScale>(image_data[index].channels[2])};
            new_data[index] = dst;
        }
        Image<DstT> output(new_data, input.getSize());
        return output;
    }

    //  getPlane template function implementation
    template<class OutputT = unsigned char>
    constexpr static auto getPlane(const Image<RGB>& input, std::size_t index)
    {
        auto input_data = input.getImageData();
        std::vector<OutputT> output_data;
        output_data.resize(input.count());
        #pragma omp parallel for
        for (std::size_t i = 0; i < input.count(); ++i)
        {
            output_data[i] = input_data[i].channels[index];
        }
        auto output = Image<OutputT>(output_data, input.getSize());
        return output;
    }
    
    //  getPlane template function implementation
    template<class T = HSV, class OutputT = double>
    requires (std::same_as<T, HSV> || std::same_as<T, RGB_DOUBLE>)
    constexpr static auto getPlane(const Image<T>& input, std::size_t index)
    {
        auto input_data = input.getImageData();
        std::vector<OutputT> output_data;
        output_data.resize(input.count());
        #pragma omp parallel for
        for (std::size_t i = 0; i < input.count(); ++i)
        {
            output_data[i] = input_data[i].channels[index];
        }
        auto output = Image<OutputT>(output_data, input.getSize());
        return output;
    }

    //  getPlane template function implementation
    template<std::size_t channel_count = 3, class T>
    constexpr static auto getPlane(const Image<MultiChannel<T, channel_count>>& input, std::size_t index)
    {
        if (index >= channel_count)
        {
            throw std::runtime_error("Error: index must be less than channel_count.");
        }
        auto input_data = input.getImageData();
        std::vector<T> output_data;
        output_data.resize(input.count());
        #pragma omp parallel for
        for (std::size_t i = 0; i < input.count(); ++i)
        {
            output_data[i] = input_data[i].channels[index];
        }
        auto output = Image<T>(output_data, input.getSize());
        return output;
    }

    //  getRplane function implementation
    constexpr static auto getRplane(const Image<RGB>& input)
    {
        return getPlane(input, 0);
    }

    //  getRplane function implementation
    constexpr static auto getRplane(const Image<RGB_DOUBLE>& input)
    {
        return getPlane(input, 0);
    }

    //  getGplane function implementation
    constexpr static auto getGplane(const Image<RGB>& input)
    {
        return getPlane(input, 1);
    }

    //  getGplane function implementation
    constexpr static auto getGplane(const Image<RGB_DOUBLE>& input)
    {
        return getPlane(input, 1);
    }

    //  getBplane function implementation
    constexpr static auto getBplane(const Image<RGB>& input)
    {
        return getPlane(input, 2);
    }

    //  getBplane function implementation
    constexpr static auto getBplane(const Image<RGB_DOUBLE>& input)
    {
        return getPlane(input, 2);
    }

    //  getHplane function implementation
    constexpr static auto getHplane(const Image<HSV>& input)
    {
        return getPlane(input, 0);
    }

    //  getSplane function implementation
    constexpr static auto getSplane(const Image<HSV>& input)
    {
        return getPlane(input, 1);
    }

    //  getVplane function implementation
    constexpr static auto getVplane(const Image<HSV>& input)
    {
        return getPlane(input, 2);
    }

    //  histogram template function implementation
    //  https://codereview.stackexchange.com/q/295448/231235
    template<class ElementT = int>
    constexpr static auto histogram(const Image<ElementT>& input)
    {
        Histogram<ElementT> output(input);
        return output;
    }

    //  histogram_normalized template function implementation
    //  https://codereview.stackexchange.com/q/295419/231235
    template<class ElementT = std::uint8_t, class ProbabilityType = double>
    requires (std::same_as<ElementT, std::uint8_t> or
              std::same_as<ElementT, std::uint16_t>)
    constexpr static auto histogram_normalized(const Image<ElementT>& input)
    {
        std::array<ProbabilityType, std::numeric_limits<ElementT>::max() - std::numeric_limits<ElementT>::lowest() + 1> histogram_output{};
        auto image_data = input.getImageData();
        for (std::size_t i = 0; i < image_data.size(); ++i)
        {
            histogram_output[image_data[i]] += (1.0 / static_cast<ProbabilityType>(input.count()));
        }
        return histogram_output;
    }

    //  histogram_normalized template function implementation
    template<class ElementT = int, class ProbabilityType = double>
    constexpr static auto histogram_normalized(const Image<ElementT>& input)
    {
        std::map<ElementT, ProbabilityType> histogram_output{};
        auto image_data = input.getImageData();
        for (std::size_t i = 0; i < image_data.size(); ++i)
        {
            if (histogram_output.contains(image_data[i]))
            {
                histogram_output[image_data[i]] += (1.0 / static_cast<ProbabilityType>(input.count()));
            }
            else
            {
                histogram_output.emplace(image_data[i], 1.0 / static_cast<ProbabilityType>(input.count()));
            }
        }
        return histogram_output;
    }

    //  histogram_with_bins template function implementation
    template<std::size_t bins_count = 8, class ElementT = std::uint8_t>
    constexpr static auto histogram_with_bins(
        const Image<ElementT>& input,
        const ElementT lowest = std::numeric_limits<ElementT>::lowest(),
        const ElementT max = std::numeric_limits<ElementT>::max()
    )
    {
        std::array<std::size_t, bins_count> histogram_output{};
        
        auto image_data = input.getImageData();
        for (std::size_t i = 0; i < image_data.size(); ++i)
        {
            std::size_t bin_index = std::floor((static_cast<double>(image_data[i]) * static_cast<double>(bins_count)) / (static_cast<double>(max) - static_cast<double>(lowest)));
            if (bin_index == bins_count)
            {
                ++histogram_output[bins_count - 1];
            }
            else
            {
                ++histogram_output[bin_index];
            }
        }
        return histogram_output;
    }

    //  get_normalized_input template function implementation
    //  https://codereview.stackexchange.com/a/295540/231235
    template<class ElementT, std::size_t Count, class ProbabilityType = double>
    constexpr static auto get_normalized_input(
        const std::array<ElementT, Count>& input,
        const ProbabilityType& sum)
    {
        std::array<ProbabilityType, Count> output{};
        std::transform(std::ranges::cbegin(input), std::ranges::cend(input), std::ranges::begin(output),
            [&](auto&& element)
            {
                return static_cast<ProbabilityType>(element) / sum;
            });
        return output;
    }

    //  get_normalized_input template function implementation
    template<class ExPo, class ElementT, std::size_t Count, class ProbabilityType = double>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto get_normalized_input(
        ExPo&& execution_policy,
        const std::array<ElementT, Count>& input,
        const ProbabilityType& sum)
    {
        std::array<ProbabilityType, Count> output{};
        std::transform(
            std::forward<ExPo>(execution_policy), std::ranges::cbegin(input), std::ranges::cend(input), std::ranges::begin(output),
            [&](auto&& element)
            {
                return static_cast<ProbabilityType>(element) / sum;
            });
        return output;
    }

    //  normalize_histogram template function implementation for std::array
    template<class ElementT, std::size_t Count, class ProbabilityType = double>
    constexpr static auto normalize_histogram(const std::array<ElementT, Count>& input)
    {
        auto sum = std::reduce(std::ranges::cbegin(input), std::ranges::cend(input));
        return get_normalized_input(input, static_cast<ProbabilityType>(sum));
    }

    //  normalize_histogram template function implementation for std::array (with Execution Policy)
    template<class ExecutionPolicy, class ElementT, std::size_t Count, class ProbabilityType = double>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto normalize_histogram(ExecutionPolicy&& execution_policy, const std::array<ElementT, Count>& input)
    {
        auto sum = std::reduce(std::forward<ExecutionPolicy>(execution_policy), std::ranges::cbegin(input), std::ranges::cend(input));
        return get_normalized_input(std::forward<ExecutionPolicy>(execution_policy), input, static_cast<ProbabilityType>(sum));
    }

    //  normalize_histogram template function implementation (with Execution Policy)
    template<class ExecutionPolicy, class ElementT, class CountT, class ProbabilityType = double>
    requires((std::floating_point<CountT> || std::integral<CountT>) and
             (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>))
    constexpr static auto normalize_histogram(ExecutionPolicy&& execution_policy, const std::map<ElementT, CountT>& input)
    {
        auto sum = sum_second_element(input);
        std::map<ElementT, ProbabilityType> output{};
        for (const auto& [key, value] : input)
        {
            output.emplace(key, static_cast<ProbabilityType>(value) / static_cast<ProbabilityType>(sum));
        }
        return output;
    }

    //  normalize_histogram template function implementation
    template<class ElementT, class CountT, class ProbabilityType = double>
    requires(std::floating_point<CountT> || std::integral<CountT>)
    constexpr static auto normalize_histogram(const std::map<ElementT, CountT>& input)
    {
        return normalize_histogram(std::execution::seq, input);
    }
        
    //  otsu_threshold template function implementation (with Execution Policy)
    //  mimic https://github.com/DIPlib/diplib/blob/004755a94fa25a1da79928fd59728abe177bf304/src/histogram/threshold_algorithms.h#L30
    template <class ExPo, std::ranges::input_range RangeT, class FloatingType = double>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto otsu_threshold(
        ExPo&& execution_policy,
        const RangeT& histogram)
    {
        FloatingType w1{};
        FloatingType w2{};
        FloatingType m1{};
        FloatingType m2{};
        for (std::size_t ii = 0; ii < histogram.size(); ++ii)
        {
            w2 += static_cast<FloatingType>(histogram[ii]);
            m2 += static_cast<FloatingType>(histogram[ii]) * static_cast<FloatingType>(ii);
        }
        FloatingType ssMax = -1e6;
        std::size_t maxInd{};
        for (std::size_t ii = 0; ii < histogram.size(); ++ii)
        {
            FloatingType tmp = static_cast<FloatingType>(histogram[ii]);
            w1 += tmp;
            w2 -= tmp;
            tmp *= static_cast<FloatingType>(ii);
            m1 += tmp;
            m2 -= tmp;
            FloatingType c1 = m1 / w1;
            FloatingType c2 = m2 / w2;
            FloatingType c = c1 - c2;
            FloatingType ss = w1 * w2 * c * c;
            if (ss > ssMax)
            {
                ssMax = ss;
                maxInd = ii;
            }
        }
        return (ssMax == -1e6) ? histogram.size() : maxInd;
    }

    //  otsu_threshold template function implementation
    template <std::ranges::input_range RangeT>
    constexpr static auto otsu_threshold(const RangeT& histogram)
    {
        return otsu_threshold(std::execution::seq, histogram);
    }

    //  otsu_threshold template function implementation
    template <class ExPo, class ElementT>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto otsu_threshold(
        ExPo&& execution_policy,
        const Image<ElementT>& input)
    {
        return otsu_threshold(
            std::forward<ExPo>(execution_policy),
            Histogram<ElementT>(input).to_probabilities_vector(
                std::forward<ExPo>(execution_policy)
            )
        );
    }

    //  otsu_threshold template function implementation
    template<class ElementT>
    constexpr static auto otsu_threshold(
        const Image<ElementT>& input)
    {
        return otsu_threshold(std::execution::seq, Histogram<ElementT>(input).to_probabilities_vector());
    }

    //  apply_each_pixel template function implementation
    template<class ExPo, class ElementT, class F, class... Args>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto apply_each_pixel(ExPo&& execution_policy, const Image<ElementT>& input, F operation, Args&&... args)
    {
        std::vector<std::invoke_result_t<F, ElementT, Args...>> output_vector;
        auto input_count = input.count();
        auto input_vector = input.getImageData();
        output_vector.resize(input_count);
        std::transform(std::forward<ExPo>(execution_policy), input_vector.begin(), input_vector.end(), output_vector.begin(), [&](auto& elem) { return std::invoke(operation, elem, args...); });
        return Image<std::invoke_result_t<F, ElementT, Args...>>(output_vector, input.getSize());
    }

    //  apply_each_pixel_openmp template function implementation
    template<class ElementT, class F, class... Args>
    constexpr static auto apply_each_pixel_openmp(const Image<ElementT>& input, F operation, Args&&... args)
    {
        std::vector<std::invoke_result_t<F, ElementT, Args...>> output_vector;
        auto input_count = input.count();
        auto input_vector = input.getImageData();
        output_vector.resize(input_count);
        #pragma omp parallel for
        for (std::size_t i = 0; i < input_count; ++i)
        {
            output_vector[i] = std::invoke(operation, input_vector[i], args...);
        }
        return Image<std::invoke_result_t<F, ElementT, Args...>>(output_vector, input.getSize());
    }

    //  apply_threshold_openmp template function implementation
    template <typename ElementT>
    constexpr static auto apply_threshold_openmp(const Image<ElementT>& image, const ElementT threshold)
    {
        return apply_each_pixel_openmp(image,
            [&](const ElementT& each_pixel)
            {
                return (each_pixel > threshold) ? std::numeric_limits<ElementT>::max() : std::numeric_limits<ElementT>::lowest();
            });
    }

    //  apply_each template function implementation
    template<class F, class... Args>
    constexpr static auto apply_each(const Image<RGB>& input, F operation, Args&&... args)
    {
        auto Rplane = std::async(std::launch::async, [&] { return std::invoke(operation, getRplane(input), args...); });
        auto Gplane = std::async(std::launch::async, [&] { return std::invoke(operation, getGplane(input), args...); });
        auto Bplane = std::async(std::launch::async, [&] { return std::invoke(operation, getBplane(input), args...); });
        if constexpr ((std::is_same<Image<GrayScale>, decltype(Rplane.get())>::value) and
                      (std::is_same<Image<GrayScale>, decltype(Gplane.get())>::value) and
                      (std::is_same<Image<GrayScale>, decltype(Bplane.get())>::value))
        {
            return constructRGB(Rplane.get(), Gplane.get(), Bplane.get());
        }
        else if constexpr ((std::is_same<Image<double>, decltype(Rplane.get())>::value) and
                           (std::is_same<Image<double>, decltype(Gplane.get())>::value) and
                           (std::is_same<Image<double>, decltype(Bplane.get())>::value))
        {
            return constructRGBDOUBLE(Rplane.get(), Gplane.get(), Bplane.get());
        }
        else
        {
            return constructMultiChannel(Rplane.get(), Gplane.get(), Bplane.get());
        }
    }

    //  apply_each template function implementation
    template<class F, class... Args>
    constexpr static auto apply_each(const Image<RGB_DOUBLE>& input, F operation, Args&&... args)
    {
        auto Rplane = std::async(std::launch::async, [&] { return std::invoke(operation, getRplane(input), args...); });
        auto Gplane = std::async(std::launch::async, [&] { return std::invoke(operation, getGplane(input), args...); });
        auto Bplane = std::async(std::launch::async, [&] { return std::invoke(operation, getBplane(input), args...); });
        if constexpr ((std::is_same<Image<double>, decltype(Rplane.get())>::value) and
                      (std::is_same<Image<double>, decltype(Gplane.get())>::value) and
                      (std::is_same<Image<double>, decltype(Bplane.get())>::value))
        {
            return constructRGBDOUBLE(Rplane.get(), Gplane.get(), Bplane.get());
        }
        else
        {
            return constructMultiChannel(Rplane.get(), Gplane.get(), Bplane.get());
        }
    }

    //  apply_each template function implementation
    template<class F, class... Args>
    constexpr static auto apply_each(const Image<HSV>& input, F operation, Args&&... args)
    {
        auto Hplane = std::async(std::launch::async, [&] { return std::invoke(operation, getHplane(input), args...); });
        auto Splane = std::async(std::launch::async, [&] { return std::invoke(operation, getSplane(input), args...); });
        auto Vplane = std::async(std::launch::async, [&] { return std::invoke(operation, getVplane(input), args...); });
        if constexpr ((std::is_same<Image<double>, decltype(Hplane.get())>::value) and
                      (std::is_same<Image<double>, decltype(Splane.get())>::value) and
                      (std::is_same<Image<double>, decltype(Vplane.get())>::value))
        {
            return constructHSV(Hplane.get(), Splane.get(), Vplane.get());
        }
        else
        {
            return constructMultiChannel(Hplane.get(), Splane.get(), Vplane.get());
        }
    }

    //  apply_each_impl template function implementation
    // Helper function implementation using index_sequence
    template<class ElementT, std::size_t Channels, class F, class... Args, std::size_t... Is>
    constexpr static auto apply_each_impl(const Image<MultiChannel<ElementT, Channels>>& input, F operation, Args&&... args, std::index_sequence<Is...>) {
        return constructMultiChannel(
            std::async(std::launch::async, [&] {
                return std::invoke(operation, getPlane(input, Is), std::forward<Args>(args)...);
                }).get()...
                    );
    }

    //  apply_each template function implementation
    template<class ElementT, std::size_t Channels, class F, class... Args>
    constexpr static auto apply_each(const Image<MultiChannel<ElementT, Channels>>& input, F operation, Args&&... args)
    {
        return apply_each_impl(input, operation, std::forward<Args>(args)..., std::make_index_sequence<Channels>{});
    }

    //  apply_each template function implementation
    template<class ElementT, class F, class... Args>
    constexpr static auto apply_each(const Image<MultiChannel<ElementT, 3>>& input, F operation, Args&&... args)
    {
        return constructMultiChannel(
            std::async(std::launch::async, [&] { return std::invoke(operation, getPlane(input, 0), args...); }).get(),
            std::async(std::launch::async, [&] { return std::invoke(operation, getPlane(input, 1), args...); }).get(),
            std::async(std::launch::async, [&] { return std::invoke(operation, getPlane(input, 2), args...); }).get()
        );
    }

    //  apply_each template function implementation
    template<class F, class... Args>
    constexpr static auto apply_each(const Image<RGB>& input1, const Image<RGB>& input2, F operation, Args&&... args)
    {
        auto Rplane = std::async(std::launch::async, [&] { return std::invoke(operation, getRplane(input1), getRplane(input2), args...); });
        auto Gplane = std::async(std::launch::async, [&] { return std::invoke(operation, getGplane(input1), getGplane(input2), args...); });
        auto Bplane = std::async(std::launch::async, [&] { return std::invoke(operation, getBplane(input1), getBplane(input2), args...); });
        return constructRGB(Rplane.get(), Gplane.get(), Bplane.get());
    }

    //  apply_each template function implementation
    template<class F, class... Args>
    constexpr static auto apply_each(const Image<RGB_DOUBLE>& input1, const Image<RGB_DOUBLE>& input2, F operation, Args&&... args)
    {
        auto Rplane = std::async(std::launch::async, [&] { return std::invoke(operation, getRplane(input1), getRplane(input2), args...); });
        auto Gplane = std::async(std::launch::async, [&] { return std::invoke(operation, getGplane(input1), getGplane(input2), args...); });
        auto Bplane = std::async(std::launch::async, [&] { return std::invoke(operation, getBplane(input1), getBplane(input2), args...); });
        return constructRGBDOUBLE(Rplane.get(), Gplane.get(), Bplane.get());
    }

    //  apply_each template function implementation
    template<class F, class... Args>
    constexpr static auto apply_each(const Image<HSV> input1, const Image<HSV> input2, F operation, Args&&... args)
    {
        auto Hplane = std::async(std::launch::async, [&] { return std::invoke(operation, getHplane(input1), getHplane(input2), args...); });
        auto Splane = std::async(std::launch::async, [&] { return std::invoke(operation, getSplane(input1), getSplane(input2), args...); });
        auto Vplane = std::async(std::launch::async, [&] { return std::invoke(operation, getVplane(input1), getVplane(input2), args...); });
        return constructHSV(Hplane.get(), Splane.get(), Vplane.get());
    }

    //  apply_each template function implementation
    template<class ElementT, class F, class... Args>
    constexpr static auto apply_each(const Image<MultiChannel<ElementT, 3>> input1, const Image<MultiChannel<ElementT, 3>> input2, F operation, Args&&... args)
    {
        auto plane1 = std::async(std::launch::async, [&] { return std::invoke(operation, getPlane(input1, 0), getPlane(input2, 0), args...); });
        auto plane2 = std::async(std::launch::async, [&] { return std::invoke(operation, getPlane(input1, 1), getPlane(input2, 1), args...); });
        auto plane3 = std::async(std::launch::async, [&] { return std::invoke(operation, getPlane(input1, 2), getPlane(input2, 2), args...); });
        return constructMultiChannel(plane1.get(), plane2.get(), plane3.get());
    }

    //  apply_each_single_output template function implementation
    template<class ElementT, class F, class... Args>
    constexpr static auto apply_each_single_output(const std::size_t channel_count, const Image<ElementT>& input1, const Image<ElementT>& input2, F operation, Args&&... args)
    {
        std::vector<decltype(std::invoke(operation, getPlane(input1, 0), getPlane(input2, 0), args...))> output;
        output.reserve(channel_count);
        for (std::size_t channel_index = 0; channel_index < channel_count; ++channel_index)
        {
            output.emplace_back(std::invoke(operation, getPlane(input1, channel_index), getPlane(input2, channel_index), args...));
        }
        return output;
    }

    //  im2double function implementation
    constexpr static auto im2double(Image<RGB> input)
    {
        return convert_image<RGB_DOUBLE>(input);
    }

    //  im2double function implementation
    constexpr static auto im2double(Image<GrayScale> input)
    {
        return input.cast<double>();
    }

    //  im2uint8 function implementation
    constexpr static auto im2uint8(Image<RGB_DOUBLE> input)
    {
        return convert_image<RGB>(input);
    }

    //  im2uint8 function implementation
    constexpr static auto im2uint8(Image<double> input)
    {
        return input.cast<GrayScale>();
    }

    //  print_with_latex function implementation
    static void print_with_latex(Image<RGB> input)
    {
        std::cout << "\\begin{tikzpicture}[x=1cm,y=0.4cm]\n";
        for (std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.getWidth(); ++x)
            {
                auto R = input.at(x, y).channels[0];
                auto G = input.at(x, y).channels[1];
                auto B = input.at(x, y).channels[2];

                std::cout << "\\draw (" << x << "," << y << 
                    ") node[anchor=south,fill={rgb:red," << +R << ";green," << +G << ";blue," << +B << "}] {};\n";
            }
        }
        std::cout << "\\end{tikzpicture}\n";
    }

    //  print_with_latex_to_file function implementation
    static void print_with_latex_to_file(Image<RGB> input, std::string filename)
    {
        std::ofstream newfile;
        newfile.open(filename);
        newfile << "\\begin{tikzpicture}[x=1cm,y=0.4cm]\n";
        for (std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.getWidth(); ++x)
            {
                auto R = input.at(x, y).channels[0];
                auto G = input.at(x, y).channels[1];
                auto B = input.at(x, y).channels[2];

                newfile << "\\draw (" << x << "," << y <<
                    ") node[anchor=south,fill={rgb:red," << +R << ";green," << +G << ";blue," << +B << "}] {};\n";
            }
        }
        newfile << "\\end{tikzpicture}\n";
        newfile.close();
        return;
    }

    //  subimage template function implementation for N dimensional image
    template<class ExecutionPolicy, typename ElementT, std::ranges::input_range Sizes, std::ranges::input_range Centers>
    requires (
        (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>) &&
        (std::same_as<std::ranges::range_value_t<Sizes>, std::size_t> ||
         std::same_as<std::ranges::range_value_t<Sizes>, int>) &&
        (std::same_as<std::ranges::range_value_t<Centers>, std::size_t> ||
         std::same_as<std::ranges::range_value_t<Centers>, int>)
    )
    constexpr static auto subimage(
        ExecutionPolicy&& execution_policy,
        const Image<ElementT>& input,
        const Sizes& new_sizes,
        const Centers& centers,
        const ElementT default_element = ElementT{})
    {
        const std::size_t dim = input.getDimensionality();
        if (new_sizes.size() != dim || centers.size() != dim)
        {
            throw std::runtime_error("subimage: new_sizes and centers must match input dimensionality");
        }

        // Compute start indices for each dimension
        std::vector<std::size_t> start(dim);
        for (std::size_t i = 0; i < dim; ++i)
        {
            start[i] = centers[i] - static_cast<std::size_t>(std::floor(static_cast<double>(new_sizes[i]) / 2.0));
        }

        auto count = std::reduce(
            std::ranges::cbegin(new_sizes),
            std::ranges::cend(new_sizes),
            std::size_t{ 1 },
            std::multiplies()
        );
        Image<ElementT> output(
            std::vector<ElementT>(count),
            new_sizes
        );

        std::vector<std::size_t> new_sizes_vec;
        new_sizes_vec.resize(std::ranges::size(new_sizes));  // ✅ Pre-size the vector

        std::transform(
            std::forward<ExecutionPolicy>(execution_policy),
            std::ranges::cbegin(new_sizes),
            std::ranges::cend(new_sizes),
            std::ranges::begin(new_sizes_vec),  // Valid since vector is pre-sized
            [](auto&& element) { return static_cast<std::size_t>(element); }
        );

        // Precompute strides using new_sizes_vec
        std::vector<std::size_t> strides(dim, 1);
        for (std::size_t i = 1; i < dim; ++i)
        {
            strides[i] = strides[i - 1] * new_sizes_vec[i - 1];
        }

        // Iterate through all output elements
        const std::size_t total_elements = output.count();
        for (std::size_t idx = 0; idx < total_elements; ++idx)
        {
            // Use helper function to get indices
            const auto output_indices = linear_index_to_indices(idx, strides, new_sizes_vec);

            // Compute input indices and validate
            bool valid = true;
            std::vector<std::size_t> input_indices(dim);
            for (std::size_t i = 0; i < dim; ++i)
            {
                input_indices[i] = start[i] + output_indices[i];
                if (input_indices[i] >= input.getSize(i))
                {
                    valid = false;
                    break;
                }
            }

            // Assign value
            output.set(idx) = valid ? 
                input.at(input_indices) : 
                default_element;
        }

        return output;
    }

    //  subimage template function implementation
    //  Old Test (only for 2D case): https://godbolt.org/z/9vv3eGYhq
    //  Old Test (not using range-based parameters): https://godbolt.org/z/za9o7KnMP
    //  Test: https://godbolt.org/z/oc1M3ojWs
    template<typename ElementT>
    constexpr static auto subimage(
        const Image<ElementT>& input,
        const std::size_t width,
        const std::size_t height,
        const std::size_t xcenter,
        const std::size_t ycenter,
        const ElementT default_element = ElementT{}
    )
    {
        return subimage(
            std::execution::seq,
            input, 
            std::vector<std::size_t>{width, height}, 
            std::vector<std::size_t>{xcenter, ycenter}, 
            default_element);
    }

    //  subimage2 template function implementation
    template<typename ElementT>
    constexpr static auto subimage2(const Image<ElementT>& input, const std::size_t startx, const std::size_t endx, const std::size_t starty, const std::size_t endy)
    {
        assert(startx <= endx);
        assert(starty <= endy);
        Image<ElementT> output(endx - startx + 1, endy - starty + 1);
        const auto width = output.getWidth();
        const auto height = output.getHeight();
        #pragma omp parallel for collapse(2)
        for (std::size_t y = 0; y < height; ++y)
        {
            for (std::size_t x = 0; x < width; ++x)
            {
                output.at_without_boundary_check(x, y) = input.at_without_boundary_check(startx + x, starty + y);
            }
        }
        return output;
    }

    template<typename ElementT>
    requires (std::same_as<ElementT, RGB>)
    constexpr static auto highlight_region(
        const Image<ElementT>& input,
        const std::size_t startx, const std::size_t endx, const std::size_t starty, const std::size_t endy,
        const std::size_t width = 5, const std::uint8_t value_r = 223, const std::uint8_t value_g = 0, const std::uint8_t value_b = 34)
    {
        assert(startx <= endx);
        assert(starty <= endy);
        auto output = input;
        for (std::size_t y = starty - width / 2; y < endy + width / 2; ++y)
        {
            for (std::size_t x = startx - width / 2; x < endx + width / 2; ++x)
            {
                if (std::abs(static_cast<int>(x) - static_cast<int>(startx)) < width ||
                    std::abs(static_cast<int>(x) - static_cast<int>(endx)) < width ||
                    std::abs(static_cast<int>(y) - static_cast<int>(starty)) < width ||
                    std::abs(static_cast<int>(y) - static_cast<int>(endy)) < width)
                {
                    output.at(x, y).channels[0] = value_r;
                    output.at(x, y).channels[1] = value_g;
                    output.at(x, y).channels[2] = value_b;
                }
            }
        }
        return output;
    }

    /*  split function
    *   xsegments is a number for the block count in x axis
    *   ysegments is a number for the block count in y axis
    */
    template<typename ElementT>
    constexpr static auto split(const Image<ElementT>& input, std::size_t xsegments, std::size_t ysegments)
    {
        std::vector<std::vector<Image<ElementT>>> output;
        std::size_t block_size_x = input.getWidth() / xsegments;
        std::size_t block_size_y = input.getHeight() / ysegments;
        for (std::size_t y = 0; y < ysegments; y++)
        {
            std::vector<Image<ElementT>> output2;
            for (std::size_t x = 0; x < xsegments; x ++)
            {
                output2.push_back(subimage2(input,
                    x * block_size_x,
                    (x + 1) * block_size_x - 1,
                    y * block_size_y,
                    (y + 1) * block_size_y - 1));
            }
            output.push_back(output2);
        }
        return output;
    }

    //  pixelwiseOperation template function implementation
    template<std::size_t unwrap_level = 1, class... Args>
    constexpr static auto pixelwiseOperation(auto op, const Args&... inputs)
    {
        auto transformed_data = recursive_transform<unwrap_level>(
                op,
                inputs.getImageData()...);
        auto output = Image<recursive_unwrap_type_t<unwrap_level, decltype(transformed_data)>>(
            transformed_data,
            first_of(inputs...).getSize());
        return output;
    }

    //  pixelwiseOperation template function implementation
    template<std::size_t unwrap_level = 1, class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto pixelwiseOperation(ExPo&& execution_policy, auto op, const Image<InputT>& input1)
    {
        auto transformed_data = recursive_transform<unwrap_level>(
                                    std::forward<ExPo>(execution_policy),
                                    op,
                                    (input1.getImageData()));
        auto output = Image<recursive_unwrap_type_t<unwrap_level, decltype(transformed_data)>>(
            transformed_data,
            input1.getSize());
        return output;
    }

    //  rgb2hsv template function implementation
    template<typename ElementT, typename OutputT = HSV>
    requires (std::same_as<ElementT, RGB> || std::same_as<ElementT, RGB_DOUBLE>)
    constexpr static auto rgb2hsv(const Image<ElementT>& input)
    {
        return pixelwiseOperation([](ElementT input) { return rgb2hsv(input); }, input);
    }

    //  rgb2hsv template function implementation
    template<class ExPo, typename ElementT, typename OutputT = HSV>
    requires (std::same_as<ElementT, RGB> || std::same_as<ElementT, RGB_DOUBLE>) && 
    (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto rgb2hsv(ExPo&& execution_policy, const Image<ElementT>& input)
    {
        return pixelwiseOperation(std::forward<ExPo>(execution_policy), [](ElementT input) { return rgb2hsv(input); }, input);
    }

    //  hsv2rgb template function implementation
    template<typename OutputT = RGB>
    constexpr static auto hsv2rgb(const Image<HSV>& input)
    {
        return pixelwiseOperation([](HSV input) { return hsv2rgb(input); }, input);
    }

    //  hsv2rgb template function implementation
    template<class ExPo>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto hsv2rgb(ExPo&& execution_policy, const Image<HSV>& input)
    {
        return pixelwiseOperation(std::forward<ExPo>(execution_policy), [](HSV input) { return hsv2rgb(input); }, input);
    }

    template<typename ElementT>
    constexpr static auto concat_horizontal(Image<ElementT> input1, Image<ElementT> input2)
    {
        check_height_same(input1, input2);
        Image<ElementT> output(input1.getWidth() + input2.getWidth(), input1.getHeight());
        for (std::size_t y = 0; y < input1.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input1.getWidth(); ++x)
            {
                output.at(x, y) = input1.at(x, y);
            }
        }
        for (std::size_t y = 0; y < input2.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input2.getWidth(); ++x)
            {
                output.at(input1.getWidth() + x, y) = input2.at(x, y);
            }
        }
        return output;
    }

    template<typename ElementT>
    constexpr static auto concat_horizontal(const std::vector<Image<ElementT>>& input)
    {
        //return recursive_reduce(input, Image<ElementT>(0, input[0].getHeight()), [](Image<ElementT> element1, Image<ElementT> element2) { return concat_horizontal(element1, element2); });
        auto output = input[0];
        for (std::size_t i = 1; i < input.size(); i++)
        {
            output = concat_horizontal(output, input[i]);
        }
        return output;
    }

    template<typename ElementT>
    constexpr static auto concat_vertical(const Image<ElementT>& input1, const Image<ElementT>& input2)
    {
        check_width_same(input1, input2);
        Image<ElementT> output(input1.getWidth(), input1.getHeight() + input2.getHeight());
        for (std::size_t y = 0; y < input1.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input1.getWidth(); ++x)
            {
                output.at(x, y) = input1.at(x, y);
            }
        }
        for (std::size_t y = 0; y < input2.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input2.getWidth(); ++x)
            {
                output.at(x, input1.getHeight() + y) = input2.at(x, y);
            }
        }
        return output;
    }

    template<typename ElementT>
    constexpr static auto concat_vertical(const std::vector<Image<ElementT>>& input)
    {
        auto output = input[0];
        for (std::size_t i = 1; i < input.size(); i++)
        {
            output = concat_vertical(output, input[i]);
        }
        return output;
    }

    template<typename ElementT>
    constexpr static auto concat(const std::vector<std::vector<Image<ElementT>>>& input)
    {
        auto result1 = recursive_transform<1>(
            //std::execution::par,
            [](std::vector<Image<ElementT>> input) { return concat_horizontal(input); },
            input);
        return concat_vertical(result1);
    }

    //  normalDistribution1D template function implementation
    template<typename T = double>
    constexpr static auto normalDistribution1D(const T x, const T standard_deviation)
    {
        return std::exp(-x * x / (2 * standard_deviation * standard_deviation));
    }

    //  normalDistribution1D template function implementation
    template<typename T1, typename T2 = double>
    requires (std::floating_point<T1> || std::integral<T1>)
    constexpr static auto normalDistribution1D(const T1 x, const T2 standard_deviation)
    {
        return std::exp(-static_cast<T2>(x) * static_cast<T2>(x) / (2 * standard_deviation * standard_deviation));
    }

    //  normalDistribution1D template function implementation
    template<typename T = double>
    requires (std::floating_point<T> || std::integral<T>)
    constexpr static auto normalDistribution1D(const std::complex<T>& x, const std::complex<T>& standard_deviation)
    {
        return std::exp(-x * x / (std::complex{ static_cast<T>(2) , static_cast<T>(0) } *standard_deviation * standard_deviation));
    }

    //  normalDistribution2D template function implementation
    template<typename T = double>
    constexpr static auto normalDistribution2D(const T xlocation, const T ylocation, const T standard_deviation)
    {
        return std::exp(-(xlocation * xlocation + ylocation * ylocation) / (2 * standard_deviation * standard_deviation)) / (2 * std::numbers::pi * standard_deviation * standard_deviation);
    }

    //  normalize template function implementation
    template<class ElementT = double>
    constexpr static auto normalize(const Image<ElementT>& input)
    {
        auto max_value = max(input);
        auto min_value = min(input);
        auto difference_value = max_value - min_value;
        Image<ElementT> min_value_plane = input;
        min_value_plane.setAllValue(min_value);
        Image<ElementT> difference_value_plane = input;
        difference_value_plane.setAllValue(difference_value);
        return divides(subtract(input, min_value_plane), difference_value_plane);
    }

    template<class InputT1, class InputT2>
    constexpr static auto cubic_interpolate(const InputT1 v0, const InputT1 v1, const InputT1 v2, const InputT1 v3, const InputT2 frac)
    {
        auto A = (v3-v2)-(v0-v1);
        auto B = (v0-v1)-A;
        auto C = v2-v0;
        auto D = v1;
        return D + frac * (C + frac * (B + frac * A));
    }

    template<class InputT = float, class ElementT>
    constexpr static auto bicubicPolate(const ElementT* const ndata, const InputT fracx, const InputT fracy)
    {
        auto x1 = cubic_interpolate( ndata[0], ndata[1], ndata[2], ndata[3], fracx );
        auto x2 = cubic_interpolate( ndata[4], ndata[5], ndata[6], ndata[7], fracx );
        auto x3 = cubic_interpolate( ndata[8], ndata[9], ndata[10], ndata[11], fracx );
        auto x4 = cubic_interpolate( ndata[12], ndata[13], ndata[14], ndata[15], fracx );

        return std::clamp(
                    static_cast<InputT>(cubic_interpolate(x1, x2, x3, x4, fracy)),
                    static_cast<InputT>(std::numeric_limits<ElementT>::min()),
                    static_cast<InputT>(std::numeric_limits<ElementT>::max()));
    }

    //  copyResizeBicubic template function implementation
    template<class FloatingType = float, arithmetic ElementT>
    constexpr static Image<ElementT> copyResizeBicubic(const Image<ElementT>& image, const std::size_t width, const std::size_t height)
    {
        Image<ElementT> output(width, height);
        //  get used to the C++ way of casting
        const auto ratiox = static_cast<FloatingType>(image.getWidth()) / static_cast<FloatingType>(width);
        const auto ratioy = static_cast<FloatingType>(image.getHeight()) / static_cast<FloatingType>(height);
        #pragma omp parallel for collapse(2)
        for (std::size_t y = 0; y < height; ++y)
        {
            for (std::size_t x = 0; x < width; ++x)
            {
                const FloatingType xMappingToOrigin = static_cast<FloatingType>(x) * ratiox;
                const FloatingType yMappingToOrigin = static_cast<FloatingType>(y) * ratioy;
                const FloatingType xMappingToOriginFloor = std::floor(xMappingToOrigin);
                const FloatingType yMappingToOriginFloor = std::floor(yMappingToOrigin);
                const FloatingType xMappingToOriginFrac = xMappingToOrigin - xMappingToOriginFloor;
                const FloatingType yMappingToOriginFrac = yMappingToOrigin - yMappingToOriginFloor;
                
                ElementT ndata[4 * 4];
                for (int ndatay = -1; ndatay <= 2; ++ndatay)
                {
                    for (int ndatax = -1; ndatax <= 2; ++ndatax)
                    {
                        ndata[(ndatay + 1) * 4 + (ndatax + 1)] = image.at(
                            static_cast<std::size_t>(std::clamp(xMappingToOriginFloor + ndatax, static_cast<FloatingType>(0), static_cast<FloatingType>(image.getWidth()) - static_cast<FloatingType>(1))),
                            static_cast<std::size_t>(std::clamp(yMappingToOriginFloor + ndatay, static_cast<FloatingType>(0), static_cast<FloatingType>(image.getHeight()) - static_cast<FloatingType>(1))));
                    }
                    
                }
                output.at(x, y) = bicubicPolate(ndata, xMappingToOriginFrac, yMappingToOriginFrac);
            }
        }
        return output;
    }

    //  copyResizeBicubic template function implementation for color image
    template<class FloatingType = double, class ElementT>
    requires (std::same_as<ElementT, RGB> || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>) || is_MultiChannel<ElementT>::value)
    Image<ElementT> copyResizeBicubic(
        const Image<ElementT>& image,
        const std::size_t width,
        const std::size_t height)
    {
        return apply_each(image, [&](auto&& each_plane)
        {
            return copyResizeBicubic<FloatingType>(each_plane, width, height);
        });
    }


    //  gaussianFigure1D template function implementation
    template<class InputT>
    constexpr static Image<InputT> gaussianFigure1D(
        const std::size_t size, const std::size_t center, const InputT standard_deviation)
    {
        auto row_vector = Image<InputT>(size, std::size_t{1});
        for (std::size_t x = 0; x < size; ++x)
        {
            row_vector.at(x, static_cast<std::size_t>(0)) = normalDistribution1D(static_cast<InputT>(x) - static_cast<InputT>(center), standard_deviation);
        }
        return row_vector;
    }

    //  gaussianFigure2D template function implementation
    //  multiple standard deviations
    template<class InputT>
    constexpr static Image<InputT> gaussianFigure2D(
        const std::size_t xsize, const std::size_t ysize, 
        const std::size_t centerx, const std::size_t centery,
        const InputT standard_deviation_x, const InputT standard_deviation_y)
    {
        auto output = Image<InputT>(xsize, ysize);
        auto row_vector_x = gaussianFigure1D(xsize, centerx, standard_deviation_x);

        auto row_vector_y = gaussianFigure1D(ysize, centery, standard_deviation_y);
        
        for (std::size_t y = 0; y < ysize; ++y)
        {
            for (std::size_t x = 0; x < xsize; ++x)
            {
                output.at(x, y) = row_vector_x.at(x, static_cast<std::size_t>(0)) * row_vector_y.at(y, static_cast<std::size_t>(0));
            }
        }
        return output;
    }

    //  gaussianFigure2D Template Function Implementation
    //  General two-dimensional elliptical Gaussian
    //  https://fabiandablander.com/statistics/Two-Properties.html
    template<class InputT = double>
    constexpr static auto gaussianFigure2D(
        const std::size_t xsize, const std::size_t ysize,
        const std::size_t centerx, const std::size_t centery,
        const InputT sigma1_2, const InputT sigma2_2,
        const InputT rho, const InputT normalize_factor_input = 1.0)
    {
        Image<InputT> output(xsize, ysize);
        auto sigma1 = std::sqrt(sigma1_2);
        auto sigma2 = std::sqrt(sigma2_2);
        auto normalize_factor =
            normalize_factor_input / (2.0 * std::numbers::pi_v<InputT> * sigma1 * sigma2 * std::sqrt(1.0 - std::pow(rho, 2.0)));

        auto exp_para = -1.0 / (2.0 * sigma1_2 * sigma2_2 * (1.0 - std::pow(rho, 2.0)));
        #pragma omp parallel for
        for (std::size_t y = 0; y < ysize; ++y)
        {
            auto x2 = static_cast<InputT>(y) - static_cast<InputT>(centery);
            auto x2_2 = std::pow(x2, 2.0);
            for (std::size_t x = 0; x < xsize; ++x)
            {
                auto x1 = static_cast<InputT>(x) - static_cast<InputT>(centerx);
                auto x1_2 = std::pow(x1, 2.0);
                output.at(x, y) = normalize_factor*
                    std::exp(
                        exp_para * (
                            sigma2_2 * x1_2 - 
                            (2 * rho * sigma1 * sigma2 * x1 * x2) +
                            sigma1_2 * x2_2
                            )
                        );
            }
        }
        return output;
    }

    //  gaussianFigure2D Template Function Implementation
    //  single standard deviation
    template<class InputT>
    constexpr static Image<InputT> gaussianFigure2D(
        const std::size_t xsize, const std::size_t ysize,
        const std::size_t centerx, const std::size_t centery,
        const InputT standard_deviation)
    {
        return gaussianFigure2D(xsize, ysize, centerx, centery, standard_deviation, standard_deviation);
    }

    //  gaussianFigure3D Template Function Implementation
    //  multiple standard deviations
    template<class InputT>
    requires(std::floating_point<InputT> || std::integral<InputT>)
    constexpr static auto gaussianFigure3D(
        const std::size_t xsize, const std::size_t ysize, const std::size_t zsize,
        const std::size_t centerx, const std::size_t centery, const std::size_t centerz,
        const InputT standard_deviation_x, const InputT standard_deviation_y, const InputT standard_deviation_z)
    {
        Image<InputT> output(xsize, ysize, zsize);
        auto gaussian_image2d = gaussianFigure2D(xsize, ysize, centerx, centery, standard_deviation_x, standard_deviation_y);
        for (std::size_t z = 0; z < zsize; ++z)
        {
            auto each_plane = gaussian_image2d *
                normalDistribution1D(static_cast<InputT>(z) - static_cast<InputT>(centerz), standard_deviation_z);
            for (std::size_t y = 0; y < ysize; ++y)
            {
                for (std::size_t x = 0; x < xsize; ++x)
                {
                    output.at_without_boundary_check(x, y, z) = each_plane.at_without_boundary_check(x, y);
                }
            }
        }
        return output;
    }

    template<class InputT>
    constexpr static Image<InputT> plus(const Image<InputT>& input1)
    {
        return input1;
    }

    template<class InputT, class... Args>
    constexpr static Image<InputT> plus(const Image<InputT>& input1, const Args&... inputs)
    {
        return pixelwiseOperation(std::plus<>{}, input1, plus(inputs...));
    }

    template<class InputT, class... Args>
    constexpr static auto plus(const std::vector<Image<InputT>>& input1, const Args&... inputs)
    {
        return recursive_transform<1>(
            [](auto&& input1_element, auto&&... inputs_element)
            {
                return plus(input1_element, inputs_element...);
            }, input1, inputs...);
    }

    //  subtract Template Function Implementation
    template<class InputT>
    constexpr static Image<InputT> subtract(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        check_size_same(input1, input2);
        return pixelwiseOperation(std::minus<>{}, input1, input2);
    }

    //  subtract Template Function Implementation
    template<class InputT>
    constexpr static auto subtract(const std::vector<Image<InputT>>& input1, const std::vector<Image<InputT>>& input2)
    {
        if (input1.size() != input2.size())
        {
            throw std::runtime_error("Size mismatched!");
        }
        return recursive_transform<1>(
            [](auto&& input1_element, auto&& input2_element)
            {
                return subtract(input1_element, input2_element);
            }, input1, input2);
    }

    //  subtract Function Implementation
    constexpr static Image<RGB> subtract(const Image<RGB>& input1, const Image<RGB>& input2)
    {
        check_size_same(input1, input2);
        return pixelwiseOperation(
                [](RGB x, RGB y)
                {
                    RGB rgb;
                    for(std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                    {
                        rgb.channels[channel_index] = 
                        std::clamp(
                            x.channels[channel_index] - 
                            y.channels[channel_index],
                            0,
                            255);
                    }
                    return rgb;
                },
                input1,
                input2
            );
    }

    //  subtract Template Function Implementation
    template<class InputT>
    requires((std::same_as<InputT, RGB_DOUBLE>) || (std::same_as<InputT, HSV>))
    constexpr static auto subtract(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        check_size_same(input1, input2);
        return pixelwiseOperation(
                [](InputT x, InputT y)
                {
                    InputT output;
                    for(std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                    {
                        output.channels[channel_index] = x.channels[channel_index] - y.channels[channel_index];
                    }
                    return output;
                },
                input1,
                input2
            );
    }

    //  pixelwise_multiplies Template Function Implementation
    template<class InputT1, class InputT2>
    constexpr static auto pixelwise_multiplies(const Image<InputT1>& input1, const Image<InputT2>& input2)
    {
        if (input1.getSize() != input2.getSize())
        {
            throw std::runtime_error("Size mismatched!");
        }
        return pixelwiseOperation(std::multiplies<>{}, input1, input2);
    }

    //  multiplies Template Function Implementation
    template<class InputT, class TimesT>
    requires((std::floating_point<TimesT> || std::integral<TimesT> || is_complex<TimesT>::value) &&
             ((!std::same_as<InputT, RGB>) && (!std::same_as<InputT, RGB_DOUBLE>) && (!std::same_as<InputT, HSV>)))
    constexpr static auto multiplies(const Image<InputT>& input1, const TimesT times)
    {
        std::vector<TimesT> data;
        data.resize(input1.count());
        auto image = Image<TimesT>(data, input1.getSize());
        image.setAllValue(times);
        return pixelwise_multiplies(
            input1,
            image
        );
    }

    //  multiplies Template Function Implementation
    template<class InputT, class TimesT>
    requires((std::floating_point<TimesT> || std::integral<TimesT> || is_complex<TimesT>::value) &&
             ((std::same_as<InputT, RGB>) || (std::same_as<InputT, RGB_DOUBLE>) || (std::same_as<InputT, HSV>)))
    constexpr static auto multiplies(const Image<InputT>& input1, const TimesT times)
    {
        return apply_each(input1, [&](auto&& each_plane)
        {
            return multiplies(each_plane, times);
        });
    }

    //  multiplies Template Function Implementation
    template<class InputT, class TimesT>
    requires(std::floating_point<TimesT> || std::integral<TimesT>)
    constexpr static auto multiplies(const TimesT times, const Image<InputT>& input1)
    {
        return multiplies(input1, times);
    }

    //  pixelwise_multiplies Template Function Implementation
    template<class ExPo, class InputT1, class InputT2>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto pixelwise_multiplies(ExPo&& execution_policy, const Image<InputT1>& input1, const Image<InputT2>& input2)
    {
        return pixelwiseOperation(std::forward<ExPo>(execution_policy), std::multiplies<>{}, input1, input2);
    }

    //  pixelwise_multiplies Template Function Implementation
    template<class InputT, class... Args>
    constexpr static auto pixelwise_multiplies(const Image<InputT>& input1, const Args&... inputs)
    {
        return pixelwiseOperation(std::multiplies<>{}, input1, pixelwise_multiplies(inputs...));
    }

    //  pixelwise_multiplies Template Function Implementation
    template<class ExPo, class InputT, class... Args>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto pixelwise_multiplies(ExPo&& execution_policy, const Image<InputT>& input1, const Args&... inputs)
    {
        return pixelwiseOperation(std::forward<ExPo>(execution_policy), std::multiplies<>{}, input1, pixelwise_multiplies(inputs...));
    }

    template<class InputT, class... Args>
    constexpr static auto pixelwise_multiplies(const std::vector<Image<InputT>>& input1, const Args&... inputs)
    {
        return recursive_transform<1>(
            [](auto&& input1_element, auto&&... inputs_element)
            {
                return pixelwise_multiplies(input1_element, inputs_element...);
            }, input1, inputs...);
    }

    //  divides Template Function Implementation
    /*
    template<class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static Image<InputT> divides(ExPo&& execution_policy, const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return pixelwiseOperation(std::forward<ExPo>(execution_policy), std::divides<>{}, input1, input2);
    }

    //  divides Template Function Implementation
    template<class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto divides(ExPo&& execution_policy, const std::vector<Image<InputT>>& input1, const std::vector<Image<InputT>>& input2)
    {
        assert(input1.size() == input2.size());
        return recursive_transform<1>(
            [&](auto&& input1_element, auto&& input2_element)
            {
                return divides(std::forward<ExPo>(execution_policy), input1_element, input2_element);
            }, input1, input2);
    }
    */

    //  divides Template Function Implementation
    template<class InputT>
    constexpr static auto divides(const std::vector<Image<InputT>>& input1, const std::vector<Image<InputT>>& input2)
    {
        assert(input1.size() == input2.size());
        return recursive_transform<1>(
            [](auto&& input1_element, auto&& input2_element)
            {
                return divides(input1_element, input2_element);
            }, input1, input2);
    }

    //  divides Template Function Implementation
    template<class InputT>
    constexpr static auto divides(const Image<InputT>& input1, const InputT input2)
    {
        auto image_for_divides = Image<InputT>(input1.getImageData(), input1.getSize());
        image_for_divides.setAllValue(input2);
        return divides(input1, image_for_divides);
    }

    //  divides Template Function Implementation
    template<class InputT>
    constexpr static Image<InputT> divides(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return pixelwiseOperation(std::divides<>{}, input1, input2);
    }
    
    template<class InputT>
    constexpr static Image<InputT> modulus(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return pixelwiseOperation(std::modulus<>{}, input1, input2);
    }

    template<class InputT>
    constexpr static Image<InputT> negate(const Image<InputT>& input1)
    {
        return pixelwiseOperation(std::negate<>{}, input1);
    }

    //  negate template function implementation
    template<class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static Image<InputT> negate(ExPo&& execution_policy, const Image<InputT>& input1)
    {
        return pixelwiseOperation(std::forward<ExPo>(execution_policy), std::negate<>{}, input1);
    }

    template<std::floating_point ElementT = double, std::floating_point OutputT = ElementT>
    Image<OutputT> dct3_one_plane(const std::vector<Image<ElementT>>& input, const std::size_t plane_index)
    {
        auto N1 = static_cast<OutputT>(input[0].getWidth());
        auto N2 = static_cast<OutputT>(input[0].getHeight());
        auto N3 = input.size();
        auto alpha1 = (plane_index == 0) ? (std::numbers::sqrt2_v<OutputT> / 2) : (OutputT{1.0});
        auto output = Image<OutputT>(input[plane_index].getWidth(), input[plane_index].getHeight());
        #pragma omp parallel for
        for (std::size_t y = 0; y < output.getHeight(); ++y)
        {
            OutputT alpha2 = (y == 0) ? (std::numbers::sqrt2_v<OutputT> / 2) : (OutputT{1.0});
            for (std::size_t x = 0; x < output.getWidth(); ++x)
            {
                OutputT sum{};
                OutputT alpha3 = (x == 0) ? (std::numbers::sqrt2_v<OutputT> / 2) : (OutputT{1.0});
                for (std::size_t inner_z = 0; inner_z < N3; ++inner_z)
                {
                    auto plane = input[inner_z];
                    for (std::size_t inner_y = 0; inner_y < plane.getHeight(); ++inner_y)
                    {
                        for (std::size_t inner_x = 0; inner_x < plane.getWidth(); ++inner_x)
                        {
                            auto l1 = (std::numbers::pi_v<OutputT> / (2 * N1) * (2 * static_cast<OutputT>(inner_x) + 1) * x);
                            auto l2 = (std::numbers::pi_v<OutputT> / (2 * N2) * (2 * static_cast<OutputT>(inner_y) + 1) * y);
                            auto l3 = (std::numbers::pi_v<OutputT> / (2 * static_cast<OutputT>(N3)) * (2 * static_cast<OutputT>(inner_z) + 1) * static_cast<OutputT>(plane_index));
                            sum += static_cast<OutputT>(plane.at(inner_x, inner_y)) *
                                std::cos(l1) * std::cos(l2) * std::cos(l3);
                        }
                    }
                }
                output.at(x, y) = 8 * alpha1 * alpha2 * alpha3 * sum / (N1 * N2 * N3);
            }
        }
        return output;
    }

    template<std::floating_point ElementT = double, std::floating_point OutputT = ElementT>
    std::vector<Image<OutputT>> dct3(const std::vector<Image<ElementT>>& input)
    {
        std::vector<Image<OutputT>> output;
        output.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
        {
            output[i] = dct3_one_plane<ElementT, OutputT>(input, i);
        }
        return output;
    }

    //  idct3_one_plane template function implementation
    template<std::floating_point ElementT = double, std::floating_point OutputT = ElementT>
    Image<OutputT> idct3_one_plane(const std::vector<Image<ElementT>>& input, const std::size_t plane_index)
    {
        auto N1 = static_cast<OutputT>(input[0].getWidth());
        auto N2 = static_cast<OutputT>(input[0].getHeight());
        auto N3 = input.size();
        auto output = Image<OutputT>(input[plane_index].getWidth(), input[plane_index].getHeight());
        auto height = output.getHeight();
        auto width = output.getWidth();
        #pragma omp parallel for collapse(2)
        for (std::size_t y = 0; y < height; ++y)
        {
            for (std::size_t x = 0; x < width; ++x)
            {
                OutputT sum{};
                for (std::size_t inner_z = 0; inner_z < N3; ++inner_z)
                {
                    auto plane = input[inner_z];
                    for (std::size_t inner_y = 0; inner_y < plane.getHeight(); ++inner_y)
                    {
                        for (std::size_t inner_x = 0; inner_x < plane.getWidth(); ++inner_x)
                        {
                            auto l1 = (std::numbers::pi_v<OutputT> / (2 * N1) * (2 * x + 1) * static_cast<OutputT>(inner_x));
                            auto l2 = (std::numbers::pi_v<OutputT> / (2 * N2) * (2 * y + 1) * static_cast<OutputT>(inner_y));
                            auto l3 = (std::numbers::pi_v<OutputT> / (2 * static_cast<OutputT>(N3)) * (2 * static_cast<OutputT>(plane_index) + 1) * static_cast<OutputT>(inner_z));
                            OutputT alpha1 = (inner_x == 0) ? (std::numbers::sqrt2_v<OutputT> / 2) : (OutputT{1.0});
                            OutputT alpha2 = (inner_y == 0) ? (std::numbers::sqrt2_v<OutputT> / 2) : (OutputT{1.0});
                            OutputT alpha3 = (inner_z == 0) ? (std::numbers::sqrt2_v<OutputT> / 2) : (OutputT{1.0});
                            sum += alpha1 * alpha2 * alpha3 * static_cast<OutputT>(plane.at(inner_x, inner_y)) *
                                std::cos(l1) * std::cos(l2) * std::cos(l3);
                        }
                    }
                }
                output.at(x, y) = sum;
            }
        }
        return output;
    }

    template<std::floating_point ElementT = double, std::floating_point OutputT = ElementT>
    std::vector<Image<OutputT>> idct3(const std::vector<Image<ElementT>>& input)
    {
        std::vector<Image<OutputT>> output;
        output.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
        {
            output[i] = idct3_one_plane<ElementT, OutputT>(input, i);
        }
        return output;
    }

    template<arithmetic ElementT = double, arithmetic OutputT = ElementT>
    constexpr static Image<ElementT> dct2(Image<ElementT> input)
    {
        Image<ElementT> output;
        std::vector v{ input };
        output = dct3_one_plane(v, 0);
        return output;
    }

    template<arithmetic ElementT = double, arithmetic OutputT = ElementT>
    constexpr static Image<ElementT> idct2(Image<ElementT> input)
    {
        Image<ElementT> output;
        std::vector v{ input };
        output = idct3_one_plane(v, 0);
        return output;
    }

    //  abs template function implementation
    template<arithmetic ElementT = double>
    constexpr static auto abs(const ElementT& input)
    {
        return std::abs(input);
    }

    //  abs template function implementation with Execution Policy
    template<class ExPo, arithmetic ElementT = double>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto abs(ExPo&& execution_policy, const ElementT& input)
    {
        return std::abs(input);
    }

    template<class ElementT = double>
    constexpr static auto abs(const Image<ElementT>& input)
    {
        return pixelwiseOperation([](auto&& element) { return abs(element); }, input);
    }

    //  abs template function implementation
    template<class ExPo, class ElementT = double>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto abs(ExPo&& execution_policy, const Image<ElementT>& input)
    {
        return pixelwiseOperation(
            std::forward<ExPo>(execution_policy),
            [&](auto&& element)
            {
                return abs(std::forward<ExPo>(execution_policy), element);
            },
            input
        );
    }

    //  abs template function implementation
    template<class InputT>
    requires((std::same_as<InputT, RGB>) || (std::same_as<InputT, RGB_DOUBLE>) || (std::same_as<InputT, HSV>) || is_MultiChannel<InputT>::value)
    constexpr static auto abs(const Image<InputT>& input)
    {
        return apply_each(input, [&](auto&& planes) { return abs(planes); });
    }

    //  abs template function implementation
    template<class ExPo, class InputT>
    requires((std::same_as<InputT, RGB>) || (std::same_as<InputT, RGB_DOUBLE>) || (std::same_as<InputT, HSV>) || is_MultiChannel<InputT>::value) and
            (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto abs(ExPo&& execution_policy, const Image<InputT>& input)
    {
        return apply_each(input, [&](auto&& planes) { return abs(std::forward<ExPo>(execution_policy), planes); });
    }

    //  difference template function implementation
    template<arithmetic ElementT = double>
    constexpr static auto difference(const Image<ElementT>& input1, const Image<ElementT>& input2)
    {
        return pixelwiseOperation([](auto&& element1, auto&& element2)
            {
                if (element1 > element2)
                    return element1 - element2;
                return element2 - element1;
            }, input1, input2);
    }

    //  difference Template Function Implementation
    template<class InputT>
    requires((std::same_as<InputT, RGB>) || (std::same_as<InputT, RGB_DOUBLE>) || (std::same_as<InputT, HSV>) || is_MultiChannel<InputT>::value)
    constexpr static auto difference(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return pixelwiseOperation(
            [](InputT element1, InputT element2)
            {
                InputT output;
                for(std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                {
                    output.channels[channel_index] =
                        (element1.channels[channel_index] > element2.channels[channel_index]) ?
                        (element1.channels[channel_index] - element2.channels[channel_index]) :
                        (element2.channels[channel_index] - element1.channels[channel_index]);
                }
                return output;
            }, input1, input2);
    }

    //  difference Template Function Implementation
    template<class ExPo, std::ranges::input_range RangeT>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto difference(ExPo&& execution_policy, const RangeT& input1, const RangeT& input2)
    {
        if (input1.size() != input2.size())
        {
            throw std::runtime_error("Size mismatched!");
        }
        std::vector<std::ranges::range_value_t<RangeT>> difference_vector(input1.size());
        std::transform(
            std::forward<ExPo>(execution_policy),
            std::ranges::cbegin(input1),
            std::ranges::cend(input1),
            std::ranges::cbegin(input2),
            std::ranges::begin(difference_vector),
            [&](auto&& element1, auto&& element2)
            {
                if (element1 > element2)
                {
                    return element1 - element2;
                }
                return element2 - element1;
            }
        );
        return difference_vector;
    }

    //  manhattan_distance Template Function Implementation
    //  https://codereview.stackexchange.com/q/270857/231235
    template<arithmetic ElementT = double>
    constexpr static auto manhattan_distance(const Image<ElementT>& input1, const Image<ElementT>& input2)
    {
        if(input1.getSize() != input2.getSize())
        {
            throw std::runtime_error("Size mismatched!");
        }
        return recursive_reduce(difference(input1, input2).getImageData(), ElementT{});
    }

    //  euclidean_distance Template Function Implementation
    template <arithmetic OutputT = double, typename T, std::size_t N>
    constexpr static OutputT euclidean_distance(const std::array<T, N>& p1, const std::array<T, N>& p2)
    {
        std::array<OutputT, N> diff;
        std::transform(p1.begin(), p1.end(), p2.begin(), diff.begin(), std::minus<OutputT>());
        return std::sqrt(std::inner_product(diff.begin(), diff.end(), diff.begin(), OutputT{}));
    }

    //  euclidean_distance Template Function Implementation (with Execution Policy)
    template<
        arithmetic OutputT = double,
        class ExPo,
        typename T,
        std::size_t N
    >
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static OutputT euclidean_distance(ExPo&& execution_policy, const std::array<T, N>& p1, const std::array<T, N>& p2)
    {
        std::array<OutputT, N> diff;
        std::transform(std::forward<ExPo>(execution_policy), p1.begin(), p1.end(), p2.begin(), diff.begin(), std::minus<OutputT>());
        return std::sqrt(std::inner_product(diff.begin(), diff.end(), diff.begin(), OutputT{}));
    }

    //  euclidean_distance Template Function Implementation (with Execution Policy)
    template<
        arithmetic OutputT = double,
        class ExPo,
        arithmetic ElementT1 = double,
        arithmetic ElementT2 = double
        >
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto euclidean_distance(
        ExPo&& execution_policy,
        const Image<ElementT1>& input1,
        const Image<ElementT2>& input2,
        const OutputT output = 0.0
        )
    {
        if (input1.getSize() != input2.getSize())
        {
            throw std::runtime_error("Size mismatched!");
        }
        return std::sqrt(two_input_map_reduce(std::forward<ExPo>(execution_policy), input1.getImageData(), input2.getImageData(), OutputT{},
            [&](auto&& element1, auto&& element2) {
                return std::pow(element1 - element2, 2.0);
            }));
    }

    //  euclidean_distance Template Function Implementation
    template<
        arithmetic OutputT = double,
        arithmetic ElementT1 = double,
        arithmetic ElementT2 = double
        >
    constexpr static auto euclidean_distance(
        const Image<ElementT1>& input1,
        const Image<ElementT2>& input2,
        const OutputT output = 0.0
        )
    {
        return euclidean_distance(std::execution::seq, input1, input2, output);
    }

    //  euclidean_distance Template Function Implementation for multiple channel image
    template<
        class ExPo,
        class ElementT1,
        class ElementT2
        >
    requires((std::same_as<ElementT1, RGB>) || (std::same_as<ElementT1, RGB_DOUBLE>) || (std::same_as<ElementT1, HSV>)) and
            ((std::same_as<ElementT2, RGB>) || (std::same_as<ElementT2, RGB_DOUBLE>) || (std::same_as<ElementT2, HSV>)) and
            (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto euclidean_distance(
        ExPo&& execution_policy,
        const Image<ElementT1>& input1,
        const Image<ElementT2>& input2
        )
    {
        return sqrt(execution_policy, two_input_map_reduce(std::forward<ExPo>(execution_policy), input1.getImageData(), input2.getImageData(), MultiChannel<double>{},
            [&](auto&& element1, auto&& element2) {
                return pow(execution_policy, element1 - element2, 2.0);
            }));
    }

    //  euclidean_distance Template Function Implementation for multiple channel image
    template<
        class ExPo,
        class ElementT1,
        class ElementT2,
        std::size_t Size
        >
    constexpr static auto euclidean_distance(
        ExPo&& execution_policy,
        const Image<MultiChannel<ElementT1, Size>>& input1,
        const Image<MultiChannel<ElementT2, Size>>& input2
        )
    {
        return sqrt(std::forward<ExPo>(execution_policy), two_input_map_reduce(std::forward<ExPo>(execution_policy), input1.getImageData(), input2.getImageData(), MultiChannel<ElementT1, Size>{},
            [&](auto&& element1, auto&& element2) {
                return pow(execution_policy, element1 - element2, 2.0);
            }));
    }

    //  euclidean_distance Template Function Implementation for multiple channel image
    template<
        class ElementT1,
        class ElementT2
        >
    requires((std::same_as<ElementT1, RGB>) || (std::same_as<ElementT1, RGB_DOUBLE>) || (std::same_as<ElementT1, HSV>) || (is_MultiChannel<ElementT1>::value)) and
            ((std::same_as<ElementT2, RGB>) || (std::same_as<ElementT2, RGB_DOUBLE>) || (std::same_as<ElementT2, HSV>) || (is_MultiChannel<ElementT2>::value))
    constexpr static auto euclidean_distance(
        const Image<ElementT1>& input1,
        const Image<ElementT2>& input2
        )
    {
        return euclidean_distance(std::execution::seq, input1, input2);
    }

    template<arithmetic ElementT = double, arithmetic ExpT = double>
    constexpr static auto pow(const Image<ElementT>& input, ExpT exp)
    {
        return pixelwiseOperation([&](auto&& element) { return std::pow(element, exp); }, input);
    }

    //  pow template function implementation
    template<class ExPo, arithmetic ElementT = double, arithmetic ExpT = double>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto pow(ExPo&& execution_policy, const Image<ElementT>& input, ExpT exp)
    {
        return pixelwiseOperation(std::forward<ExPo>(execution_policy), [&](auto&& element) { return std::pow(element, exp); }, input);
    }

    //  pow template function implementation
    template<class InputT, arithmetic ExpT = double>
    requires((std::same_as<InputT, RGB>) || (std::same_as<InputT, RGB_DOUBLE>) || (std::same_as<InputT, HSV>) || is_MultiChannel<InputT>::value)
    constexpr static auto pow(const Image<InputT>& input, ExpT exp)
    {
        return apply_each(input, [&](auto&& planes) { return pow(planes, exp); });
    }

    //  sum template function implementation
    template<typename ElementT = double, typename F = std::plus<std::common_type_t<ElementT, ElementT>>>
    requires(std::regular_invocable<F, ElementT, ElementT>)
    constexpr static auto sum(const Image<ElementT>& input, F f = {})
    {
        auto image_data = input.getImageData();
        return std::reduce(std::ranges::cbegin(image_data), std::ranges::cend(image_data), ElementT{}, f);
    }

    //  sum template function implementation with execution policy
    template<class ExecutionPolicy, typename ElementT = double, typename F = std::plus<std::common_type_t<ElementT, ElementT>>>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
              std::regular_invocable<F, ElementT, ElementT>)
    constexpr static auto sum(ExecutionPolicy execution_policy, const Image<ElementT>& input, F f = {})
    {
        auto image_data = input.getImageData();
        return std::reduce(execution_policy, std::ranges::cbegin(image_data), std::ranges::cend(image_data), ElementT{}, f);
    }

    //  mean template function implementation
    template<typename ElementT = double, typename F = std::plus<std::common_type_t<ElementT, ElementT>>>
    requires(std::regular_invocable<F, ElementT, ElementT>)
    constexpr static auto mean(const Image<ElementT>& input, F f = {})
    {
        return std::invoke(std::divides<>(), sum(input), input.count());
    }


    //  min template function implementation
    template<typename ElementT = double>
    constexpr static auto min(const Image<ElementT>& input)
    {
        return std::ranges::min(input.getImageData());
    }

    //  max template function implementation
    template<typename ElementT = double>
    constexpr static auto max(const Image<ElementT>& input)
    {
        return std::ranges::max(input.getImageData());
    }

    //  minmax template function implementation
    template<typename ElementT = double>
    constexpr static auto minmax(const Image<ElementT>& input)
    {
        return std::ranges::minmax(input.getImageData());
    }

    //  count template function implementation
    template<typename ElementT>
    constexpr static auto count(const Image<ElementT>& input, ElementT target)
    {
        return std::ranges::count(input.getImageData(), target);
    }

    //  count template function implementation
    template<class ExPo, class ElementT>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto count(ExPo execution_policy, const Image<ElementT>& input, const ElementT target)
    {
        auto image_data = input.getImageData();
        return std::count(execution_policy, std::ranges::cbegin(image_data), std::ranges::cend(image_data), target);
    }

    //  unique_value template function implementation
    template<typename ElementT>
    constexpr static bool unique_value(const Image<ElementT>& input, ElementT target)
    {
        return count(input, target) == 1;
    }

    //  butterworth_fisheye template function implementation
    template<arithmetic ElementT, std::floating_point FloatingType = double>
    constexpr static auto butterworth_fisheye(const Image<ElementT>& input, ElementT D0, ElementT N)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        
        Image<ElementT> output(input.getWidth(), input.getHeight());
        for (std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.getWidth(); ++x)
            {
                FloatingType distance_x = x - static_cast<FloatingType>(input.getWidth()) / 2.0;
                FloatingType distance_y = y - static_cast<FloatingType>(input.getHeight()) / 2.0;
                FloatingType distance = std::hypot(distance_x, distance_y);
                FloatingType angle = std::atan2(distance_y, distance_x);
                FloatingType weight = 1 / std::pow((1 + std::pow(distance / D0, 2 * N)), 0.5);
                FloatingType new_distance = distance * weight;
                FloatingType new_distance_x = new_distance * std::cos(angle);
                FloatingType new_distance_y = new_distance * std::sin(angle);
                output.at(
                    static_cast<std::size_t>(new_distance_x + static_cast<FloatingType>(input.getWidth()) / 2.0),
                    static_cast<std::size_t>(new_distance_y + static_cast<FloatingType>(input.getHeight()) / 2.0)) = 
                    input.at(x, y);
            }
        }
        return output;
    }

    //  gaussian_fisheye template function implementation
    //  Reference: https://codereview.stackexchange.com/q/291059/231235
    template<arithmetic ElementT, std::floating_point FloatingType = double>
    constexpr static auto gaussian_fisheye(const Image<ElementT>& input, FloatingType D0)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        
        Image<ElementT> output(input.getWidth(), input.getHeight());
        for (std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.getWidth(); ++x)
            {
                FloatingType distance_x = x - static_cast<FloatingType>(input.getWidth()) / 2.0;
                FloatingType distance_y = y - static_cast<FloatingType>(input.getHeight()) / 2.0;
                FloatingType weight = normalDistribution2D(std::fabs(distance_x), std::fabs(distance_y), D0) / normalDistribution2D(0.0, 0.0, D0);
                FloatingType new_distance_x = distance_x * weight;
                FloatingType new_distance_y = distance_y * weight;
                output.at(
                    static_cast<std::size_t>(new_distance_x + static_cast<FloatingType>(input.getWidth()) / 2.0),
                    static_cast<std::size_t>(new_distance_y + static_cast<FloatingType>(input.getHeight()) / 2.0)) = 
                    input.at(x, y);
            }
        }
        return output;
    }

    //  gaussian_fisheye template function implementation for the types other than std::floating_point
    template<arithmetic ElementT, std::integral T = int>
    constexpr static auto gaussian_fisheye(const Image<ElementT>& input, T D0)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return gaussian_fisheye(input, static_cast<double>(D0));
    }

    //  gaussian_fisheye template function implementation
    template<typename ElementT, class FloatingType = double>
    requires ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>))
    constexpr static auto gaussian_fisheye(const Image<ElementT>& input, FloatingType D0)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return apply_each(input, [&](auto&& planes) { return gaussian_fisheye(planes, D0); });
    }

    //  paste2D template function implementation
    template<typename ElementT>
    constexpr static auto paste2D(const Image<ElementT>& background, const Image<ElementT>& target, std::size_t x_location, std::size_t y_location, ElementT default_value = ElementT{})
    {
        return paste2D(std::execution::seq, background, target, x_location, y_location, default_value);
    }

    //  paste2D template function implementation (with execution policy)
    //  Test: https://godbolt.org/z/5hjns1nGP
    template<class ExecutionPolicy, typename ElementT>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto paste2D(ExecutionPolicy&& execution_policy, const Image<ElementT>& background, const Image<ElementT>& target, std::size_t x_location, std::size_t y_location, ElementT default_value = ElementT{})
    {
        if (background.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        if (target.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        if((background.getWidth() >= target.getWidth() + x_location) &&
           (background.getHeight() >= target.getHeight() + y_location))
        {
            auto output = background;
            for (std::size_t y = 0; y < target.getHeight(); ++y)
            {
                for (std::size_t x = 0; x < target.getWidth(); ++x)
                {
                    output.at_without_boundary_check(x_location + x, y_location + y) = target.at_without_boundary_check(x, y);
                }
            }
            return output;
        }
        else
        {
            std::vector<ElementT> data;
            auto xsize = (background.getWidth() >= target.getWidth() + x_location)?
                    background.getWidth():
                    (target.getWidth() + x_location);
            auto ysize = (background.getHeight() >= target.getHeight() + y_location)?
                    background.getHeight():
                    (target.getHeight() + y_location);
            data.resize(xsize * ysize);
            std::fill(execution_policy, std::ranges::begin(data), std::ranges::end(data), default_value);
            Image<ElementT> output(data, xsize, ysize);
            for (std::size_t y = 0; y < background.getHeight(); ++y)
            {
                for (std::size_t x = 0; x < background.getWidth(); ++x)
                {
                    output.at_without_boundary_check(x, y) = background.at_without_boundary_check(x, y);
                }
            }
            for (std::size_t y = 0; y < target.getHeight(); ++y)
            {
                for (std::size_t x = 0; x < target.getWidth(); ++x)
                {
                    output.at_without_boundary_check(x_location + x, y_location + y) = target.at_without_boundary_check(x, y);
                }
            }
            return output;
        }
    }

    //  rotate_detail template function implementation
    //  rotate_detail template function performs image rotation between 0° to 90°
    template<arithmetic ElementT, std::floating_point FloatingType = double>
    constexpr static auto rotate_detail(const Image<ElementT>& input, FloatingType radians)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        //  if 0° rotation case
        if (radians == 0)
        {
            return input;
        }
        //  if 90° rotation case
        if(radians == std::numbers::pi_v<long double> / 2.0)
        {
            Image<ElementT> output(input.getHeight(), input.getWidth());
            for (std::size_t y = 0; y < input.getHeight(); ++y)
            {
                for (std::size_t x = 0; x < input.getWidth(); ++x)
                {
                    output.at(input.getHeight() - y - 1, x) = 
                        input.at(x, y);
                }
            }
            return output;
        }
        
        FloatingType half_width = static_cast<FloatingType>(input.getWidth()) / 2.0;
        FloatingType half_height = static_cast<FloatingType>(input.getHeight()) / 2.0;
        FloatingType new_width = 2 * 
            std::hypot(half_width, half_height) *
            std::abs(std::sin(std::atan2(half_width, half_height) + radians));
        FloatingType new_height = 2 *
            std::hypot(half_width, half_height) *
            std::abs(std::sin(std::atan2(half_height, half_width) + radians));
        
        Image<ElementT> output(input.getWidth(), input.getHeight());
        for (std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.getWidth(); ++x)
            {
                
                FloatingType distance_x = x - half_width;
                FloatingType distance_y = y - half_height;
                FloatingType distance = std::hypot(distance_x, distance_y);
                FloatingType angle = std::atan2(distance_y, distance_x) + radians;
                
                FloatingType width_ratio = new_width / static_cast<FloatingType>(input.getWidth());
                FloatingType height_ratio = new_height / static_cast<FloatingType>(input.getHeight());
                FloatingType distance_weight = (input.getWidth() > input.getHeight())?(1 / height_ratio):(1 / width_ratio);
                FloatingType new_distance = distance * distance_weight;
                FloatingType new_distance_x = new_distance * std::cos(angle);
                FloatingType new_distance_y = new_distance * std::sin(angle);
                output.at(
                    static_cast<std::size_t>(new_distance_x + half_width),
                    static_cast<std::size_t>(new_distance_y + half_height)) = 
                    input.at(x, y);
            }
        }
        return output;
    }

    //  rotate_detail_shear_transformation template function implementation
    //  rotate_detail_shear_transformation template function performs image rotation
    //  Reference: https://gautamnagrawal.medium.com/rotating-image-by-any-angle-shear-transformation-using-only-numpy-d28d16eb5076
    template<arithmetic ElementT, std::floating_point FloatingType = double>
    constexpr static auto rotate_detail_shear_transformation(const Image<ElementT>& input, FloatingType radians)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        radians = std::fmod(radians, 2 * std::numbers::pi_v<long double>);
        //  if negative degrees
        if(radians < 0)
        {
            radians = radians + 2 * std::numbers::pi_v<long double>;
        }
        //  if 0° rotation case
        if (radians == 0)
        {
            return input;
        }
        //  if 90° rotation case
        if(radians == std::numbers::pi_v<long double> / 2.0)
        {
            Image<ElementT> output(input.getHeight(), input.getWidth());
            for (std::size_t y = 0; y < input.getHeight(); ++y)
            {
                for (std::size_t x = 0; x < input.getWidth(); ++x)
                {
                    output.at(input.getHeight() - y - 1, x) = 
                        input.at(x, y);
                }
            }
            return output;
        }
        auto cosine = std::cos(radians);
        auto sine = std::sin(radians);
        auto height = input.getHeight();
        auto width = input.getWidth();
        FloatingType original_centre_width  = std::round((static_cast<FloatingType>(width) + 1.0) / 2.0 - 1.0);
        FloatingType original_centre_height = std::round((static_cast<FloatingType>(height) + 1.0) / 2.0 - 1.0);

        //  Define the height and width of the new image that is to be formed
        auto new_height = std::round(std::abs(height*cosine) + std::abs(width*sine)) + 1;
        auto new_width = std::round(std::abs(width*cosine) + std::abs(height*sine)) + 1;

        //  Define another image variable of dimensions of new_height and new _column filled with zeros
        Image<ElementT> output(static_cast<std::size_t>(new_width), static_cast<std::size_t>(new_height));

        //  Find the centre of the new image that will be obtained
        FloatingType new_centre_width  = std::round((static_cast<FloatingType>(new_width) + 1.0) / 2.0 - 1.0);
        FloatingType new_centre_height = std::round((static_cast<FloatingType>(new_height) + 1.0) / 2.0 - 1.0);

        for (std::size_t i = 0; i < input.getHeight(); ++i)
        {
            for (std::size_t j = 0; j < input.getWidth(); ++j)
            {
                //  co-ordinates of pixel with respect to the centre of original image
                auto y = height - 1.0 - i - original_centre_height;
                auto x = width - 1.0 - j - original_centre_width;

                //  co-ordinate of pixel with respect to the rotated image
                auto new_y = std::round(-x * sine + y * cosine);
                auto new_x = std::round(x * cosine + y * sine);

                /*  since image will be rotated the centre will change too, 
                    so to adust to that we will need to change new_x and new_y with respect to the new centre*/
                new_y = new_centre_height - new_y;
                new_x = new_centre_width - new_x;
                if((0 <= new_x) && (new_x < new_width) &&
                   (0 <= new_y) && (new_y < new_height))
                {
                    output.at(
                    static_cast<std::size_t>(new_x),
                    static_cast<std::size_t>(new_y)) = 
                    input.at(j, i);
                }
            }
        }
        return output;
    }

    //  rotate_detail_shear_transformation template function implementation
    template<typename ElementT, class FloatingType = double>
    requires ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>))       //  TODO: Create a base class for both RGB and HSV
    constexpr static auto rotate_detail_shear_transformation(const Image<ElementT>& input, FloatingType radians)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return apply_each(input, [&](auto&& planes) { return rotate_detail_shear_transformation(planes, radians); });
    }

    //  rotate_detail_shear_transformation_degree template function implementation
    template<typename ElementT, class T = double>
    constexpr static auto rotate_detail_shear_transformation_degree(const Image<ElementT>& input, T degrees)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return rotate_detail_shear_transformation(input, static_cast<double>(degrees) * std::numbers::pi_v<long double> / 180.0);
    }

    //  rotate template function implementation
    template<arithmetic ElementT, std::floating_point FloatingType = double>
    constexpr static auto rotate(const Image<ElementT>& input, FloatingType radians)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        auto output = input;
        radians = std::fmod(radians, std::numbers::pi_v<long double> / 2.0);
        while(radians > std::numbers::pi_v<long double> / 2.0)
        {
            output = rotate_detail(output, std::numbers::pi_v<long double> / 2.0);
            radians-=(std::numbers::pi_v<long double> / 2.0);
        }
        output = rotate_detail(output, radians);
        return output;
    }

    //  rotate template function implementation
    template<typename ElementT, class FloatingType = double>
    requires ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>))
    constexpr static auto rotate(const Image<ElementT>& input, FloatingType radians)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return apply_each(input, [&](auto&& planes) { return rotate(planes, radians); });
    }

    //  rotate template function implementation
    template<arithmetic ElementT, std::integral T = int>
    constexpr static auto rotate(const Image<ElementT>& input, T radians)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return rotate(input, static_cast<double>(radians));
    }

    //  rotate_degree template function implementation
    template<typename ElementT, class T = double>
    constexpr static auto rotate_degree(const Image<ElementT>& input, T degrees)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return rotate(input, static_cast<double>(degrees) * std::numbers::pi_v<long double> / 180.0);
    }

    //  transpose template function implementation
    template<typename ElementT>
    constexpr static auto transpose(const Image<ElementT>& input)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        Image<ElementT> output(input.getHeight(), input.getWidth());
        for (std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input.getWidth(); ++x)
            {
                output.at(y, x) = 
                    input.at(x, y);
            }
        }
        return output;
    }

    //  flip_horizontal template function implementation
    template<typename ElementT>
    constexpr static auto flip_horizontal(const Image<ElementT>& input)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        Image<ElementT> output = input;
        auto height = input.getHeight();
        auto width = input.getWidth();
        #pragma omp parallel for collapse(2)
        for(std::size_t y = 0; y < height; ++y)
        {
            for(std::size_t x = 0; x < width; ++x)
            {
                output.at_without_boundary_check(input.getWidth() - x - 1, y) = input.at_without_boundary_check(x, y);
            }
        }
        return output;
    }

    //  flip_vertical template function implementation
    template<typename ElementT>
    constexpr static auto flip_vertical(const Image<ElementT>& input)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        Image<ElementT> output = input;
        auto height = input.getHeight();
        auto width = input.getWidth();
        #pragma omp parallel for collapse(2)
        for(std::size_t y = 0; y < height; ++y)
        {
            for(std::size_t x = 0; x < width; ++x)
            {
                output.at_without_boundary_check(x, input.getHeight() - y - 1) = input.at_without_boundary_check(x, y);
            }
        }
        return output;
    }

    //  flip_horizontal_vertical template function implementation
    template<typename ElementT>
    constexpr static auto flip_horizontal_vertical(const Image<ElementT>& input)
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        Image<ElementT> output = input;
        auto height = input.getHeight();
        auto width = input.getWidth();
        #pragma omp parallel for collapse(2)
        for(std::size_t y = 0; y < height; ++y)
        {
            for(std::size_t x = 0; x < width; ++x)
            {
                output.at_without_boundary_check(
                    input.getWidth() - x - 1,
                    input.getHeight() - y - 1
                    ) = input.at_without_boundary_check(x, y);
            }
        }
        return output;
    }

    //  generate_constant_padding_image template function implementation
    template<typename ElementT>
    constexpr static auto generate_constant_padding_image(
        const Image<ElementT> input,
        std::size_t width_expansion,
        std::size_t height_expansion,
        ElementT default_value = ElementT{})
    {
        return generate_constant_padding_image(std::execution::seq, input, width_expansion, height_expansion, default_value);
    }
    
    //  generate_constant_padding_image template function implementation (with Execution Policy)
    template<class ExecutionPolicy, typename ElementT>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto generate_constant_padding_image(
        ExecutionPolicy&& execution_policy, 
        const Image<ElementT>& input,
        const std::size_t width_expansion,
        const std::size_t height_expansion,
        const ElementT default_value = ElementT{})
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        Image<ElementT> output(input.getWidth() + 2 * width_expansion, input.getHeight() + 2 * height_expansion);
        output.setAllValue(default_value);
        output = paste2D(execution_policy, output, input, width_expansion, height_expansion, default_value);
        return output;
    }

    //  generate_mirror_padding_image template function implementation
    template<typename ElementT>
    constexpr static auto generate_mirror_padding_image(
        const Image<ElementT> input,
        std::size_t width_expansion,
        std::size_t height_expansion,
        ElementT default_value = ElementT{})
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return generate_mirror_padding_image(std::execution::seq, input, width_expansion, height_expansion, default_value);
    }

    //  generate_mirror_padding_image template function implementation (with Execution Policy)
    template<class ExecutionPolicy, typename ElementT>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto generate_mirror_padding_image(
        ExecutionPolicy&& execution_policy, 
        const Image<ElementT> input,
        std::size_t width_expansion,
        std::size_t height_expansion,
        ElementT default_value = ElementT{})
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        auto output = generate_constant_padding_image(execution_policy, input, width_expansion, height_expansion, default_value);
        auto flipped_vertical = flip_vertical(input);
        output = paste2D(
            execution_policy,
            output,
            subimage2(flipped_vertical, 0, flipped_vertical.getWidth() - 1, input.getHeight() - height_expansion - 1, flipped_vertical.getHeight() - 1),
            width_expansion,
            0,
            default_value);
        output = paste2D(
            execution_policy,
            output,
            subimage2(flipped_vertical, 0, flipped_vertical.getWidth() - 1, 0, height_expansion),
            width_expansion,
            input.getHeight() + height_expansion - 1,
            default_value);
        auto flipped_horizontal = flip_horizontal(input);
        output = paste2D(
            execution_policy,
            output,
            subimage2(flipped_horizontal, input.getWidth() - width_expansion - 1, flipped_horizontal.getWidth() - 1, 0, flipped_horizontal.getHeight() - 1),
            0,
            height_expansion,
            default_value);
        output = paste2D(
            execution_policy,
            output,
            subimage2(flipped_horizontal, 0, width_expansion, 0, flipped_horizontal.getHeight() - 1),
            input.getWidth() + width_expansion - 1,
            height_expansion,
            default_value);
        auto flipped_horizontal_vertical = flip_horizontal_vertical(input);
        output = paste2D(
            execution_policy,
            output,
            subimage2(
                flipped_horizontal_vertical,
                flipped_horizontal_vertical.getWidth() - width_expansion - 1,
                flipped_horizontal_vertical.getWidth() - 1,
                flipped_horizontal_vertical.getHeight() - height_expansion - 1,
                flipped_horizontal_vertical.getHeight() - 1),
            0,
            0,
            default_value);
        output = paste2D(
            execution_policy,
            output,
            subimage2(
                flipped_horizontal_vertical,
                0,
                width_expansion,
                flipped_horizontal_vertical.getHeight() - height_expansion - 1,
                flipped_horizontal_vertical.getHeight() - 1),
            input.getWidth() + width_expansion - 1,
            0,
            default_value);
        output = paste2D(
            execution_policy,
            output,
            subimage2(
                flipped_horizontal_vertical,
                flipped_horizontal_vertical.getWidth() - width_expansion - 1,
                flipped_horizontal_vertical.getWidth() - 1,
                0,
                height_expansion),
            0,
            input.getHeight() + height_expansion - 1,
            default_value);
        output = paste2D(
            execution_policy,
            output,
            subimage2(
                flipped_horizontal_vertical,
                0,
                width_expansion,
                0,
                height_expansion),
            input.getWidth() + width_expansion - 1,
            input.getHeight() + height_expansion - 1,
            default_value);
        return output;
    }

    //  generate_replicate_padding_image template function implementation
    template<typename ElementT>
    constexpr static auto generate_replicate_padding_image(
        const Image<ElementT> input,
        std::size_t width_expansion,
        std::size_t height_expansion,
        ElementT default_value = ElementT{})
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        return generate_replicate_padding_image(std::execution::seq, input, width_expansion, height_expansion, default_value);
    }

    //  generate_replicate_padding_image template function implementation (with Execution Policy)
    //  Test: https://godbolt.org/z/1hebz7hEh
    template<class ExecutionPolicy, typename ElementT>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto generate_replicate_padding_image(
        ExecutionPolicy&& execution_policy, 
        const Image<ElementT> input,
        std::size_t width_expansion,
        std::size_t height_expansion,
        ElementT default_value = ElementT{})
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        auto output = generate_constant_padding_image(execution_policy, input, width_expansion, height_expansion, default_value);
        //  Top block
        for(std::size_t y = 0; y < height_expansion; ++y)
        {
            output = paste2D(
                execution_policy,
                output,
                subimage2(input, 0, input.getWidth() - 1, 0, 0),
                width_expansion,
                y,
                default_value);
        }
        //  Bottom block
        for(std::size_t y = input.getHeight() + height_expansion; y < input.getHeight() + 2 * height_expansion; ++y)
        {
            output = paste2D(
                execution_policy,
                output,
                subimage2(input, 0, input.getWidth() - 1, input.getHeight() - 1, input.getHeight() - 1),
                width_expansion,
                y,
                default_value);
        }
        //  Left block
        for(std::size_t x = 0; x < width_expansion; ++x)
        {
            output = paste2D(
                execution_policy,
                output,
                subimage2(input, 0, 0, 0, input.getHeight() - 1),
                x,
                height_expansion,
                default_value);
        }
        //  Right block
        for(std::size_t x = input.getWidth() + width_expansion; x < input.getWidth() + 2 * width_expansion; ++x)
        {
            output = paste2D(
                execution_policy,
                output,
                subimage2(input, input.getWidth() - 1, input.getWidth() - 1, 0, input.getHeight() - 1),
                x,
                height_expansion,
                default_value);
        }
        Image<ElementT> temp(width_expansion, height_expansion);
        //  Left-top corner
        temp.setAllValue(input.at(static_cast<std::size_t>(0), static_cast<std::size_t>(0)));
        output = paste2D(
            execution_policy,
            output,
            temp,
            0,
            0,
            default_value);
        //  Right-top corner
        temp.setAllValue(input.at(input.getWidth() - static_cast<std::size_t>(1), static_cast<std::size_t>(0)));
        output = paste2D(
            execution_policy,
            output,
            temp,
            width_expansion + input.getWidth(),
            0,
            default_value);
        //  Left-bottom corner
        temp.setAllValue(input.at(static_cast<std::size_t>(0), input.getHeight() - static_cast<std::size_t>(1)));
        output = paste2D(
            execution_policy,
            output,
            temp,
            0,
            height_expansion + input.getHeight(),
            default_value);
        //  Right-bottom corner
        temp.setAllValue(input.at(input.getWidth() - static_cast<std::size_t>(1), input.getHeight() - static_cast<std::size_t>(1)));
        output = paste2D(
            execution_policy,
            output,
            temp,
            width_expansion + input.getWidth(),
            height_expansion + input.getHeight(),
            default_value);
        return output;
    }

    //  computeFilterSizeFromSigma template function implementation
    template<typename ElementT>
    constexpr static auto computeFilterSizeFromSigma(ElementT sigma)
    {
        return 2 * std::ceil(2 * sigma) + 1;
    }

    enum BoundaryCondition {
        constant,
        mirror,
        replicate
    };

    //  generate_padded_image template function implementation
    template<class ExecutionPolicy, typename ElementT>
    constexpr static auto generate_padded_image(
        ExecutionPolicy&& execution_policy,
        const Image<ElementT>& input,
        const std::size_t width_expansion,
        const std::size_t height_expansion,
        const BoundaryCondition boundaryCondition = BoundaryCondition::mirror,
        const ElementT& value_for_constant_padding = ElementT{}
    )
    {
        Image<ElementT> padded_image;
        switch (boundaryCondition)
        {
        case constant:
            padded_image = generate_constant_padding_image(std::forward<ExecutionPolicy>(execution_policy), input, width_expansion, height_expansion, value_for_constant_padding);
            break;
        case mirror:
            padded_image = generate_mirror_padding_image(std::forward<ExecutionPolicy>(execution_policy), input, width_expansion, height_expansion, value_for_constant_padding);
            break;
        case replicate:
            padded_image = generate_replicate_padding_image(std::forward<ExecutionPolicy>(execution_policy), input, width_expansion, height_expansion, value_for_constant_padding);
            break;
        }
        return padded_image;
    }

    //  imgaussfilt template function implementation
    //  https://codereview.stackexchange.com/q/292985/231235
    //  giving filter_size a default value of 0, and having the function compute an appropriate size unless the user specifies a positive value.
    template<typename ElementT, typename SigmaT = double, std::integral SizeT = int>
    requires(std::floating_point<SigmaT> || std::integral<SigmaT>)
    constexpr static auto imgaussfilt(
        const Image<ElementT>& input,
        SigmaT sigma,
        SizeT filter_size = 0,
        BoundaryCondition boundaryCondition = BoundaryCondition::mirror,
        ElementT value_for_constant_padding = ElementT{})
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        if (filter_size == 0)
        {
            return imgaussfilt(
                std::execution::seq,
                input,
                sigma,
                sigma,
                static_cast<int>(computeFilterSizeFromSigma(sigma)),
                static_cast<int>(computeFilterSizeFromSigma(sigma)),
                boundaryCondition,
                value_for_constant_padding);
        }
        return imgaussfilt(
                        std::execution::seq,
                        input,
                        sigma,
                        sigma,
                        filter_size,
                        filter_size,
                        boundaryCondition,
                        value_for_constant_padding);
    }

    //  imgaussfilt template function implementation
    //  https://codereview.stackexchange.com/q/292985/231235
    template<class ExecutionPolicy, typename ElementT, typename SigmaT = double, std::integral SizeT = int>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)&&
            (std::floating_point<SigmaT> || std::integral<SigmaT>)
    constexpr static auto imgaussfilt(
        ExecutionPolicy&& execution_policy,
        const Image<ElementT>& input,
        const SigmaT sigma1,
        const SigmaT sigma2,
        const SizeT filter_size1,
        const SizeT filter_size2,
        const BoundaryCondition boundaryCondition = BoundaryCondition::mirror,
        const ElementT value_for_constant_padding = ElementT{})
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        Image<ElementT> padded_image = generate_padded_image(
            execution_policy, input,
            filter_size1,
            filter_size2,
            boundaryCondition,
            value_for_constant_padding
        );
        auto filter_mask_x = gaussianFigure1D(
                                    filter_size1,
                                    (static_cast<double>(filter_size1) + 1.0) / 2.0,
                                    sigma1);
        auto sum_result = sum(execution_policy, filter_mask_x);
        filter_mask_x = divides(filter_mask_x, sum_result);             //  Normalization
        auto output = conv2(padded_image, filter_mask_x, true);
        auto filter_mask_y = transpose(gaussianFigure1D(
                                        filter_size2,
                                        (static_cast<double>(filter_size2) + 1.0) / 2.0,
                                        sigma2));
        sum_result = sum(filter_mask_y);
        filter_mask_y = divides(filter_mask_y, sum_result);             //  Normalization
        output = conv2(output, filter_mask_y, true);
        output = subimage(output, input.getWidth(), input.getHeight(), static_cast<double>(output.getWidth()) / 2.0, static_cast<double>(output.getHeight()) / 2.0);
        return output;
    }

    //  difference_of_gaussian template function implementation
    template<typename ElementT, typename SigmaT = double>
    requires(std::floating_point<SigmaT> || std::integral<SigmaT>)
    constexpr static auto difference_of_gaussian(
        const Image<ElementT>& input,
        const SigmaT sigma1,
        const SigmaT sigma2)
    {
        return subtract(
            imgaussfilt(input, sigma1, static_cast<int>(computeFilterSizeFromSigma(sigma1))),
            imgaussfilt(input, sigma2, static_cast<int>(computeFilterSizeFromSigma(sigma2)))
            );
    }

    //  get_center_location template function implementation (with Execution Policy)
    template<typename ElementT, class ExecutionPolicy>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto get_center_location(
        ExecutionPolicy&& execution_policy,
        const Image<ElementT>& input
    )
    {
        const auto& image_size = input.getSize();
        auto center_location = image_size;
        std::transform(
            execution_policy,
            std::ranges::cbegin(image_size),
            std::ranges::cend(image_size),
            std::ranges::begin(center_location),
            [&](auto&& size) { return static_cast<double>(size) / 2.0; }
        );
        return center_location;
    }

    //  get_center_pixel template function implementation (with Execution Policy)
    //  return center pixel value for N-dimensional image
    template<typename ElementT, class ExecutionPolicy>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto get_center_pixel(
        ExecutionPolicy&& execution_policy,
        const Image<ElementT>& input
    )
    {
        return input.at_without_boundary_check(get_center_location(std::forward<ExecutionPolicy>(execution_policy), input));
    }

    //  get_center_pixel template function implementation
    template<typename ElementT>
    constexpr static auto get_center_pixel(
        const Image<ElementT>& input
    )
    {
        return get_center_pixel(std::execution::seq, input);
    }

    //  windowed_filter template function implementation
    template<class ElementT, class ExecutionPolicy, class Filter, std::ranges::input_range SizeRange>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto windowed_filter(
        ExecutionPolicy&& execution_policy,
        const Image<ElementT>& input,
        const SizeRange& window_sizes,
        const Filter filter,
        const BoundaryCondition boundaryCondition = BoundaryCondition::mirror,
        const ElementT& value_for_constant_padding = ElementT{},
        const std::function<void(double)> progress_callback = nullptr
    )
    {
        const std::size_t dim = input.getDimensionality();
        
        if (dim != window_sizes.size())
        {
            throw std::runtime_error("windowed_filter: window_sizes.size() must match input dimensionality");
        }
        auto padded_image = input;
        if (dim == 2)                   //  padding algorithm supported
        {
            padded_image = generate_padded_image(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                window_sizes[0],
                window_sizes[1],
                boundaryCondition,
                value_for_constant_padding
            );
        }
        // Iterate over all elements in the original image
        const std::size_t total_elements = input.count();
        #ifndef USE_OPENMP
        auto indices_view = std::views::iota(std::size_t{ 0 }, total_elements);
        // Progress tracking variables
        std::atomic<std::size_t> processed_count = 0;
        std::atomic<std::size_t> next_report = 0;
        std::mutex report_mutex;
        const std::size_t report_step = std::max<std::size_t>(1, total_elements / 100);
        auto process_element = [&](auto&& idx) {
            // Convert linear index to N-D indices (original image)
            auto indices = linear_index_to_indices(idx, input.getSize());

            // Convert to padded indices
            std::vector<std::size_t> padded_indices = indices;
            if (dim == 2)
            {
                for (std::size_t i = 0; i < indices.size(); ++i)
                {
                    padded_indices[i] = (indices[i] + window_sizes[i]);
                }
            }

            // Extract window
            auto window = subimage(
                std::forward<ExecutionPolicy>(execution_policy),
                padded_image,
                window_sizes,
                padded_indices
            );

            // Apply filter and store result
            padded_image.at(padded_indices) =
                std::invoke(filter, window);

            // Progress reporting
            if (progress_callback) {
                auto count = ++processed_count;
                if (count >= next_report.load(std::memory_order_relaxed)) {
                    std::lock_guard lock(report_mutex);
                    if (count >= next_report.load(std::memory_order_relaxed)) {
                        double progress = static_cast<double>(count) / total_elements;
                        progress_callback(progress);
                        next_report.store(count + report_step, std::memory_order_relaxed);
                    }
                }
            }
            };
        std::for_each(
            std::forward<ExecutionPolicy>(execution_policy),
            indices_view.begin(),
            indices_view.end(),
            process_element
        );
        #else
        #pragma omp parallel for
        for (std::size_t idx = 0; idx < total_elements; ++idx)
        {
            // Convert linear index to N-D indices (original image)
            auto indices = linear_index_to_indices(idx, input.getSize());

            // Convert to padded indices
            std::vector<std::size_t> padded_indices = indices;
            if (dim == 2)
            {
                for (std::size_t i = 0; i < indices.size(); ++i)
                {
                    padded_indices[i] = (indices[i] + window_sizes[i]);
                }
            }

            // Extract window
            auto window = subimage(
                std::forward<ExecutionPolicy>(execution_policy),
                padded_image,
                window_sizes,
                padded_indices
            );

            // Apply filter and store result
            padded_image.at_without_boundary_check(padded_indices) =
                std::invoke(filter, window);
        }
        #endif
        // Crop padded image to original size
        std::vector<std::size_t> original_sizes = input.getSize();
        std::vector<std::size_t> centers;
        for (auto& s : padded_image.getSize())
        {
            centers.emplace_back(s / 2);  // Center of padded image
        }
        auto output = subimage(std::forward<ExecutionPolicy>(execution_policy), padded_image, original_sizes, centers);
        return output;
    }

    //  bilateral_filter_detail template function implementation
    template<
        typename ElementT,
        class RangeKernel,
        class RangeDistance,
        class FloatingType = double>
    requires(std::invocable<RangeKernel, ElementT>)
    constexpr static auto bilateral_filter_detail(
        const Image<ElementT>& input,
        const RangeKernel range_kernel,
        const RangeDistance range_distance,
        const Image<FloatingType>& spatial_kernel_weight
    )
    {
        const ElementT center_pixel = get_center_pixel(std::execution::seq, input);
        std::invoke_result_t<RangeKernel, std::invoke_result_t<RangeDistance, ElementT, ElementT>> sum{};
        if constexpr (is_complex<std::invoke_result_t<RangeKernel, std::invoke_result_t<RangeDistance, ElementT, ElementT>>>::value)
        {
            std::invoke_result_t<RangeKernel, std::invoke_result_t<RangeDistance, ElementT, ElementT>> weight_sum{};
            const std::size_t total_elements = input.count();
            for (std::size_t idx = 0; idx < total_elements; ++idx)
            {
                // Convert linear index to N-D indices (original image)
                auto indices = linear_index_to_indices(idx, input.getSize());
                auto range_kernel_result = std::invoke(
                    range_kernel,
                    std::invoke(range_distance, input.at_without_boundary_check(indices), center_pixel)
                );
                auto spatial_kernel_result = std::complex{ spatial_kernel_weight.at_without_boundary_check(indices) , 0.0 };          //  Precompute spatial_kernel
                sum += input.at_without_boundary_check(indices) * range_kernel_result * spatial_kernel_result;
                weight_sum += range_kernel_result * spatial_kernel_result;
            }
            return static_cast<ElementT>(sum / weight_sum);
        }
        else
        {
            FloatingType weight_sum{};
            const std::size_t total_elements = input.count();
            for (std::size_t idx = 0; idx < total_elements; ++idx)
            {
                // Convert linear index to N-D indices (original image)
                auto indices = linear_index_to_indices(idx, input.getSize());
                auto range_kernel_result = std::invoke(
                    range_kernel,
                    std::invoke(range_distance, input.at_without_boundary_check(indices), center_pixel)
                );
                auto spatial_kernel_result = spatial_kernel_weight.at_without_boundary_check(indices);          //  Precompute spatial_kernel
                sum += input.at_without_boundary_check(indices) * range_kernel_result * spatial_kernel_result;
                weight_sum += range_kernel_result * spatial_kernel_result;
            }
            return static_cast<ElementT>(sum / weight_sum);
        }
    }

    //  bilateral_filter_spatial_kernel template function implementation
    template<
        std::ranges::input_range SizeRange,
        class SpatialKernel,
        class SpatialDistance,
        class FloatingType = double
    >
    requires(std::same_as<std::ranges::range_value_t<SizeRange>, std::size_t> ||
             std::same_as<std::ranges::range_value_t<SizeRange>, int>)
    constexpr static auto bilateral_filter_spatial_kernel(
        const SizeRange& window_sizes,
        const SpatialKernel spatial_kernel,
        const SpatialDistance spatial_distance
        )
    {
        Image<FloatingType> output(window_sizes);
        const auto& center_location = get_center_location(std::execution::seq, output);
        const std::size_t total_elements = output.count();
        for (std::size_t idx = 0; idx < total_elements; ++idx)
        {
            const auto& indices = linear_index_to_indices(idx, output.getSize());
            const auto distance = std::invoke(spatial_distance, indices, center_location);
            output.at_without_boundary_check(indices) =
                std::invoke(spatial_kernel, distance);
        }
        return output;
    }

    //  bilateral_filter template function implementation
    template<typename ElementT, class ExecutionPolicy, class RangeKernel, class SpatialKernel, std::ranges::input_range SizeRange>
    requires((std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>) and
             (std::floating_point<ElementT> || std::integral<ElementT> || is_complex<ElementT>::value) and
             (std::same_as<std::ranges::range_value_t<SizeRange>, std::size_t> ||
              std::same_as<std::ranges::range_value_t<SizeRange>, int>))
    constexpr static auto bilateral_filter(
        ExecutionPolicy&& execution_policy,
        const Image<ElementT>& input,
        const SizeRange& window_sizes,
        const RangeKernel range_kernel,
        const SpatialKernel spatial_kernel,
        const BoundaryCondition boundaryCondition = BoundaryCondition::mirror,
        std::function<void(double)> progress_callback = nullptr,
        const ElementT value_for_constant_padding = ElementT{}
        )
    {
        return windowed_filter(
            std::forward<ExecutionPolicy>(execution_policy),
            input,
            window_sizes,
            [&](auto&& window)
            {
                return bilateral_filter_detail(
                    window,
                    range_kernel,
                    [&](auto&& element1, auto&& element2)
                    {
                        return std::sqrt(std::pow(element1 - element2, 2.0));
                    },
                    bilateral_filter_spatial_kernel(
                        window_sizes,
                        spatial_kernel,
                        //  Reference: https://stackoverflow.com/a/62932369/6667035
                        [&]
                        <std::ranges::input_range Range1, std::ranges::input_range Range2>
                        (const Range1& vector1, const Range2& vector2)         //  Spatial Distance Function, L2 norm
                        {
                            if (vector1.size() != vector2.size())
                            {
                                throw std::runtime_error("Size not matched!");
                            }
                            double sum_of_squares = 0.0;
                            for (std::size_t i = 0; i < vector1.size(); i++)
                            {
                                sum_of_squares += std::pow(vector1[i] - vector2[i], 2.0);
                            }
                            return std::sqrt(sum_of_squares);
                        }
                    )
                );
            },
            boundaryCondition,
            value_for_constant_padding,
            progress_callback
        );
    }

    //  bilateral_filter template function implementation
    template<class ExecutionPolicy, typename ElementT, class RangeKernel, class SpatialKernel, std::ranges::input_range SizeRange>
    requires ((std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>) and
              ((std::same_as<ElementT, RGB>) ||
              (std::same_as<ElementT, RGB_DOUBLE>) ||
              (std::same_as<ElementT, HSV>) ||
              (is_MultiChannel<ElementT>::value)) and
              (std::same_as<std::ranges::range_value_t<SizeRange>, std::size_t> ||
               std::same_as<std::ranges::range_value_t<SizeRange>, int>))
    constexpr static auto bilateral_filter(
        ExecutionPolicy&& execution_policy,
        const Image<ElementT>& input,
        const SizeRange& window_sizes,
        RangeKernel range_kernel,
        SpatialKernel spatial_kernel,
        BoundaryCondition boundaryCondition = BoundaryCondition::mirror,
        std::function<void(double)> progress_callback = nullptr
    )
    {
        return apply_each(
            input,
            [&](auto&& each_plane)
            {
                return bilateral_filter(
                    std::forward<ExecutionPolicy>(execution_policy),
                    each_plane,
                    window_sizes,
                    range_kernel,
                    spatial_kernel,
                    boundaryCondition,
                    progress_callback
                    );
            });
    }

    //  bilateral_filter template function implementation (without Execution Policy)
    template<typename ElementT, class RangeKernel, class SpatialKernel, std::ranges::input_range SizeRange>
    requires(std::same_as<std::ranges::range_value_t<SizeRange>, std::size_t> ||
             std::same_as<std::ranges::range_value_t<SizeRange>, int>)
    constexpr static auto bilateral_filter(
        const Image<ElementT>& input,
        const SizeRange& window_sizes,
        const RangeKernel range_kernel,
        const SpatialKernel spatial_kernel,
        const BoundaryCondition boundaryCondition = BoundaryCondition::mirror
    )
    {
        return bilateral_filter(
            std::execution::seq,
            input,
            window_sizes,
            range_kernel,
            spatial_kernel,
            boundaryCondition
        );
    }

    //  increase_intensity template function implementation
    template<typename ElementT, class TimesT>
    requires (std::same_as<ElementT, RGB> || std::same_as<ElementT, RGB_DOUBLE>)
    constexpr static auto increase_intensity(const Image<ElementT>& input, const TimesT times)
    {
        auto HSV_image = TinyDIP::rgb2hsv(input);
        auto Hplane = TinyDIP::getHplane(HSV_image);
        auto Splane = TinyDIP::getSplane(HSV_image);
        auto Vplane = TinyDIP::getVplane(HSV_image);
        return hsv2rgb(constructHSV(Hplane, Splane, multiplies(Vplane, times)));
    }

    //  draw_point template function implementation
    template<typename ElementT, std::size_t dimension = 2>
    constexpr static auto draw_point(
        const Image<ElementT>& input,
        Point<dimension> point,
        ElementT draw_value = ElementT{},
        std::size_t radius = 3
        )
    {
        auto point_x = point.p[0];
        auto point_y = point.p[1];
        auto output = input;
        auto height = input.getHeight();
        auto width = input.getWidth();
        #pragma omp parallel for collapse(2)
        for (std::size_t y = point_y - radius; y <= point_y + radius; ++y)
        {
            for (std::size_t x = point_x - radius; x <= point_x + radius; ++x)
            {
                if (x >= width || y >= height)
                {
                    continue;
                }
                if(std::pow(static_cast<double>(x) - static_cast<double>(point_x), 2.0) +
                   std::pow(static_cast<double>(y) - static_cast<double>(point_y), 2.0) < std::pow(radius, 2))
                {
                    output.at_without_boundary_check(x, y) = draw_value;
                }
            }
        }
        return output;
    }

    //  draw_points template function implementation
    template<typename ElementT, std::size_t dimension = 2>
    constexpr static auto draw_points(
        const Image<ElementT>& input,
        std::vector<Point<dimension>> points,
        ElementT draw_value = ElementT{},
        std::size_t radius = 3
    )
    {
        auto output = input;
        for (auto&& each_point : points)
        {
            output = draw_point(output, each_point, draw_value, radius);
        }
        return output;
    }
    
    //  draw_circle template function implementation
    //  https://codereview.stackexchange.com/q/293417/231235
    //  Test: https://godbolt.org/z/7zKfhG3x9
    //  Test with output.set: https://godbolt.org/z/1GYdrbs5q
    template<typename ElementT>
    constexpr static auto draw_circle(
        const Image<ElementT>& input,
        std::tuple<std::size_t, std::size_t> central_point,
        std::size_t radius = 2,
        ElementT draw_value = ElementT{}
    )
    {
        if (input.getDimensionality() != 2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        auto point_x = std::get<0>(central_point);
        auto point_y = std::get<1>(central_point);
        auto output = input;
        auto height = input.getHeight();
        auto width = input.getWidth();
        if (radius <= 0)
        {
            // early out avoids y going negative in loop
            return output;
        }
        for (std::ptrdiff_t x = 0, y = radius; x <= y; x++)
        {
            // try to decrement y, then accept or revert
            y--;
            if (x * x + y * y < radius * radius)
            {
                y++;
            }
            // do nothing if out of bounds, otherwise draw
            output.set(std::make_tuple(point_x + x, point_y + y), draw_value);
            output.set(std::make_tuple(point_x - x, point_y + y), draw_value);
            output.set(std::make_tuple(point_x + x, point_y - y), draw_value);
            output.set(std::make_tuple(point_x - x, point_y - y), draw_value);
            output.set(std::make_tuple(point_x + y, point_y + x), draw_value);
            output.set(std::make_tuple(point_x - y, point_y + x), draw_value);
            output.set(std::make_tuple(point_x + y, point_y - x), draw_value);
            output.set(std::make_tuple(point_x - y, point_y - x), draw_value);
        }
        return output;
    }

    //  to_complex template function implementation
    template<class ExecutionPolicy, typename ElementT, std::size_t Size>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto to_complex(ExecutionPolicy&& execution_policy, const MultiChannel<ElementT, Size> input)
    {
        std::array<std::complex<ElementT>, Size> output;
        std::transform(
            std::forward<ExecutionPolicy>(execution_policy),
            std::ranges::cbegin(input.channels),
            std::ranges::cend(input.channels),
            std::ranges::begin(output),
            [&](auto&& element) { return static_cast<std::complex<ElementT>>(element); }
        );
        return MultiChannel<std::complex<ElementT>, Size>{output};
    }

    //  to_complex template function implementation
    template<typename ElementT, std::size_t Size>
    constexpr static auto to_complex(const MultiChannel<ElementT, Size> input)
    {
        return to_complex(std::execution::seq, input);
    }
    
    //  to_complex template function implementation
    template<typename ElementT = double>
    constexpr static auto to_complex(const Image<ElementT>& input)
    {
        if constexpr ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>) || is_MultiChannel<ElementT>::value)
        {
            return apply_each(input, [&](auto&& planes) { return to_complex(planes); });
        }
        else
        {
            Image<std::complex<ElementT>> output_image(
                TinyDIP::recursive_transform<1>([](auto&& _input) { return std::complex{ _input, ElementT{} }; }, input.getImageData()),
                input.getSize()
            );
            return output_image;
        }
    }

    //  to_double template function implementation (with Execution Policy)
    template<class ExecutionPolicy, typename ElementT = double>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto to_double(ExecutionPolicy&& execution_policy, const Image<ElementT>& input)
    {
        if constexpr ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>))
        {
            return apply_each(input, [&](auto&& planes) { return to_double(std::forward<ExecutionPolicy>(execution_policy), planes); });
        }
        else
        {
            Image<double> output_image(
                TinyDIP::recursive_transform<1>(std::forward<ExecutionPolicy>(execution_policy), [](auto&& _input) { return static_cast<double>(_input); }, input.getImageData()),
                input.getSize()
            );
            return output_image;
        }
    }

    //  to_double template function implementation
    template<typename ElementT = double>
    constexpr static auto to_double(const Image<ElementT>& input)
    {
        return to_double(std::execution::seq, input);
    }

    //  to_string template function implementation
    template<typename ElementT = double>
    constexpr static auto to_string(const Image<ElementT>& input)
    {
        Image<std::string> output_image(
            TinyDIP::recursive_transform<1>([](auto&& _input) { return std::to_string(_input); }, input.getImageData()),
            input.getSize()
        );
        return output_image;
    }

    //  cot template function implementation
    template<typename ElementT = double>
    constexpr static auto cot(const Image<ElementT>& input)
    {
        if constexpr ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>) || is_MultiChannel<ElementT>::value)
        {
            return apply_each(input, [&](auto&& planes) { return cot(planes); });
        }
        else if constexpr (is_complex<ElementT>::value)
        {
            Image<ElementT> output_image(
                TinyDIP::recursive_transform<1>([&](auto&& _input) { return std::complex{ static_cast<ElementT>(1) } / std::tan(_input); }, input.getImageData()),
                input.getSize()
            );
            return output_image;
        }
        else
        {
            Image<ElementT> output_image(
                TinyDIP::recursive_transform<1>([&](auto&& _input) { return 1 / std::tan(_input); }, input.getImageData()),
                input.getSize()
            );
            return output_image;
        }
    }

    namespace SIFT_impl {
        /*  is_it_extremum template function implementation
            input1, input2 and input3 are 3 * 3 images. If the center pixel (at location (1,1) ) of input2 is the largest / smallest one, 
            return true; otherwise, return false.
            Test: https://godbolt.org/z/Kb34EW5Yj
            Test [Updated]: https://godbolt.org/z/bsG96hEn9
        */
        template<typename ElementT>
        constexpr static bool is_it_extremum(
            const Image<ElementT>& input1,
            const Image<ElementT>& input2,
            const Image<ElementT>& input3,
            const double threshold = 0.03)
        {
            for (auto* input : { &input1, &input2, &input3 })
            {
                if (input->getDimensionality() != 2)
                {
                    throw std::runtime_error("Unsupported dimensionality");
                }
                if (input->getWidth() != 3 || input->getHeight() != 3)
                {
                    throw std::runtime_error("Unsupported size");
                }
            }
            auto center_pixel = input2.at(static_cast<std::size_t>(1), static_cast<std::size_t>(1));
            auto input2_img_data = input2.getImageData();
            input2_img_data.erase(input2_img_data.begin() + 4);                         //  https://stackoverflow.com/a/875117/6667035
            if (std::abs(center_pixel) <= threshold)
            {
                return false;
            }
            auto image_datas = { input1.getImageData(), input2_img_data, input3.getImageData() };
            if (recursive_all_of<2>(image_datas, [&](auto&& i) { return center_pixel > i; }))
            {
                return true;
            }
            if (recursive_all_of<2>(image_datas, [&](auto&& i) { return center_pixel < i; }))
            {
                return true;
            }
            return false;
        }

        //  keypoint_refinement template function implementation
        //  refine the given keypoint's location using Taylor expansion
        template<typename ElementT = double>
        constexpr static auto keypoint_refinement(
            const Image<ElementT>& input,
            const std::tuple<std::size_t, std::size_t>& point
            )
        {
            //  Calculate the gradient at the keypoint (x, y)
            const ElementT first_derivative_x = (input.at(std::size_t{ 2 }, std::size_t{ 1 }) - input.at(std::size_t{ 0 }, std::size_t{ 1 })) / 2.0;
            const ElementT first_derivative_y = (input.at(std::size_t{ 1 }, std::size_t{ 2 }) - input.at(std::size_t{ 1 }, std::size_t{ 0 })) / 2.0;
            const ElementT second_derivative_x = (input.at(std::size_t{ 2 }, std::size_t{ 1 }) + input.at(std::size_t{ 0 }, std::size_t{ 1 }) - 2.0 * input.at(std::size_t{ 1 }, std::size_t{ 1 }));
            const ElementT second_derivative_y = (input.at(std::size_t{ 1 }, std::size_t{ 2 }) + input.at(std::size_t{ 1 }, std::size_t{ 0 }) - 2.0 * input.at(std::size_t{ 1 }, std::size_t{ 1 }));
            const ElementT second_derivative_xy = (input.at(std::size_t{ 2 }, std::size_t{ 2 }) - input.at(std::size_t{ 2 }, std::size_t{ 0 }) - input.at(std::size_t{ 0 }, std::size_t{ 2 }) + input.at(std::size_t{ 0 }, std::size_t{ 0 })) / 4.0;
            const ElementT A = -second_derivative_x / (second_derivative_x * second_derivative_y - second_derivative_xy * second_derivative_xy);
            const ElementT B = second_derivative_xy / (second_derivative_x * second_derivative_y - second_derivative_xy * second_derivative_xy);
            const ElementT C = B;
            const ElementT D = -second_derivative_y / (second_derivative_x * second_derivative_y - second_derivative_xy * second_derivative_xy);
            const ElementT offset_x = A * first_derivative_x + B * first_derivative_y;
            const ElementT offset_y = C * first_derivative_x + D * first_derivative_y;
            return std::make_tuple(
                static_cast<ElementT>(std::get<0>(point)) + offset_x,
                static_cast<ElementT>(std::get<1>(point)) + offset_y);
        }

        //  keypoint_filtering template function implementation
        template<typename ElementT = double>
        constexpr static bool keypoint_filtering(
            const Image<ElementT>& input,
            const ElementT contrast_check_threshold = 8,
            const ElementT edge_response_threshold = 12.1)
        {
            //  Calculate Hessian matrix at the keypoint (x, y)
            const ElementT second_derivative_x = (input.at(std::size_t{ 2 }, std::size_t{ 1 }) + input.at(std::size_t{ 0 }, std::size_t{ 1 }) - 2.0 * input.at(std::size_t{ 1 }, std::size_t{ 1 }));                    //  D_{xx}
            const ElementT second_derivative_y = (input.at(std::size_t{ 1 }, std::size_t{ 2 }) + input.at(std::size_t{ 1 }, std::size_t{ 0 }) - 2.0 * input.at(std::size_t{ 1 }, std::size_t{ 1 }));                    //  D_{yy}
            const ElementT second_derivative_xy = (input.at(std::size_t{ 2 }, std::size_t{ 2 }) - input.at(std::size_t{ 2 }, std::size_t{ 0 }) - input.at(std::size_t{ 0 }, std::size_t{ 2 }) + input.at(std::size_t{ 0 }, std::size_t{ 0 })) / 4.0;  //  D_{xy}
            const ElementT trace_Hessian_matrix = second_derivative_x + second_derivative_y;
            const ElementT determinant_Hessian_matrix = second_derivative_x * second_derivative_y - second_derivative_xy * second_derivative_xy;
            const ElementT principal_curvature_ratio = trace_Hessian_matrix * trace_Hessian_matrix / determinant_Hessian_matrix;
            if (input.at(std::size_t{ 1 }, std::size_t{ 1 }) <= contrast_check_threshold || principal_curvature_ratio >= edge_response_threshold)
            {
                return false;
            }
            return true;
        }

        //  mapping_point template function implementation
        template<class FloatingType = double>
        constexpr static auto mapping_point(
            const std::tuple<FloatingType, FloatingType>& input_location,
            const std::size_t input_width,
            const std::size_t input_height,
            const std::size_t target_width,
            const std::size_t target_height)
        {
            FloatingType width_percentage = static_cast<FloatingType>(std::get<0>(input_location)) / static_cast<FloatingType>(input_width);
            FloatingType height_percentage = static_cast<FloatingType>(std::get<1>(input_location)) / static_cast<FloatingType>(input_height);
            return Point<2>{
                static_cast<std::size_t>(width_percentage * static_cast<FloatingType>(target_width)),
                static_cast<std::size_t>(height_percentage * static_cast<FloatingType>(target_height))
            };
        }

        //  find_local_extrema template function implementation
        template<typename ElementT>
        constexpr static auto find_local_extrema(
            Image<ElementT> input1,
            Image<ElementT> input2,
            Image<ElementT> input3,
            std::size_t octave_index,
            std::size_t scale_index,
            ElementT contrast_check_threshold = 8,
            ElementT edge_response_threshold = 12.1)
        {
            if (input1.getDimensionality() != 2)
            {
                throw std::runtime_error("Unsupported dimension!");
            }
            if (input2.getDimensionality() != 2)
            {
                throw std::runtime_error("Unsupported dimension!");
            }
            if (input3.getDimensionality() != 2)
            {
                throw std::runtime_error("Unsupported dimension!");
            }
            if (input1.getSize() != input2.getSize())
            {
                throw std::runtime_error("Size mismatched!");
            }
            if (input2.getSize() != input3.getSize())
            {
                throw std::runtime_error("Size mismatched!");
            }
            const int block_size = 3;
            std::vector<std::tuple<std::size_t, std::size_t, ElementT, ElementT>> output;
            auto width = input1.getWidth() - 1;
            auto height = input1.getHeight() - 1;
            #pragma omp parallel for collapse(2)
            for (std::size_t y = 1; y < height; ++y)
            {
                for (std::size_t x = 1; x < width; ++x)
                {
                    auto subimage1 = subimage(input1, block_size, block_size, x, y);
                    auto subimage2 = subimage(input2, block_size, block_size, x, y);
                    auto subimage3 = subimage(input3, block_size, block_size, x, y);
                    if (is_it_extremum(subimage1, subimage2, subimage3, contrast_check_threshold) && keypoint_filtering(subimage2, contrast_check_threshold, edge_response_threshold))
                    {
                        auto new_location = keypoint_refinement(subimage2, std::make_tuple(x, y));
                        output.emplace_back(
                            std::make_tuple(
                                octave_index,
                                scale_index,
                                std::get<0>(new_location),
                                std::get<1>(new_location)));
                    }
                }
            }
            return output;
        }

        //  generate_octave template function implementation
        template<typename ElementT, typename SigmaT = double>
        requires(std::floating_point<SigmaT> || std::integral<SigmaT>)
        constexpr static auto generate_octave(
            const Image<ElementT>& input,
            std::size_t number_of_scale_levels = 5,
            SigmaT initial_sigma = 1.6,
            double k = 1.4142135623730950488016887242097)
        {
            std::vector<Image<ElementT>> difference_of_gaussian_images;
            difference_of_gaussian_images.resize(number_of_scale_levels - 1);
            #pragma omp parallel for
            for (int index = 0; index < number_of_scale_levels - 1; ++index)
            {
                difference_of_gaussian_images[index] = (difference_of_gaussian(input, initial_sigma * std::pow(k, index), initial_sigma * std::pow(k, index + 1)));
            }
            return difference_of_gaussian_images;
        }

        //  get_potential_keypoint template function implementation
        template<typename ElementT = double, typename SigmaT = double>
        requires(   (std::floating_point<ElementT> || std::integral<ElementT>) &&
                    (std::floating_point<SigmaT> || std::integral<SigmaT>))
        constexpr static auto get_potential_keypoint(
            const Image<ElementT>& input,
            const std::size_t octaves_count = 4,
            const std::size_t number_of_scale_levels = 5,
            const SigmaT initial_sigma = 1.6,           //  sigma is selected to 1.6 based on paper https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf page 10
            const double k = std::numbers::sqrt2_v<double>,
            ElementT contrast_check_threshold = 8,
            ElementT edge_response_threshold = 12.1)
        {
            if (input.getDimensionality() != 2)
            {
                throw std::runtime_error("Unsupported dimension!");
            }
            //  Generate octaves
            std::vector<std::vector<Image<ElementT>>> octaves;
            octaves.reserve(octaves_count);
            Image<ElementT> base_image = input;
            for (std::size_t octave_index = 0; octave_index < octaves_count; ++octave_index)
            {
                octaves.emplace_back(generate_octave(base_image, number_of_scale_levels, initial_sigma, k));
                base_image = copyResizeBicubic(
                    base_image,
                    static_cast<std::size_t>(static_cast<double>(base_image.getWidth()) / 2.0),
                    static_cast<std::size_t>(static_cast<double>(base_image.getHeight()) / 2.0));
            }

            //  Find potential KeyPoints
            /*  KeyPoint structure: octave_index, scale_index + 1, location_x, location_y
            */
            std::vector<std::tuple<std::size_t, std::size_t, ElementT, ElementT>> keypoints;
            #pragma omp parallel for
            for (int octave_index = 0; octave_index < octaves_count; ++octave_index)
            {
                auto each_octave = octaves[octave_index];
                #pragma omp parallel for
                for (int scale_index = 0; scale_index < each_octave.size() - 2; ++scale_index)
                {
                    //if `append_range` function is supported
                    #ifdef __cpp_lib_containers_ranges
                    keypoints.append_range(
                        find_local_extrema(
                            each_octave[scale_index],
                            each_octave[scale_index + 1],
                            each_octave[scale_index + 2],
                            octave_index,
                            scale_index,
                            contrast_check_threshold,
                            edge_response_threshold)
                    );
                    #else
                    for (auto&& element : find_local_extrema(
                        each_octave[scale_index],
                        each_octave[scale_index + 1],
                        each_octave[scale_index + 2],
                        octave_index,
                        scale_index,
                        contrast_check_threshold,
                        edge_response_threshold))
                    {
                        keypoints.emplace_back(element);
                    }
                    #endif
                }
            }

            //  mapping keypoints in different scale to the original image
            std::vector<Point<2>> mapped_keypoints;
            for (auto&& each_keypoint : keypoints)
            {
                auto input_width = octaves[std::get<0>(each_keypoint)][0].getWidth();
                auto input_height = octaves[std::get<0>(each_keypoint)][0].getHeight();
                mapped_keypoints.emplace_back(
                    mapping_point(
                        std::make_tuple(std::get<2>(each_keypoint), std::get<3>(each_keypoint)),
                        input_width,
                        input_height,
                        input.getWidth(),
                        input.getHeight()
                    ));
            }

            return mapped_keypoints;
        }

        //  compute_each_pixel_orientation template function implementation
        /*  input is 3 * 3 image, calculate the gradient magnitude
        *    M(1, 1) = ((input(2, 1) - input(0, 1))^(2) + (input(1, 2) - input(1, 0))^(2))^(1/2)
        *   orientation
        *    θ(1, 1) = tan^(-1)((input(1, 2) - input(1, 0)) / (input(2, 1) - input(0, 1)))
        *   the value range of orientation is 0° ~ 360°
        */
        template<typename ElementT, class FloatingType = double>
        constexpr static auto compute_each_pixel_orientation(const Image<ElementT>& input)
        {
            if (input.getDimensionality() != 2)
            {
                throw std::runtime_error("Unsupported dimension!");
            }
            if (input.getWidth() != 3 || input.getHeight() != 3)
                throw std::runtime_error("Input size error!");
            FloatingType gradient_magnitude =
                std::sqrt(
                    std::pow((static_cast<FloatingType>(input.at_without_boundary_check(static_cast<std::size_t>(2), static_cast<std::size_t>(1))) - static_cast<FloatingType>(input.at_without_boundary_check(static_cast<std::size_t>(0), static_cast<std::size_t>(1)))), 2.0) +
                    std::pow((static_cast<FloatingType>(input.at_without_boundary_check(static_cast<std::size_t>(1), static_cast<std::size_t>(2))) - static_cast<FloatingType>(input.at_without_boundary_check(static_cast<std::size_t>(1), static_cast<std::size_t>(0)))), 2.0)
                );
            double orientation = std::atan2(
                    static_cast<FloatingType>(input.at_without_boundary_check(static_cast<std::size_t>(1), static_cast<std::size_t>(2))) - static_cast<FloatingType>(input.at_without_boundary_check(static_cast<std::size_t>(1), static_cast<std::size_t>(0))),
                    static_cast<FloatingType>(input.at_without_boundary_check(static_cast<std::size_t>(2), static_cast<std::size_t>(1))) - static_cast<FloatingType>(input.at_without_boundary_check(static_cast<std::size_t>(0), static_cast<std::size_t>(1)))
                );
            orientation *= (180.0 / std::numbers::pi_v<FloatingType>);
            orientation += 180;
            return std::make_tuple(gradient_magnitude, orientation);
        }

        //  get_orientation_histogram template function implementation
        template<typename ElementT>
        requires((std::floating_point<ElementT> || std::integral<ElementT>))
        constexpr static auto get_orientation_histogram(
            const Image<ElementT>& input,
            std::tuple<std::size_t, std::size_t> point,
            std::size_t block_size = 3
        )
        {
            if (input.getDimensionality() != 2)
            {
                throw std::runtime_error("Unsupported dimension!");
            }
            std::vector<double> raw_histogram;
            raw_histogram.resize(37);
            for (std::size_t y = std::get<1>(point) - block_size; y <= std::get<1>(point) + block_size; ++y)
            {
                for (std::size_t x = std::get<0>(point) - block_size; x <= std::get<0>(point) + block_size; ++x)
                {
                    if (x >= input.getWidth() || y >= input.getHeight())
                    {
                        continue;
                    }
                    auto each_pixel_orientation = compute_each_pixel_orientation(subimage(input, 3, 3, x, y));
                    std::size_t bin_index = static_cast<std::size_t>(std::get<1>(each_pixel_orientation) / 10.0);
                    raw_histogram[bin_index] += std::get<0>(each_pixel_orientation);
                }
            }
            return raw_histogram;
        }

        //  get_keypoint_descriptor template function implementation
        template<typename ElementT, class FloatingType = double>
        requires((std::floating_point<ElementT> || std::integral<ElementT>))
        constexpr static auto get_keypoint_descriptor(
            const Image<ElementT>& input,
            const Point<2>& keypoint_location,
            const std::size_t block_size = 8
        )
        {
            //  Calculate Gradient Magnitudes and Orientations for 16x16 neighborhood
            Image<std::tuple<FloatingType, FloatingType>> pixel_orientations(block_size * 2, block_size * 2);
            const auto center_x = keypoint_location.p[0];
            const auto center_y = keypoint_location.p[1];
            for (std::size_t y = center_y - block_size; y < center_y + block_size; ++y)
            {
                for (std::size_t x = center_x - block_size; x < center_x + block_size; ++x)
                {
                    if (x >= input.getWidth() || y >= input.getHeight())
                    {
                        continue;
                    }
                    pixel_orientations.at(x - center_x + block_size, y - center_y + block_size) =
                        compute_each_pixel_orientation<ElementT, FloatingType>(subimage(input, 3, 3, x, y));
                }
            }
            
            //  Group Pixels into 4x4 Subregions
            Image<std::vector<double>> subregion_orientations(4, 4);
            for (std::size_t subregion_y = 0; subregion_y < 4; ++subregion_y)
            {
                for (std::size_t subregion_x = 0; subregion_x < 4; ++subregion_x)
                {
                    std::vector<double> raw_histogram;
                    raw_histogram.resize(8);
                    for (std::size_t y = 0; y < 4; ++y)
                    {
                        for (std::size_t x = 0; x < 4; ++x)
                        {
                            auto each_pixel_orientation =
                                pixel_orientations.at(subregion_x * 4 + x, subregion_y * 4 + y);
                            std::size_t bin_index = static_cast<std::size_t>(std::get<1>(each_pixel_orientation) / 45.0);
                            bin_index = bin_index == 8 ? 7 : bin_index;
                            raw_histogram[bin_index] += std::get<0>(each_pixel_orientation);
                        }
                    }
                    subregion_orientations.at(subregion_x, subregion_y) = raw_histogram;
                }
            }
            std::vector<double> feature_vector;
            feature_vector.reserve(128);
            for (std::size_t subregion_y = 0; subregion_y < subregion_orientations.getHeight(); ++subregion_y)
            {
                for (std::size_t subregion_x = 0; subregion_x < subregion_orientations.getWidth(); ++subregion_x)
                {
                    #ifdef USE_APPEND_RANGE
                    feature_vector.append_range(subregion_orientations.at(subregion_x, subregion_y));
                    #else
                    for (auto&& element : subregion_orientations.at(subregion_x, subregion_y))
                    {
                        feature_vector.emplace_back(element);
                    }
                    #endif
                }
            }
            return feature_vector;
        }
    }

}

#endif