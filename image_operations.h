/* Developed by Jimmy Hu */

#ifndef ImageOperations_H
#define ImageOperations_H

#include <concepts>
#include <execution>
#include <fstream>
#include <numbers>
#include <string>
#include "base_types.h"
#include "basic_functions.h"
#include "image.h"
#include <opencv2/opencv.hpp>

namespace TinyDIP
{
    //  all_of template function implementation
    template<typename ElementT, class UnaryPredicate>
    constexpr auto all_of(const Image<ElementT>& input, UnaryPredicate p)
    {
        return std::ranges::all_of(std::ranges::begin(input.getImageData()), std::ranges::end(input.getImageData()), p);
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

    template<typename ElementT>
    constexpr void check_size_same(const Image<ElementT>& x, const Image<ElementT>& y)
    {
        check_width_same(x, y);
        check_height_same(x, y);
    }

    //  zeros template function implementation
    template<typename ElementT, std::same_as<std::size_t>... Sizes>
    constexpr static auto zeros(Sizes... sizes)
    {
        auto output = Image<ElementT>(sizes...);
        return output;
    }

    //  ones template function implementation
    template<typename ElementT, std::same_as<std::size_t>... Sizes>
    constexpr static auto ones(Sizes... sizes)
    {
        auto output = zeros<ElementT>(sizes...);
        output.setAllValue(1);
        return output;
    }

    //  conv2 template function implementation
    template<typename ElementT>
    requires(std::floating_point<ElementT> || std::integral<ElementT> || is_complex<ElementT>::value)
    constexpr static auto conv2(const Image<ElementT>& x, const Image<ElementT>& y, bool is_size_same = false)
    {
        auto output = Image<ElementT>(x.getWidth() + y.getWidth() - 1, x.getHeight() + y.getHeight() - 1);
        for (std::size_t y1 = 0; y1 < x.getHeight(); ++y1) {
            auto* x_row = &(x.at(0, y1));
            for (std::size_t y2 = 0; y2 < y.getHeight(); ++y2) {
                auto* y_row = &(y.at(0, y2));
                auto* out_row = &(output.at(0, y1 + y2));
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
            #pragma omp parallel for collapse(2)
            for (std::size_t i = 0; i < kernel.getSize(level); ++i)
            {
                for (std::size_t j = 0; j < image.getSize(level); ++j)
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
                        convolution_detail(execution_policy, image, kernel, output, level - 1, output_index, index2, index3, f);
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
        impl::convolution_detail(execution_policy, image, kernel, output, image.getSize().size() - 1);
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
                output.at(x, y) = normalization_factor * sum;
            }
        }
        return output;
    }

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

    //  rgb2hsv function implementation
    constexpr static auto rgb2hsv(RGB input)
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
    constexpr static auto rgb2hsv(RGB_DOUBLE input)
    {
        RGB rgb{static_cast<std::uint8_t>(input.channels[0]),
                static_cast<std::uint8_t>(input.channels[1]),
                static_cast<std::uint8_t>(input.channels[2])};
        return rgb2hsv(rgb);
    }

    //  hsv2rgb function implementation
    constexpr static auto hsv2rgb(HSV input)
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

    //  constructRGB template function implementation
    template<typename OutputT = RGB>
    constexpr static auto constructRGB(Image<GrayScale> r, Image<GrayScale> g, Image<GrayScale> b)
    {
        check_size_same(r, g);
        check_size_same(g, b);
        auto image_data_r = r.getImageData();
        auto image_data_g = g.getImageData();
        auto image_data_b = b.getImageData();
        std::vector<OutputT> new_data;
        new_data.reserve(r.count());
        for (std::size_t index = 0; index < r.count(); ++index)
        {
            OutputT rgb {   image_data_r[index],
                        image_data_g[index],
                        image_data_b[index]};
            new_data.emplace_back(rgb);
        }
        Image<OutputT> output(new_data, r.getSize());
        return output;
    }

    //  constructRGBDOUBLE template function implementation
    template<typename OutputT = RGB_DOUBLE>
    constexpr static auto constructRGBDOUBLE(Image<double> r, Image<double> g, Image<double> b)
    {
        check_size_same(r, g);
        check_size_same(g, b);
        auto image_data_r = r.getImageData();
        auto image_data_g = g.getImageData();
        auto image_data_b = b.getImageData();
        std::vector<OutputT> new_data;
        for (std::size_t index = 0; index < r.count(); ++index)
        {
            OutputT rgb_double { image_data_r[index],
                                    image_data_g[index],
                                    image_data_b[index]};
            new_data.emplace_back(rgb_double);
        }
        Image<OutputT> output(new_data, r.getSize());
        return output;
    }

    //  constructHSV template function implementation
    template<typename OutputT = HSV>
    constexpr static auto constructHSV(Image<double> h, Image<double> s, Image<double> v)
    {
        check_size_same(h, s);
        check_size_same(s, v);
        auto image_data_h = h.getImageData();
        auto image_data_s = s.getImageData();
        auto image_data_v = v.getImageData();
        std::vector<OutputT> new_data;
        for (size_t index = 0; index < h.count(); ++index)
        {
            OutputT hsv {   image_data_h[index],
                        image_data_s[index],
                        image_data_v[index]};
            new_data.emplace_back(hsv);
        }
        Image<OutputT> output(new_data, h.getSize());
        return output;
    }

    //  convert_image template function implementation
    //  Reference: https://codereview.stackexchange.com/a/292847/231235
    template<typename DstT, typename SrcT>
    constexpr static auto convert_image(Image<SrcT> input)
    {
        auto image_data = input.getImageData();
        std::vector<DstT> new_data;
        for (size_t index = 0; index < input.count(); ++index)
        {
            DstT rgb_double { static_cast<double>(image_data[index].channels[0]),
                                    static_cast<double>(image_data[index].channels[1]),
                                    static_cast<double>(image_data[index].channels[2])};
            new_data.emplace_back(rgb_double);
        }
        Image<DstT> output(new_data, input.getSize());
        return output;
    }

    //  getPlane template function implementation
    template<class OutputT = unsigned char>
    constexpr static auto getPlane(Image<RGB> input, std::size_t index)
    {
        auto input_data = input.getImageData();
        std::vector<OutputT> output_data;
        output_data.resize(input.count());
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
    constexpr static auto getPlane(Image<T> input, std::size_t index)
    {
        auto input_data = input.getImageData();
        std::vector<OutputT> output_data;
        output_data.resize(input.count());
        for (std::size_t i = 0; i < input.count(); ++i)
        {
            output_data[i] = input_data[i].channels[index];
        }
        auto output = Image<OutputT>(output_data, input.getSize());
        return output;
    }

    //  getRplane function implementation
    constexpr static auto getRplane(Image<RGB> input)
    {
        return getPlane(input, 0);
    }

    //  getRplane function implementation
    constexpr static auto getRplane(Image<RGB_DOUBLE> input)
    {
        return getPlane(input, 0);
    }

    //  getGplane function implementation
    constexpr static auto getGplane(Image<RGB> input)
    {
        return getPlane(input, 1);
    }

    //  getGplane function implementation
    constexpr static auto getGplane(Image<RGB_DOUBLE> input)
    {
        return getPlane(input, 1);
    }

    //  getBplane function implementation
    constexpr static auto getBplane(Image<RGB> input)
    {
        return getPlane(input, 2);
    }

    //  getBplane function implementation
    constexpr static auto getBplane(Image<RGB_DOUBLE> input)
    {
        return getPlane(input, 2);
    }

    template<class T = HSV>
    requires (std::same_as<T, HSV>)
    constexpr static auto getHplane(Image<T> input)
    {
        return getPlane(input, 0);
    }

    template<class T = HSV>
    requires (std::same_as<T, HSV>)
    constexpr static auto getSplane(Image<T> input)
    {
        return getPlane(input, 1);
    }

    template<class T = HSV>
    requires (std::same_as<T, HSV>)
    constexpr static auto getVplane(Image<T> input)
    {
        return getPlane(input, 2);
    }

    //  apply_each template function implementation
    template<class F, class... Args>
    constexpr static auto apply_each(Image<RGB> input, F operation, Args&&... args)
    {
        return constructRGB(operation(getRplane(input), args...), operation(getGplane(input), args...), operation(getBplane(input), args...));
    }

    //  apply_each template function implementation
    template<class F, class... Args>
    constexpr static auto apply_each(Image<RGB_DOUBLE> input, F operation, Args&&... args)
    {
        return constructRGBDOUBLE(operation(getRplane(input), args...), operation(getGplane(input), args...), operation(getBplane(input), args...));
    }

    template<class F, class... Args>
    constexpr static auto apply_each(Image<HSV> input, F operation, Args&&... args)
    {
        return constructHSV(operation(getHplane(input), args...), operation(getSplane(input), args...), operation(getVplane(input), args...));
    }

    //  im2double function implementation
    constexpr static auto im2double(Image<RGB> input)
    {
        return convert_image<RGB_DOUBLE>(input);
    }

    //  im2uint8 function implementation
    constexpr static auto im2uint8(Image<RGB_DOUBLE> input)
    {
        return convert_image<RGB>(input);
    }

    //  print_with_latex function implementation
    static void print_with_latex(Image<RGB> input)
    {
        std::cout << "\\begin{tikzpicture}[x=1cm,y=0.4cm]\n";
        for (size_t y = 0; y < input.getHeight(); ++y)
        {
            for (size_t x = 0; x < input.getWidth(); x++)
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
    constexpr static void print_with_latex_to_file(Image<RGB> input, std::string filename)
    {
        std::ofstream newfile;
        newfile.open(filename);
        newfile << "\\begin{tikzpicture}[x=1cm,y=0.4cm]\n";
        for (size_t y = 0; y < input.getHeight(); y++)
        {
            for (size_t x = 0; x < input.getWidth(); x++)
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

    template<typename ElementT>
    constexpr static auto subimage(const Image<ElementT>& input, const std::size_t width, std::size_t height, std::size_t xcenter, std::size_t ycenter)
    {
        auto output = Image<ElementT>(width, height);
        std::size_t cornerx = xcenter - static_cast<std::size_t>(std::floor(static_cast<double>(width) / 2));
        std::size_t cornery = ycenter - static_cast<std::size_t>(std::floor(static_cast<double>(height) / 2));
        for (std::size_t y = 0; y < output.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < output.getWidth(); ++x)
            {
                output.at(x, y) = input.at(cornerx + x, cornery + y);
            }
        }
        return output;
    }

    //  subimage template function implementation
    template<typename ElementT>
    requires ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>))
    constexpr static auto subimage(const Image<ElementT>& input, std::size_t width, std::size_t height, std::size_t xcenter, std::size_t ycenter)
    {
        return apply_each(input, [width, height, xcenter, ycenter](auto&& planes) { return subimage(planes, width, height, xcenter, ycenter); });
    }

    template<typename ElementT>
    constexpr static auto subimage2(const Image<ElementT>& input, const std::size_t startx, const std::size_t endx, const std::size_t starty, const std::size_t endy)
    {
        assert(startx <= endx);
        assert(starty <= endy);
        auto output = Image<ElementT>(endx - startx + 1, endy - starty + 1);
        for (std::size_t y = 0; y < output.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < output.getWidth(); ++x)
            {
                output.at(x, y) = input.at(startx + x, starty + y);
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
                [&](auto&& element1, auto&&... elements) 
                    {
                        return op(element1, elements...);
                    },
                inputs.getImageData()...);
        auto output = Image<recursive_unwrap_type_t<unwrap_level, decltype(transformed_data)>>(
            transformed_data,
            first_of(inputs...).getSize());
        return output;
    }

    //  pixelwiseOperation template function implementation
    template<std::size_t unwrap_level = 1, class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto pixelwiseOperation(ExPo execution_policy, auto op, const Image<InputT>& input1)
    {
        auto transformed_data = recursive_transform<unwrap_level>(
                                    execution_policy,
                                    [&](auto&& element1) 
                                        {
                                            return op(element1);
                                        },
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
    constexpr static auto rgb2hsv(ExPo execution_policy, const Image<ElementT>& input)
    {
        return pixelwiseOperation(execution_policy, [](ElementT input) { return rgb2hsv(input); }, input);
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
    constexpr static auto hsv2rgb(ExPo execution_policy, const Image<HSV>& input)
    {
        return pixelwiseOperation(execution_policy, [](HSV input) { return hsv2rgb(input); }, input);
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

    template<typename T>
    T normalDistribution1D(const T x, const T standard_deviation)
    {
        return std::exp(-x * x / (2 * standard_deviation * standard_deviation));
    }

    template<typename T>
    T normalDistribution2D(const T xlocation, const T ylocation, const T standard_deviation)
    {
        return std::exp(-(xlocation * xlocation + ylocation * ylocation) / (2 * standard_deviation * standard_deviation)) / (2 * std::numbers::pi * standard_deviation * standard_deviation);
    }

    template<class InputT1, class InputT2>
    constexpr static auto cubicPolate(const InputT1 v0, const InputT1 v1, const InputT1 v2, const InputT1 v3, const InputT2 frac)
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
        auto x1 = cubicPolate( ndata[0], ndata[1], ndata[2], ndata[3], fracx );
        auto x2 = cubicPolate( ndata[4], ndata[5], ndata[6], ndata[7], fracx );
        auto x3 = cubicPolate( ndata[8], ndata[9], ndata[10], ndata[11], fracx );
        auto x4 = cubicPolate( ndata[12], ndata[13], ndata[14], ndata[15], fracx );

        return std::clamp(
                    static_cast<InputT>(cubicPolate(x1, x2, x3, x4, fracy)),
                    static_cast<InputT>(std::numeric_limits<ElementT>::min()),
                    static_cast<InputT>(std::numeric_limits<ElementT>::max()));
    }

    //  copyResizeBicubic template function implementation
    template<class FloatingType = float, arithmetic ElementT>
    Image<ElementT> copyResizeBicubic(Image<ElementT>& image, size_t width, size_t height)
    {
        auto output = Image<ElementT>(width, height);
        //  get used to the C++ way of casting
        auto ratiox = static_cast<FloatingType>(image.getWidth()) / static_cast<FloatingType>(width);
        auto ratioy = static_cast<FloatingType>(image.getHeight()) / static_cast<FloatingType>(height);
        
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                FloatingType xMappingToOrigin = static_cast<FloatingType>(x) * ratiox;
                FloatingType yMappingToOrigin = static_cast<FloatingType>(y) * ratioy;
                FloatingType xMappingToOriginFloor = std::floor(xMappingToOrigin);
                FloatingType yMappingToOriginFloor = std::floor(yMappingToOrigin);
                FloatingType xMappingToOriginFrac = xMappingToOrigin - xMappingToOriginFloor;
                FloatingType yMappingToOriginFrac = yMappingToOrigin - yMappingToOriginFloor;
                
                ElementT ndata[4 * 4];
                for (int ndatay = -1; ndatay <= 2; ++ndatay)
                {
                    for (int ndatax = -1; ndatax <= 2; ++ndatax)
                    {
                        ndata[(ndatay + 1) * 4 + (ndatax + 1)] = image.at(
                            std::clamp(xMappingToOriginFloor + ndatax, static_cast<FloatingType>(0), static_cast<FloatingType>(image.getWidth()) - static_cast<FloatingType>(1)), 
                            std::clamp(yMappingToOriginFloor + ndatay, static_cast<FloatingType>(0), static_cast<FloatingType>(image.getHeight()) - static_cast<FloatingType>(1)));
                    }
                    
                }
                output.at(x, y) = bicubicPolate(ndata, xMappingToOriginFrac, yMappingToOriginFrac);
            }
        }
        return output;
    }

    //  copyResizeBicubic template function implementation for color image
    template<class FloatingType = double, class ElementT>
    requires (std::same_as<ElementT, RGB>)
    Image<ElementT> copyResizeBicubic(Image<ElementT>& image, size_t width, size_t height)
    {
        return TinyDIP::apply_each(image, [&](auto&& each_plane)
        {
            return TinyDIP::copyResizeBicubic<FloatingType>(each_plane, width, height);
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
            row_vector.at(x, 0) = normalDistribution1D(static_cast<InputT>(x) - static_cast<InputT>(center), standard_deviation);
        }
        return row_vector;
    }

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
                output.at(x, y) = row_vector_x.at(x, 0) * row_vector_y.at(y, 0);
            }
        }
        return output;
    }

    //  General two-dimensional elliptical Gaussian
    //  f(x, y) = A*e^(-a(x - x0)^2 + 2b(x - x0)(y - y0)+c(y - y0)^2)
    template<class InputT>
    constexpr static auto gaussianFigure2D(
        const size_t xsize, const size_t ysize,
        const size_t centerx, const size_t centery,
        const InputT a, const InputT b,
        const InputT c, const InputT normalize_factor = 1.0)
    {
        auto output = Image<InputT>(xsize, ysize);
        for (size_t y = 0; y < ysize; ++y)
        {
            for (size_t x = 0; x < xsize; ++x)
            {
                output.at(x, y) = normalize_factor*
                    std::exp(
                        -a*std::pow((static_cast<InputT>(x) - static_cast<InputT>(centerx)), 2) +
                        2*b*(static_cast<InputT>(x) - static_cast<InputT>(centerx))*(static_cast<InputT>(y) - static_cast<InputT>(centery)) +
                        c*std::pow((static_cast<InputT>(y) - static_cast<InputT>(centery)), 2)
                        );
            }
        }
        return output;
    }

    //  single standard deviation
    template<class InputT>
    constexpr static Image<InputT> gaussianFigure2D(
        const size_t xsize, const size_t ysize,
        const size_t centerx, const size_t centery,
        const InputT standard_deviation)
    {
        return gaussianFigure2D(xsize, ysize, centerx, centery, standard_deviation, standard_deviation);
    }

    //  gaussianFigure3D Template Function Implementation
    //  multiple standard deviations
    template<class InputT>
    requires(std::floating_point<InputT> || std::integral<InputT>)
    constexpr static auto gaussianFigure3D(
        const size_t xsize, const size_t ysize, const size_t zsize,
        const size_t centerx, const size_t centery, const size_t centerz,
        const InputT standard_deviation_x, const InputT standard_deviation_y, const InputT standard_deviation_z)
    {
        auto output = std::vector<Image<InputT>>();
        output.reserve(zsize);
        auto gaussian_image2d = gaussianFigure2D(xsize, ysize, centerx, centery, standard_deviation_x, standard_deviation_y);
        for (size_t z = 0; z < zsize; ++z)
        {
            output.emplace_back(
                gaussian_image2d *
                normalDistribution1D(static_cast<InputT>(z) - static_cast<InputT>(centerz), standard_deviation_z)
            );
        }
        return VolumetricImage(output);
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

    template<class InputT>
    constexpr static Image<InputT> subtract(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        check_size_same(input1, input2);
        return pixelwiseOperation(std::minus<>{}, input1, input2);
    }

    template<class InputT>
    constexpr static auto subtract(const std::vector<Image<InputT>>& input1, const std::vector<Image<InputT>>& input2)
    {
        assert(input1.size() == input2.size());
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
        return pixelwiseOperation(std::multiplies<>{}, input1, input2);
    }

    //  multiplies Template Function Implementation
    template<class InputT, class TimesT>
    requires((std::floating_point<TimesT> || std::integral<TimesT> || is_complex<TimesT>::value) &&
             ((!std::same_as<InputT, RGB>) && (!std::same_as<InputT, RGB_DOUBLE>) && (!std::same_as<InputT, HSV>)))
    constexpr static Image<InputT> multiplies(const Image<InputT>& input1, const TimesT times)
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
    constexpr static Image<InputT> multiplies(const Image<InputT>& input1, const TimesT times)
    {
        return apply_each(input1, [&](auto&& each_plane)
        {
            return multiplies(each_plane, times);
        });
    }

    
    template<class InputT, class TimesT>
    requires(std::floating_point<TimesT> || std::integral<TimesT>)
    constexpr static Image<InputT> multiplies(const TimesT times, const Image<InputT>& input1)
    {
        return multiplies(input1, times);
    }

    template<class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static Image<InputT> pixelwise_multiplies(ExPo execution_policy, const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return pixelwiseOperation(execution_policy, std::multiplies<>{}, input1, input2);
    }

    template<class InputT, class... Args>
    constexpr static Image<InputT> pixelwise_multiplies(const Image<InputT>& input1, const Args&... inputs)
    {
        return pixelwiseOperation(std::multiplies<>{}, input1, pixelwise_multiplies(inputs...));
    }

    template<class ExPo, class InputT, class... Args>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static Image<InputT> pixelwise_multiplies(ExPo execution_policy, const Image<InputT>& input1, const Args&... inputs)
    {
        return pixelwiseOperation(execution_policy, std::multiplies<>{}, input1, pixelwise_multiplies(inputs...));
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

    template<class InputT>
    constexpr static Image<InputT> divides(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return pixelwiseOperation(std::divides<>{}, input1, input2);
    }

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

    template<class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static Image<InputT> divides(ExPo execution_policy, const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return pixelwiseOperation(execution_policy, std::divides<>{}, input1, input2);
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

    template<std::floating_point ElementT = double, std::floating_point OutputT = ElementT>
    Image<OutputT> dct3_one_plane(const std::vector<Image<ElementT>>& input, const std::size_t plane_index)
    {
        auto N1 = static_cast<OutputT>(input[0].getWidth());
        auto N2 = static_cast<OutputT>(input[0].getHeight());
        auto N3 = input.size();
        auto alpha1 = (plane_index == 0) ? (std::numbers::sqrt2_v<OutputT> / 2) : (OutputT{1.0});
        auto output = Image<OutputT>(input[plane_index].getWidth(), input[plane_index].getHeight());
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

    template<std::floating_point ElementT = double, std::floating_point OutputT = ElementT>
    Image<OutputT> idct3_one_plane(const std::vector<Image<ElementT>>& input, const std::size_t plane_index)
    {
        auto N1 = static_cast<OutputT>(input[0].getWidth());
        auto N2 = static_cast<OutputT>(input[0].getHeight());
        auto N3 = input.size();
        auto output = Image<OutputT>(input[plane_index].getWidth(), input[plane_index].getHeight());
        for (std::size_t y = 0; y < output.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < output.getWidth(); ++x)
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

    template<arithmetic ElementT = double>
    constexpr static auto abs(const Image<ElementT>& input)
    {
        return pixelwiseOperation([](auto&& element) { return std::abs(element); }, input);
    }

    template<class ExPo, arithmetic ElementT = double>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto abs(ExPo execution_policy, const Image<ElementT>& input)
    {
        return pixelwiseOperation(execution_policy, [](auto&& element) { return std::abs(element); }, input);
    }

    //  abs template function implementation
    template<class InputT>
    requires((std::same_as<InputT, RGB>) || (std::same_as<InputT, RGB_DOUBLE>) || (std::same_as<InputT, HSV>))
    constexpr static auto abs(const Image<InputT>& input)
    {
        return apply_each(input, [&](auto&& planes) { return abs(planes); });
    }

    template<arithmetic ElementT = double>
    constexpr static auto difference(const Image<ElementT>& input1, const Image<ElementT>& input2)
    {
        return pixelwiseOperation([](auto&& element1, auto&& element2) { return std::abs(element1 - element2); }, input1, input2);
    }

    //  difference Template Function Implementation
    template<class InputT>
    requires((std::same_as<InputT, RGB>) || (std::same_as<InputT, RGB_DOUBLE>) || (std::same_as<InputT, HSV>))
    constexpr static auto difference(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return pixelwiseOperation(
            [](InputT element1, InputT element2)
            {
                InputT output;
                for(std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                {
                    output.channels[channel_index] = 
                        std::abs(element1.channels[channel_index] - element2.channels[channel_index]);
                }
                return output;
            }, input1, input2);
    }

    //  manhattan_distance Template Function Implementation
    template<arithmetic ElementT = double>
    constexpr static ElementT manhattan_distance(const Image<ElementT>& input1, const Image<ElementT>& input2)
    {
        if(input1.getSize() != input2.getSize())
        {
            throw std::runtime_error("Size mismatched!");
        }
        return recursive_reduce(difference(input1, input2).getImageData(), ElementT{});
    }

    template<arithmetic ElementT = double, arithmetic ExpT = double>
    constexpr static auto pow(const Image<ElementT>& input, ExpT exp)
    {
        return pixelwiseOperation([&](auto&& element) { return std::pow(element, exp); }, input);
    }

    //  pow template function implementation
    template<class ExPo, arithmetic ElementT = double, arithmetic ExpT = double>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto pow(ExPo execution_policy, const Image<ElementT>& input, ExpT exp)
    {
        return pixelwiseOperation(execution_policy, [&](auto&& element) { return std::pow(element, exp); }, input);
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
                    output.at(x_location + x, y_location + y) = target.at_without_boundary_check(x, y);
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
                    output.at(x, y) = background.at_without_boundary_check(x, y);
                }
            }
            for (std::size_t y = 0; y < target.getHeight(); ++y)
            {
                for (std::size_t x = 0; x < target.getWidth(); ++x)
                {
                    output.at(x_location + x, y_location + y) = target.at_without_boundary_check(x, y);
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
    requires ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, HSV>))       //  TODO: Create a base class for both RGB and HSV
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
    requires ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, HSV>))
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
        Image<ElementT> output = input;
        for(std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for(std::size_t x = 0; x < input.getWidth(); ++x)
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
        Image<ElementT> output = input;
        for(std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for(std::size_t x = 0; x < input.getWidth(); ++x)
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
        Image<ElementT> output = input;
        for(std::size_t y = 0; y < input.getHeight(); ++y)
        {
            for(std::size_t x = 0; x < input.getWidth(); ++x)
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
        const Image<ElementT> input,
        std::size_t width_expansion,
        std::size_t height_expansion,
        ElementT default_value = ElementT{})
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
        temp.setAllValue(input.at(0, 0));
        output = paste2D(
            execution_policy,
            output,
            temp,
            0,
            0,
            default_value);
        //  Right-top corner
        temp.setAllValue(input.at(input.getWidth() - 1, 0));
        output = paste2D(
            execution_policy,
            output,
            temp,
            width_expansion + input.getWidth(),
            0,
            default_value);
        //  Left-bottom corner
        temp.setAllValue(input.at(0, input.getHeight() - 1));
        output = paste2D(
            execution_policy,
            output,
            temp,
            0,
            height_expansion + input.getHeight(),
            default_value);
        //  Right-bottom corner
        temp.setAllValue(input.at(input.getWidth() - 1, input.getHeight() - 1));
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
        SigmaT sigma1,
        SigmaT sigma2,
        SizeT filter_size1,
        SizeT filter_size2,
        BoundaryCondition boundaryCondition = BoundaryCondition::mirror,
        ElementT value_for_constant_padding = ElementT{})
    {
        if (input.getDimensionality()!=2)
        {
            throw std::runtime_error("Unsupported dimension!");
        }
        Image<ElementT> padded_image;
        switch(boundaryCondition)
        {
            case constant:
                padded_image = generate_constant_padding_image(execution_policy, input, filter_size1, filter_size2, value_for_constant_padding);
                break;
            case mirror:
                padded_image = generate_mirror_padding_image(execution_policy, input, filter_size1, filter_size2, value_for_constant_padding);
                break;
            case replicate:
                padded_image = generate_replicate_padding_image(execution_policy, input, filter_size1, filter_size2, value_for_constant_padding);
                break;
        }
        
        auto filter_mask_x = gaussianFigure1D(
                                    filter_size1,
                                    (static_cast<double>(filter_size1) + 1.0) / 2.0,
                                    sigma1);
        auto sum_result = sum(filter_mask_x);
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
        SigmaT sigma1,
        SigmaT sigma2)
    {
        return subtract(
            imgaussfilt(input, sigma1, static_cast<int>(computeFilterSizeFromSigma(sigma1))),
            imgaussfilt(input, sigma2, static_cast<int>(computeFilterSizeFromSigma(sigma2)))
            );
    }

    namespace SIFT_impl {
        /*  is_it_extremum template function implementation
            input1, input2 and input3 are 3 * 3 images. If the center pixel (at location (1,1) ) of input2 is the largest / smallest one, 
            return true; otherwise, return false.
            Test: https://godbolt.org/z/Kb34EW5Yj
        */
        template<typename ElementT>
        constexpr static bool is_it_extremum(Image<ElementT> input1, Image<ElementT> input2, Image<ElementT> input3, double threshold = -1.0)
        {
            auto center_pixel = input2.at(1, 1);
            auto input2_img_data = input2.getImageData();
            input2_img_data.erase(input2_img_data.begin() + 4);                         //  https://stackoverflow.com/a/875117/6667035
            if (std::abs(center_pixel) > threshold)
            {
                if (std::ranges::all_of(input1.getImageData(), [&](ElementT i) { return center_pixel > i; }) &&
                    std::ranges::all_of(input3.getImageData(), [&](ElementT i) { return center_pixel > i; }) &&
                    std::ranges::all_of(input2_img_data, [&](ElementT i) { return center_pixel > i; }))
                {
                    return true;
                }
                if (std::ranges::all_of(input1.getImageData(), [&](ElementT i) { return center_pixel < i; }) &&
                    std::ranges::all_of(input3.getImageData(), [&](ElementT i) { return center_pixel < i; }) &&
                    std::ranges::all_of(input2_img_data, [&](ElementT i) { return center_pixel < i; }))
                {
                    return true;
                }
            }
            return false;
        }

        //  keypoint_refinement template function implementation
        //  refine the given keypoint's location using Taylor expansion
        template<typename ElementT = double>
        constexpr static auto keypoint_refinement(Image<ElementT> input, std::size_t x, std::size_t y)
        {
            //  Calculate the gradient at the keypoint (x, y)
            ElementT first_derivative_x = (input.at(2, 1) - input.at(0, 1)) / 2.0;
            ElementT first_derivative_y = (input.at(1, 2) - input.at(1, 0)) / 2.0;
            ElementT second_derivative_x = (input.at(2, 1) + input.at(0, 1) - 2.0 * input.at(1, 1));
            ElementT second_derivative_y = (input.at(1, 2) + input.at(1, 0) - 2.0 * input.at(1, 1));
            ElementT second_derivative_xy = (input.at(2, 2) - input.at(2, 0) - input.at(0, 2) + input.at(0, 0)) / 4.0;
            ElementT A = -second_derivative_x / (second_derivative_x * second_derivative_y - second_derivative_xy * second_derivative_xy);
            ElementT B = second_derivative_xy / (second_derivative_x * second_derivative_y - second_derivative_xy * second_derivative_xy);
            ElementT C = B;
            ElementT D = -second_derivative_y / (second_derivative_x * second_derivative_y - second_derivative_xy * second_derivative_xy);
            ElementT offset_x = A * first_derivative_x + B * first_derivative_y;
            ElementT offset_y = C * first_derivative_x + D * first_derivative_y;
            return std::make_tuple(
                static_cast<ElementT>(x) + offset_x,
                static_cast<ElementT>(y) + offset_y);
        }

        //  keypoint_filtering template function implementation
        template<typename ElementT = double>
        constexpr static bool keypoint_filtering(
            Image<ElementT>& input,
            ElementT contrast_check_threshold = 0.03,
            ElementT edge_response_threshold = 12.1)
        {
            //  Calculate Hessian matrix at the keypoint (x, y)
            ElementT second_derivative_x = (input.at(2, 1) + input.at(0, 1) - 2.0 * input.at(1, 1));                    //  D_{xx}
            ElementT second_derivative_y = (input.at(1, 2) + input.at(1, 0) - 2.0 * input.at(1, 1));                    //  D_{yy}
            ElementT second_derivative_xy = (input.at(2, 2) - input.at(2, 0) - input.at(0, 2) + input.at(0, 0)) / 4.0;  //  D_{xy}
            ElementT trace_Hessian_matrix = second_derivative_x + second_derivative_y;
            ElementT determinant_Hessian_matrix = second_derivative_x * second_derivative_y - second_derivative_xy * second_derivative_xy;
            ElementT principal_curvature_ratio = trace_Hessian_matrix * trace_Hessian_matrix / determinant_Hessian_matrix;
            if (input.at(1, 1) <= contrast_check_threshold || principal_curvature_ratio >= edge_response_threshold)
            {
                return false;
            }
            return true;
        }

        //  mapping_point function implementation
        std::tuple<std::size_t, std::size_t> mapping_point(
            std::tuple<std::size_t, std::size_t> input_location,
            std::size_t input_width,
            std::size_t input_height,
            std::size_t target_width,
            std::size_t target_height)
        {
            double width_percentage = static_cast<double>(std::get<0>(input_location)) / static_cast<double>(input_width);
            double height_percentage = static_cast<double>(std::get<1>(input_location)) / static_cast<double>(input_height);
            return std::make_tuple(
                static_cast<std::size_t>(width_percentage * static_cast<double>(target_width)),
                static_cast<std::size_t>(height_percentage * static_cast<double>(target_height))
            );
        }

        //  find_local_extrema template function implementation
        template<typename ElementT>
        constexpr static auto find_local_extrema(
            Image<ElementT> input1,
            Image<ElementT> input2,
            Image<ElementT> input3,
            std::size_t origin_width,
            std::size_t origin_height)
        {
            if (input1.getSize() != input2.getSize())
            {
                throw std::runtime_error("Size mismatched!");
            }
            if (input2.getSize() != input3.getSize())
            {
                throw std::runtime_error("Size mismatched!");
            }
            const int block_size = 3;
            std::vector<std::tuple<std::size_t, std::size_t>> output;
            #pragma omp parallel for collapse(2)
            for (int y = 1; y < input1.getHeight() - 1; ++y)
            {
                for (int x = 1; x < input1.getWidth() - 1; ++x)
                {
                    auto subimage1 = subimage(input1, block_size, block_size, x, y);
                    auto subimage2 = subimage(input2, block_size, block_size, x, y);
                    auto subimage3 = subimage(input3, block_size, block_size, x, y);
                    if (is_it_extremum(subimage1, subimage2, subimage3))
                    {
                        output.emplace_back(
                            mapping_point(
                                std::make_tuple(x, y),
                                input1.getWidth(),
                                input1.getHeight(),
                                origin_width,
                                origin_height));
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
        requires(std::floating_point<SigmaT> || std::integral<SigmaT>)
        constexpr static auto get_potential_keypoint(
            const Image<ElementT>& input,
            std::size_t octaves_count = 4,
            std::size_t number_of_scale_levels = 5,
            SigmaT initial_sigma = 1.6,
            double k = 1.4142135623730950488016887242097)
        {
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
                    keypoints.append_range(
                        find_local_extrema(
                            each_octave[scale_index],
                            each_octave[scale_index + 1],
                            each_octave[scale_index + 2],
                            octave_index,
                            scale_index)
                    );
                }
            }

            //  
            std::vector<std::tuple<std::size_t, std::size_t>> mapped_keypoints;
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
    }

    //  imbilatfilt template function implementation
    template<typename ElementT, typename SigmaT = double>
    requires(std::floating_point<SigmaT> || std::integral<SigmaT>)
    constexpr static auto imbilatfilt(const Image<ElementT>& input)
    {

    }

}

#endif