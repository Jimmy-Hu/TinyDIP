/* Developed by Jimmy Hu */

#ifndef ImageOperations_H
#define ImageOperations_H

#include <execution>
#include <fstream>
#include <numbers>
#include <string>
#include "base_types.h"
#include "image.h"

#define is_size_same(x, y) {assert(x.getWidth() == y.getWidth()); assert(x.getHeight() == y.getHeight());}

namespace TinyDIP
{
    static auto rgb2hsv(RGB input)
    {
        HSV output{};
        BYTE Red = input.channels[0], Green = input.channels[1], Blue = input.channels[2];
        std::vector<BYTE> v{ Red, Green, Blue };
        std::ranges::sort(v);
        BYTE Max = v[2], Mid = v[1], Min = v[0];

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

    static auto hsv2rgb(HSV input)
    {
        RGB output{};
        long double H = input.channels[0], S = input.channels[1], Max = input.channels[2];
        BYTE hi = static_cast<BYTE>(floor(H / 60.0));
        long double f = (H / 60.0) - hi;
        long double Min, q, t;
        Min = Max * (1.0 - S);
        q = Max * (1.0 - f * S);
        t = Max * (1.0 - (1.0 - f) * S);
        if (hi == 0)
        {
            output.channels[0] = static_cast<BYTE>(Max);
            output.channels[1] = static_cast<BYTE>(t);
            output.channels[2] = static_cast<BYTE>(Min);
        }
        else if (hi == 1)
        {
            output.channels[0] = static_cast<BYTE>(q);
            output.channels[1] = static_cast<BYTE>(Max);
            output.channels[2] = static_cast<BYTE>(Min);
        }
        else if (hi == 2)
        {
            output.channels[0] = static_cast<BYTE>(Min);
            output.channels[1] = static_cast<BYTE>(Max);
            output.channels[2] = static_cast<BYTE>(t);
        }
        else if (hi == 3)
        {
            output.channels[0] = static_cast<BYTE>(Min);
            output.channels[1] = static_cast<BYTE>(q);
            output.channels[2] = static_cast<BYTE>(Max);
        }
        else if (hi == 4)
        {
            output.channels[0] = static_cast<BYTE>(t);
            output.channels[1] = static_cast<BYTE>(Min);
            output.channels[2] = static_cast<BYTE>(Max);
        }
        else if (hi == 5)
        {
            output.channels[0] = static_cast<BYTE>(Max);
            output.channels[1] = static_cast<BYTE>(Min);
            output.channels[2] = static_cast<BYTE>(q);
        }
        return output;
    }

    // Forward Declaration class Image
    template <typename ElementT>
    class Image;

    template<class T = GrayScale>
    requires (std::same_as<T, GrayScale>)
    constexpr static auto constructRGB(Image<T> r, Image<T> g, Image<T> b)
    {
        is_size_same(r, g);
        is_size_same(g, b);
        is_size_same(r, b);
        Image<RGB> output(r.getWidth(), r.getHeight());
        for (std::size_t y = 0; y < r.getHeight(); y++)
        {
            for (std::size_t x = 0; x < r.getWidth(); x++)
            {
                output.at(x, y).channels[0] = r.at(x, y);
                output.at(x, y).channels[1] = g.at(x, y);
                output.at(x, y).channels[2] = b.at(x, y);
            }
        }
        return output;
    }

    template<class T = double>
    requires (std::same_as<T, double>)
    constexpr static auto constructHSV(Image<T> h, Image<T> s, Image<T> v)
    {
        is_size_same(h, s);
        is_size_same(s, v);
        is_size_same(h, v);
        Image<HSV> output(h.getWidth(), h.getHeight());
        for (std::size_t y = 0; y < h.getHeight(); y++)
        {
            for (std::size_t x = 0; x < h.getWidth(); x++)
            {
                output.at(x, y).channels[0] = h.at(x, y);
                output.at(x, y).channels[1] = s.at(x, y);
                output.at(x, y).channels[2] = v.at(x, y);
            }
        }
        return output;
    }

    template<class ElementT>
    requires ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, HSV>))
    constexpr static auto getPlane(Image<ElementT> input, std::size_t index)
    {
        auto output = TinyDIP::Image(input.getWidth(), input.getHeight(), input.at(0, 0).channels[0]);
        for (std::size_t y = 0; y < input.getHeight(); y++)
        {
            for (std::size_t x = 0; x < input.getWidth(); x++)
            {
                output.at(x, y) = input.at(x, y).channels[index];
            }
        }
        return output;
    }

    template<class T = RGB>
    requires (std::same_as<T, RGB>)
    constexpr static auto getRplane(Image<T> input)
    {
        return getPlane(input, 0);
    }

    template<class T = RGB>
    requires (std::same_as<T, RGB>)
    constexpr static auto getGplane(Image<T> input)
    {
        return getPlane(input, 1);
    }

    template<class T = RGB>
    requires (std::same_as<T, RGB>)
    constexpr static auto getBplane(Image<T> input)
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

    template<class T = RGB, class F, class... Args>
    requires (std::same_as<T, RGB>)
    constexpr static auto apply_each(Image<T> input, F operation, Args&&... args)
    {
        return constructRGB(operation(getRplane(input), args...), operation(getGplane(input), args...), operation(getBplane(input), args...));
    }

    template<class T = HSV, class F, class... Args>
    requires (std::same_as<T, HSV>)
    constexpr static auto apply_each(Image<T> input, F operation, Args&&... args)
    {
        return constructHSV(operation(getHplane(input), args...), operation(getSplane(input), args...), operation(getVplane(input), args...));
    }

    template<class T = RGB>
    requires (std::same_as<T, RGB>)
    constexpr static void print_with_latex(Image<T> input)
    {
        std::cout << "\\begin{tikzpicture}[x=1cm,y=0.4cm]\n";
        for (size_t y = 0; y < input.getHeight(); y++)
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

    template<class T = RGB>
    requires (std::same_as<T, RGB>)
    constexpr static void print_with_latex_to_file(Image<T> input, std::string filename)
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
    constexpr static auto subimage(const Image<ElementT>& input, std::size_t width, std::size_t height, std::size_t xcenter, std::size_t ycenter)
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

    template<typename ElementT>
    requires ((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, HSV>))
    constexpr static auto subimage(const Image<ElementT>& input, std::size_t width, std::size_t height, std::size_t xcenter, std::size_t ycenter)
    {
        return apply_each(input, [width, height, xcenter, ycenter](auto&& planes) { return subimage(planes, width, height, xcenter, ycenter); });
    }

    template<typename ElementT>
    constexpr static auto subimage2(const Image<ElementT>& input, std::size_t startx, std::size_t endx, std::size_t starty, std::size_t endy)
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

    template<typename ElementT>
    requires (std::same_as<ElementT, RGB>)
    constexpr static auto rgb2hsv(const Image<ElementT>& input)
    {
        auto output = Image<HSV>(
            recursive_transform<1>([](RGB input) { return rgb2hsv(input); }, input.getImageData()),
            input.getWidth(),
            input.getHeight());
        return output;
    }

    template<typename ElementT>
    requires (std::same_as<ElementT, HSV>)
    constexpr static auto hsv2rgb(const Image<ElementT>& input)
    {
        auto output = Image<RGB>(
            recursive_transform<1>([](HSV input) { return hsv2rgb(input); }, input.getImageData()),
            input.getWidth(),
            input.getHeight());
        return output;
    }

    template<typename ElementT>
    constexpr static auto concat_horizontal(Image<ElementT> input1, Image<ElementT> input2)
    {
        assert(input1.getHeight() == input2.getHeight());
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
    constexpr static auto concat_horizontal(std::vector<Image<ElementT>> input)
    {
        //return TinyDIP::recursive_reduce(input, TinyDIP::Image<ElementT>(0, input[0].getHeight()), [](Image<ElementT> element1, Image<ElementT> element2) { return TinyDIP::concat_horizontal(element1, element2); });
        auto output = input[0];
        for (std::size_t i = 1; i < input.size(); i++)
        {
            output = concat_horizontal(output, input[i]);
        }
        return output;
    }

    template<typename ElementT>
    constexpr static auto concat_vertical(Image<ElementT> input1, Image<ElementT> input2)
    {
        assert(input1.getWidth() == input2.getWidth());
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
    constexpr static auto concat_vertical(std::vector<Image<ElementT>> input)
    {
        auto output = input[0];
        for (std::size_t i = 1; i < input.size(); i++)
        {
            output = concat_vertical(output, input[i]);
        }
        return output;
    }

    template<typename ElementT>
    constexpr static auto concat(std::vector<std::vector<Image<ElementT>>> input)
    {
        auto result1 = TinyDIP::recursive_transform<1>(
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
                    cubicPolate( x1, x2, x3, x4, fracy ), 
                    static_cast<InputT>(std::numeric_limits<ElementT>::min()),
                    static_cast<InputT>(std::numeric_limits<ElementT>::max()));
    }

    template<class FloatingType = float, class ElementT>
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
                            std::clamp(xMappingToOriginFloor + ndatax, static_cast<FloatingType>(0), image.getWidth() - static_cast<FloatingType>(1)), 
                            std::clamp(yMappingToOriginFloor + ndatay, static_cast<FloatingType>(0), image.getHeight() - static_cast<FloatingType>(1)));
                    }
                    
                }
                output.at(x, y) = bicubicPolate(ndata, xMappingToOriginFrac, yMappingToOriginFrac);
            }
        }
        return output;
    }

    //  multiple standard deviations
    template<class InputT>
    constexpr static Image<InputT> gaussianFigure2D(
        const size_t xsize, const size_t ysize, 
        const size_t centerx, const size_t centery,
        const InputT standard_deviation_x, const InputT standard_deviation_y)
    {
        auto output = TinyDIP::Image<InputT>(xsize, ysize);
        auto row_vector_x = TinyDIP::Image<InputT>(xsize, 1);
        for (size_t x = 0; x < xsize; ++x)
        {
            row_vector_x.at(x, 0) = normalDistribution1D(static_cast<InputT>(x) - static_cast<InputT>(centerx), standard_deviation_x);
        }

        auto row_vector_y = TinyDIP::Image<InputT>(ysize, 1);
        for (size_t y = 0; y < ysize; ++y)
        {
            row_vector_y.at(y, 0) = normalDistribution1D(static_cast<InputT>(y) - static_cast<InputT>(centery), standard_deviation_y);
        }
        
        for (size_t y = 0; y < ysize; ++y)
        {
            for (size_t x = 0; x < xsize; ++x)
            {
                output.at(x, y) = row_vector_x.at(x, 0) * row_vector_y.at(y, 0);
            }
        }
        return output;
    }

    //  multiple standard deviations with correlation
    //  0 <= correlation <= 1
    template<class InputT>
    constexpr static Image<double> gaussianFigure2D(
        const size_t xsize, const size_t ysize,
        const size_t centerx, const size_t centery,
        const double standard_deviation_x, const double standard_deviation_y,
        const double correlation, const double normalize_factor = 1.0)
    {

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

    template<typename Op, class InputT, class... Args>
    constexpr static Image<InputT> pixelwiseOperation(Op op, const Image<InputT>& input1, const Args&... inputs)
    {
        Image<InputT> output(
            recursive_transform<1>(
                [&](auto&& element1, auto&&... elements) 
                    {
                        auto result = op(element1, elements...);
                        return static_cast<InputT>(std::clamp(
                            result,
                            static_cast<decltype(result)>(std::numeric_limits<InputT>::min()),
                            static_cast<decltype(result)>(std::numeric_limits<InputT>::max())));
                    },
                (input1.getImageData()),
                (inputs.getImageData())...),
            input1.getWidth(),
            input1.getHeight());
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
        return TinyDIP::pixelwiseOperation(std::plus<>{}, input1, plus(inputs...));
    }

    template<class InputT>
    constexpr static Image<InputT> subtract(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        is_size_same(input1, input2);
        return TinyDIP::pixelwiseOperation(std::minus<>{}, input1, input2);
    }

    template<class InputT = RGB>
    requires (std::same_as<InputT, RGB>)
    constexpr static Image<InputT> subtract(Image<InputT>& input1, Image<InputT>& input2)
    {
        is_size_same(input1, input2);
        Image<InputT> output(input1.getWidth(), input1.getHeight());
        for (std::size_t y = 0; y < input1.getHeight(); ++y)
        {
            for (std::size_t x = 0; x < input1.getWidth(); ++x)
            {
                for(std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                {
                    output.at(x, y).channels[channel_index] = 
                    std::clamp(
                        input1.at(x, y).channels[channel_index] - 
                        input2.at(x, y).channels[channel_index],
                        0,
                        255);
                }
            }
        }
        return output;
    }

    template<class InputT>
    constexpr static Image<InputT> multiplies(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return TinyDIP::pixelwiseOperation(std::multiplies<>{}, input1, input2);
    }

    template<class InputT, class... Args>
    constexpr static Image<InputT> multiplies(const Image<InputT>& input1, const Args&... inputs)
    {
        return TinyDIP::pixelwiseOperation(std::multiplies<>{}, input1, multiplies(inputs...));
    }

    template<class InputT>
    constexpr static Image<InputT> divides(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return TinyDIP::pixelwiseOperation(std::divides<>{}, input1, input2);
    }
    
    template<class InputT>
    constexpr static Image<InputT> modulus(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return TinyDIP::pixelwiseOperation(std::modulus<>{}, input1, input2);
    }

    template<class InputT>
    constexpr static Image<InputT> negate(const Image<InputT>& input1, const Image<InputT>& input2)
    {
        return TinyDIP::pixelwiseOperation(std::negate<>{}, input1);
    }

    template<arithmetic ElementT = double, arithmetic OutputT = ElementT>
    constexpr static Image<ElementT> dct3_detail(std::vector<Image<ElementT>> input, int plane_index)
    {
        std::size_t N3 = input.size();
        OutputT alpha1 = (plane_index == 0) ? (static_cast<OutputT>(1.0) / static_cast<OutputT>(std::sqrt(2))) : (static_cast<OutputT>(1.0));
        auto output = TinyDIP::Image<OutputT>(input[plane_index].getWidth(), input[plane_index].getHeight());
        for (std::size_t y = 0; y < output.getHeight(); y++)
        {
            OutputT alpha2 = (y == 0) ? (static_cast<OutputT>(1.0) / static_cast<OutputT>(std::sqrt(2))) : (static_cast<OutputT>(1.0));
            for (std::size_t x = 0; x < output.getWidth(); x++)
            {
                OutputT sum{};
                OutputT alpha3 = (x == 0) ? (static_cast<OutputT>(1.0) / static_cast<OutputT>(std::sqrt(2))) : (static_cast<OutputT>(1.0));
                for (std::size_t inner_z = 0; inner_z < N3; inner_z++)
                {
                    auto plane = input[inner_z];
                    auto N1 = static_cast<OutputT>(plane.getWidth());
                    auto N2 = static_cast<OutputT>(plane.getHeight());
                    for (std::size_t inner_y = 0; inner_y < plane.getHeight(); inner_y++)
                    {
                        for (std::size_t inner_x = 0; inner_x < plane.getWidth(); inner_x++)
                        {
                            auto l1 = (std::numbers::pi / (2 * N1) * (2 * static_cast<OutputT>(inner_x) + 1) * x);
                            auto l2 = (std::numbers::pi / (2 * N2) * (2 * static_cast<OutputT>(inner_y) + 1) * y);
                            auto l3 = (std::numbers::pi / (2 * static_cast<OutputT>(N3)) * (2 * static_cast<OutputT>(inner_z) + 1) * static_cast<OutputT>(plane_index));
                            sum += static_cast<OutputT>(plane.at(inner_x, inner_y)) *
                                std::cos(l1) * std::cos(l2) * std::cos(l3);
                        }
                    }
                }
                auto N1 = static_cast<OutputT>(input[0].getWidth());
                auto N2 = static_cast<OutputT>(input[0].getHeight());
                output.at(x, y) = 8 * alpha1 * alpha2 * alpha3 * sum / (N1 * N2 * N3);
            }
        }
        return output;
    }

    template<arithmetic ElementT = double, arithmetic OutputT = ElementT>
    constexpr static std::vector<Image<ElementT>> dct3(std::vector<Image<ElementT>> input)
    {
        std::vector<Image<ElementT>> output;
        for (std::size_t i = 0; i < input.size(); i++)
        {
            output.push_back(dct3_detail(input, i));
        }
        return output;
    }

    template<arithmetic ElementT = double, arithmetic OutputT = double>
    constexpr static Image<ElementT> idct3_detail(std::vector<Image<ElementT>> input, int plane_index)
    {
        std::size_t N3 = input.size();
        auto output = TinyDIP::Image<OutputT>(input[plane_index].getWidth(), input[plane_index].getHeight());
        for (std::size_t y = 0; y < output.getHeight(); y++)
        {
            for (std::size_t x = 0; x < output.getWidth(); x++)
            {
                OutputT sum{};
                for (std::size_t inner_z = 0; inner_z < N3; inner_z++)
                {
                    auto plane = input[inner_z];
                    auto N1 = static_cast<OutputT>(plane.getWidth());
                    auto N2 = static_cast<OutputT>(plane.getHeight());
                    for (std::size_t inner_y = 0; inner_y < plane.getHeight(); inner_y++)
                    {
                        for (std::size_t inner_x = 0; inner_x < plane.getWidth(); inner_x++)
                        {
                            auto l1 = (std::numbers::pi / (2 * N1) * (2 * x + 1) * static_cast<OutputT>(inner_x));
                            auto l2 = (std::numbers::pi / (2 * N2) * (2 * y + 1) * static_cast<OutputT>(inner_y));
                            auto l3 = (std::numbers::pi / (2 * static_cast<OutputT>(N3)) * (2 * static_cast<OutputT>(plane_index) + 1) * static_cast<OutputT>(inner_z));
                            OutputT alpha1 = (inner_x == 0) ? (static_cast<OutputT>(1.0) / static_cast<OutputT>(std::sqrt(2))) : (static_cast<OutputT>(1.0));
                            OutputT alpha2 = (inner_y == 0) ? (static_cast<OutputT>(1.0) / static_cast<OutputT>(std::sqrt(2))) : (static_cast<OutputT>(1.0));
                            OutputT alpha3 = (inner_z == 0) ? (static_cast<OutputT>(1.0) / static_cast<OutputT>(std::sqrt(2))) : (static_cast<OutputT>(1.0));
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

    template<arithmetic ElementT = double, arithmetic OutputT = double>
    constexpr static std::vector<Image<ElementT>> idct3(std::vector<Image<ElementT>> input)
    {
        std::vector<Image<ElementT>> output;
        for (std::size_t i = 0; i < input.size(); i++)
        {
            output.push_back(idct3_detail(input, i));
        }
        return output;
    }

    template<arithmetic ElementT = double, arithmetic OutputT = ElementT>
    constexpr static Image<ElementT> dct2(Image<ElementT> input)
    {
        Image<ElementT> output;
        std::vector v{ input };
        output = dct3_detail(v, 0);
        return output;
    }

    template<arithmetic ElementT = double, arithmetic OutputT = ElementT>
    constexpr static Image<ElementT> idct2(Image<ElementT> input)
    {
        Image<ElementT> output;
        std::vector v{ input };
        output = idct3_detail(v, 0);
        return output;
    }
}

#endif