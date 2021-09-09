/* Developed by Jimmy Hu */

#ifndef ImageOperations_H
#define ImageOperations_H

#include <fstream>
#include <numbers>
#include <string>
#include "base_types.h"
#include "image.h"

#define is_size_same(x, y) {assert(x.getWidth() == y.getWidth()); assert(x.getHeight() == y.getHeight());}

namespace TinyDIP
{
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

    template<class InputT = RGB, class OutputT = GrayScale>
    constexpr static auto getPlane(Image<InputT> input, std::size_t index)
    {
        Image<OutputT> output(input.getWidth(), input.getHeight());
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

    template<class T = RGB, class F, class... Args>
    requires (std::same_as<T, RGB>)
    constexpr static auto apply_each(Image<T> input, F operation, Args&&... args)
    {
        return constructRGB(operation(getRplane(input), args...), operation(getGplane(input), args...), operation(getBplane(input), args...));
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
        const double correlation, const double normalize_factor = 1.0);

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
}

#endif