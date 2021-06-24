/* Develop by Jimmy Hu */

#ifndef Image_H
#define Image_H

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <concepts>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <numeric>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
#include "basic_functions.h"

namespace TinyDIP
{
    template <typename ElementT>
    class Image
    {
    public:
        Image()
        {
        }

        Image(const size_t newWidth, const size_t newHeight)
        {
            this->width = newWidth;
            this->height = newHeight;
            this->image_data.resize(this->height * this->width);
            this->image_data = recursive_transform<1>(this->image_data, [](ElementT element) { return ElementT{}; });
            return;
        }

        Image(const int newWidth, const int newHeight, const ElementT initVal)
        {
            this->width = newWidth;
            this->height = newHeight;
            this->image_data.resize(this->height * this->width);
            this->image_data = recursive_transform<1>(this->image_data, [initVal](ElementT element) { return initVal; });
            return;
        }

        Image(const std::vector<std::vector<ElementT>>& input)
        {
            for (auto& rows : input)
            {
                this->image_data.insert(this->image_data.end(), std::begin(input), std::end(input));
            }
            return;
        }

        template<class OutputT>
        constexpr auto cast()
        {
            return this->transform([](ElementT element) { return static_cast<OutputT>(element); });
        }

        constexpr ElementT& at(const unsigned int x, const unsigned int y) { return this->image_data[y * width + x]; }

        constexpr ElementT const& at(const unsigned int x, const unsigned int y) const { return this->image_data[y * width + x]; }

        constexpr size_t getWidth()
        {
            return this->width;
        }

        constexpr size_t getHeight()
        {
            return this->height;
        }

        constexpr auto getData()
        {
            return this->transform([](ElementT element) { return element; });   //  Deep copy
        }

        void print()
        {
            for (auto& row_element : this->toString())
            {
                for (auto& element : row_element)
                {
                    std::cout << element << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
            return;
        }

        constexpr auto toString()
        {
            return this->transform([](ElementT element) { return std::to_string(element); });
        }

        Image<ElementT>& operator=(Image<ElementT> const& input) = default;  //  Copy Assign

        Image<ElementT>& operator=(Image<ElementT>&& other) = default;       //  Move Assign

        Image(const Image<ElementT> &input) = default;                       //  Copy Constructor

        /*    Move Constructor
         */
        Image(Image<ElementT> &&input) : image_data(std::move(input.image_data))
        {
        }
        
    private:
        std::vector<std::vector<ElementT>> image_data;

        template<class F>
        constexpr auto transform(const F& f)
        {
            return recursive_transform<2>(this->image_data, f);
        }

        template<class InputT>
        constexpr auto bicubicPolate(const ElementT* const ndata, const InputT& fracx, const InputT& fracy)
        {
            auto x1 = cubicPolate( ndata[0], ndata[1], ndata[2], ndata[3], fracx );
            auto x2 = cubicPolate( ndata[4], ndata[5], ndata[6], ndata[7], fracx );
            auto x3 = cubicPolate( ndata[8], ndata[9], ndata[10], ndata[11], fracx );
            auto x4 = cubicPolate( ndata[12], ndata[13], ndata[14], ndata[15], fracx );

            return clip(cubicPolate( x1, x2, x3, x4, fracy ), 0.0, 255.0);
        }

        template<class InputT1, class InputT2>
        constexpr auto cubicPolate(const InputT1& v0, const InputT1& v1, const InputT1& v2, const InputT1& v3, const InputT2& frac)
        {
            auto A = (v3-v2)-(v0-v1);
            auto B = (v0-v1)-A;
            auto C = v2-v0;
            auto D = v1;
            return D + frac * (C + frac * (B + frac * A));
        }

        template<class InputT1, class InputT2, class InputT3>
        constexpr auto clip(const InputT1& input, const InputT2& lowerbound, const InputT3& upperbound)
        {
            if (input < lowerbound)
            {
                return static_cast<InputT1>(lowerbound);
            }
            if (input > upperbound)
            {
                return static_cast<InputT1>(upperbound);
            }
            return input;
        }

        void setImageDataToBlack()
        {
            
        }
    };
}


#endif

