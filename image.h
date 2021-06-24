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
#include "image_operations.h"

namespace TinyDIP
{
    template <typename ElementT>
    class Image
    {
    public:
        Image()
        {
        }

        Image(const size_t width, const size_t height):
            width(width),
            height(height),
            image_data(width * height) { }

        Image(const int width, const int height, const ElementT initVal):
            width(width),
            height(height),
            image_data(width * height)
        {
            this->image_data = recursive_transform<1>(this->image_data, [initVal](ElementT element) { return initVal; });
            return;
        }

        Image(const std::vector<ElementT>& input, size_t newWidth, size_t newHeight)
        {
            this->width = newWidth;
            this->height = newHeight;
            this->image_data = recursive_transform<1>(input, [](ElementT element) { return element; });   //  Deep copy
        }

        Image(const std::vector<std::vector<ElementT>>& input)
        {
            this->height = input.size();
            this->width = input[0].size();
            
            for (auto& rows : input)
            {
                this->image_data.insert(this->image_data.end(), std::begin(input), std::end(input));
            }
            return;
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

        std::vector<ElementT> const& getImageData() const { return this->image_data; }      //  expose the internal data

        void print()
        {
            for (size_t y = 0; y < this->height; y++)
            {
                for (size_t x = 0; x < this->width; x++)
                {
                    //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                    std::cout << +this->at(x, y) << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
            return;
        }

        constexpr auto toString()
        {
            return TinyDIP::Image(this->transform([](ElementT element) { return std::to_string(element); }), this->width, this->height);
        }

        Image<ElementT>& operator=(Image<ElementT> const& input) = default;  //  Copy Assign

        Image<ElementT>& operator=(Image<ElementT>&& other) = default;       //  Move Assign

        Image(const Image<ElementT> &input) = default;                       //  Copy Constructor

        Image(Image<ElementT> &&input) = default;                            //  Move Constructor
        
    private:
        size_t width;
        size_t height;
        std::vector<ElementT> image_data;

        template<class F>
        constexpr auto transform(const F& f)
        {
            return recursive_transform<2>(this->image_data, f);
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

