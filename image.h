/* Developed by Jimmy Hu */

#ifndef Image_H
#define Image_H

#include <algorithm>
#include <array>
#include <cassert>
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
        Image() = default;

        Image(const size_t width, const size_t height):
            width(width),
            height(height),
            image_data(width * height) { }

        Image(const int width, const int height, const ElementT initVal):
            width(width),
            height(height),
            image_data(width * height, initVal) {}

        Image(const std::vector<ElementT>& input, size_t newWidth, size_t newHeight):
            width(newWidth),
            height(newHeight)
        {
            assert(input.size() == newWidth * newHeight);
            this->image_data = input;   //  Deep copy
        }

        Image(const std::vector<std::vector<ElementT>>& input)
        {
            this->height = input.size();
            this->width = input[0].size();
            
            for (auto& rows : input)
            {
                this->image_data.insert(this->image_data.end(), std::begin(input), std::end(input));    //  flatten
            }
            return;
        }

        constexpr ElementT& at(const unsigned int x, const unsigned int y)
        { 
            checkBoundary(x, y);
            return this->image_data[y * width + x];
        }

        constexpr ElementT const& at(const unsigned int x, const unsigned int y) const
        {
            checkBoundary(x, y);
            return this->image_data[y * width + x];
        }

        constexpr size_t getWidth()
        {
            return this->width;
        }

        constexpr size_t getHeight()
        {
            return this->height;
        }

        std::vector<ElementT> const& getImageData() const { return this->image_data; }      //  expose the internal data

        void print()
        {
            for (size_t y = 0; y < this->height; ++y)
            {
                for (size_t x = 0; x < this->width; ++x)
                {
                    //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                    std::cout << +this->at(x, y) << "\t";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
            return;
        }

        Image<ElementT>& operator+=(const Image<ElementT>& rhs)
        {
            assert(rhs.width == this->width);
            assert(rhs.height == this->height);
            std::transform(image_data.cbegin(), image_data.cend(), rhs.image_data.cbegin(),
                   image_data.begin(), std::plus<>{});
            return *this;
        }

        Image<ElementT>& operator-=(const Image<ElementT>& rhs)
        {
            assert(rhs.width == this->width);
            assert(rhs.height == this->height);
            std::transform(image_data.cbegin(), image_data.cend(), rhs.image_data.cbegin(),
                   image_data.begin(), std::minus<>{});
            return *this;
        }

        Image<ElementT>& operator*=(const Image<ElementT>& rhs)
        {
            assert(rhs.width == this->width);
            assert(rhs.height == this->height);
            std::transform(image_data.cbegin(), image_data.cend(), rhs.image_data.cbegin(),
                   image_data.begin(), std::multiplies<>{});
            return *this;
        }

        Image<ElementT>& operator/=(const Image<ElementT>& rhs)
        {
            assert(rhs.width == this->width);
            assert(rhs.height == this->height);
            std::transform(image_data.cbegin(), image_data.cend(), rhs.image_data.cbegin(),
                   image_data.begin(), std::divides<>{});
            return *this;
        }

        Image<ElementT>& operator=(Image<ElementT> const& input) = default;  //  Copy Assign

        Image<ElementT>& operator=(Image<ElementT>&& other) = default;       //  Move Assign

        Image(const Image<ElementT> &input) = default;                       //  Copy Constructor

        Image(Image<ElementT> &&input) = default;                            //  Move Constructor
        
    private:
        size_t width;
        size_t height;
        std::vector<ElementT> image_data;

        void checkBoundary(const size_t x, const size_t y)
        {
            assert(x < width);
            assert(y < height);
        }
    };
}


#endif

