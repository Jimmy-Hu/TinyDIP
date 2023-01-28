/* Developed by Jimmy Hu */

#ifndef Cube_H
#define Cube_H

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
	class Cube
	{
	public:
		Cube() = default;

        Cube(const std::size_t newWidth, const std::size_t newHeight, const std::size_t newDepth):
            width(width),
            height(height),
            depth(newDepth),
            data(width * height * depth) { }

        Cube(const int newWidth, const int newHeight, const int newDepth, ElementT initVal):
            width(newWidth),
            height(newHeight),
            depth(newDepth),
            data(width * height * depth, initVal) {}

        Cube(const std::vector<ElementT>& input, std::size_t newWidth, std::size_t newHeight, const std::size_t newDepth):
            width(newWidth),
            height(newHeight),
            depth(newDepth)
        {
            if (input.size() != newWidth * newHeight * newDepth)
            {
                throw std::runtime_error("Data input and the given size are mismatched!");
            }
            data = input;
        }    

        Cube(const std::vector<Image<ElementT>>& input)
        {
            width = input[0].getWidth();
            height = input[0].getHeight();
            depth = input.size();
            data.resize(width * height * depth);
            
            for (std::size_t z = 0; z < input.size(); ++z)
            {
                auto image = input[z];
                for (std::size_t y = 0; y < height; ++y)
                {
                    for (std::size_t x = 0; x < width; ++x)
                    {
                        data[z * width * height + y * width + x] = image.at(x, y);
                    }
                }
            }
            return;
        }

        constexpr ElementT& at(const unsigned int x, const unsigned int y, const unsigned int z)
        { 
            checkBoundary(x, y, z);
            return data[z * width * height + y * width + x];
        }

        constexpr ElementT const& at(const unsigned int x, const unsigned int y, const unsigned int z) const
        {
            checkBoundary(x, y, z);
            return data[z * width * height + y * width + x];
        }

        constexpr auto getSizeX() const noexcept
        {
            return width;
        }

        constexpr auto getSizeY() const noexcept
        {
            return height;
        }

        constexpr auto getSizeZ() const noexcept
        {
            return depth;
        }

        std::vector<ElementT> const& getData() const noexcept { return data; }      //  expose the internal data

        void print(std::string separator = "\t", std::ostream& os = std::cout) const
        {
            for(std::size_t z = 0; z < depth; ++z)
            {
                for (std::size_t y = 0; y < height; ++y)
                {
                    for (std::size_t x = 0; x < width; ++x)
                    {
                        //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                        os << +at(x, y, z) << separator;
                    }
                    os << "\n";
                }
                os << "\n";
            }
            os << "\n";
            return;
        }

        friend std::ostream& operator<<(std::ostream& os, const Cube<ElementT>& rhs)
        {
            const std::string separator = "\t";
            rhs.print(separator, os);
            return os;
        }

        Cube<ElementT>& operator+=(const Cube<ElementT>& rhs)
        {
            check_size_same(rhs, *this);
            std::transform(std::ranges::cbegin(data), std::ranges::cend(data), std::ranges::cbegin(rhs.data),
                   std::ranges::begin(data), std::plus<>{});
            return *this;
        }

        Cube<ElementT>& operator-=(const Cube<ElementT>& rhs)
        {
            check_size_same(rhs, *this);
            std::transform(std::ranges::cbegin(data), std::ranges::cend(data), std::ranges::cbegin(rhs.data),
                   std::ranges::begin(data), std::minus<>{});
            return *this;
        }

        Cube<ElementT>& operator*=(const Cube<ElementT>& rhs)
        {
            check_size_same(rhs, *this);
            std::transform(std::ranges::cbegin(data), std::ranges::cend(data), std::ranges::cbegin(rhs.data),
                   std::ranges::begin(data), std::multiplies<>{});
            return *this;
        }

        Cube<ElementT>& operator/=(const Cube<ElementT>& rhs)
        {
            check_size_same(rhs, *this);
            std::transform(std::ranges::cbegin(data), std::ranges::cend(data), std::ranges::cbegin(rhs.data),
                   std::ranges::begin(data), std::divides<>{});
            return *this;
        }

        friend bool operator==(Cube<ElementT> const&, Cube<ElementT> const&) = default;

        friend bool operator!=(Cube<ElementT> const&, Cube<ElementT> const&) = default;

        friend Cube<ElementT> operator+(Cube<ElementT> input1, const Cube<ElementT>& input2)
        {
            return input1 += input2;
        }

        friend Cube<ElementT> operator-(Cube<ElementT> input1, const Cube<ElementT>& input2)
        {
            return input1 -= input2;
        }

        friend Image<ElementT> operator*(Image<ElementT> input1, ElementT input2)
        {
            return multiplies(input1, input2);
        }

        Cube<ElementT>& operator=(Cube<ElementT> const& input) = default;   //  Copy Assign

        Cube<ElementT>& operator=(Cube<ElementT>&& other) = default;        //  Move Assign

        Cube(const Cube<ElementT> &input) = default;                        //  Copy Constructor

        Cube(Cube<ElementT> &&input) = default;                             //  Move Constructor
        
    private:
        std::size_t width;
        std::size_t height;
        std::size_t depth;
        std::vector<ElementT> data;
        
        void checkBoundary(const size_t x, const size_t y, const size_t z) const
        {
            if (x >= width)
                throw std::out_of_range("Given x out of range!");
            if (y >= height)
                throw std::out_of_range("Given y out of range!");
            if (z >= depth)
                throw std::out_of_range("Given z out of range!");
        }
    };
}


#endif

