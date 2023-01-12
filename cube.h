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

        Cube(const std::size_t newWidth, const std::size_t newHeight, const std::size_t newDepth)
        {
            this->data.resize(newDepth);
            for (size_t i = 0; i < newDepth; ++i) {
                this->data[i].resize(newHeight);
                for (size_t j = 0; j < newHeight; j++) {
                    this->data[i][j].resize(newWidth);
                }
            }
            this->data = recursive_transform<3>(this->data, [](ElementT element) { return ElementT{}; });
            return;
        }

        Cube(const int newWidth, const int newHeight, const int newDepth, ElementT initVal)
        {
            data.resize(newDepth);
            for (size_t i = 0; i < newDepth; ++i) {
                data[i].resize(newHeight);
                for (size_t j = 0; j < newHeight; j++) {
                    this->data[i][j].resize(newWidth);
                }
            }
            this->data = recursive_transform<3>(this->data, [initVal](ElementT element) { return initVal; });
            return;
        }

        Cube(const std::vector<std::vector<std::vector<ElementT>>>& input)
        {
            this->data = recursive_transform<3>(input, [](ElementT element) {return element; } ); //  Deep copy
            return;
        }

        template<class OutputT>
        constexpr auto cast()
        {
            return this->transform([](ElementT element) { return static_cast<OutputT>(element); });
        }

        constexpr auto get(const unsigned int locationx, const unsigned int locationy, const unsigned int locationz)
        {
            return this->data[locationz][locationy][locationx];
        }

        constexpr auto set(const unsigned int locationx, const unsigned int locationy, const unsigned int locationz, ElementT element)
        {
            this->data[locationz][locationy][locationx] = element;
            return *this;
        }

        template<class InputT>
        constexpr auto set(const unsigned int locationx, const unsigned int locationy, const unsigned int locationz, const InputT& element)
        {
            this->image_data[locationz][locationy][locationx] = static_cast<ElementT>(element);
            return *this;
        }

        constexpr auto getSizeX()
        {
            return width;
        }

        constexpr auto getSizeY()
        {
            return height;
        }

        constexpr auto getSizeZ()
        {
            return depth;
        }

        std::vector<ElementT> const& getData() const noexcept { return data; }      //  expose the internal data

        void print(std::string separator = "\t", std::ostream& os = std::cout) const
        {
            for (std::size_t y = 0; y < height; ++y)
            {
                for (std::size_t x = 0; x < width; ++x)
                {
                    //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                    os << +at(x, y) << separator;
                }
                os << "\n";
            }
            os << "\n";
            return;
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

        Cube<ElementT>& operator=(Cube<ElementT> const& input) = default;   //  Copy Assign

        Cube<ElementT>& operator=(Cube<ElementT>&& other) = default;        //  Move Assign

        Cube(const Image<Cube> &input) = default;                           //  Copy Constructor

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

