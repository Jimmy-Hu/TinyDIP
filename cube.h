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

        Cube(const int newWidth, const int newHeight, const int newDepth)
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
            this->data.resize(newDepth);
            for (size_t i = 0; i < newDepth; ++i) {
                this->data[i].resize(newHeight);
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
            return this->data[0][0].size();
        }

        constexpr auto getSizeY()
        {
            return this->data[0].size();
        }

        constexpr auto getSizeZ()
        {
            return this->data.size();
        }

        constexpr auto getData()
        {
            return this->transform([](ElementT element) { return element; });   //  Deep copy
        }

        void print()
        {
            for (auto& element_group1 : this->toString())
            {
                for (auto& element_group2 : element_group1)
                {
                    for (auto& element : element_group2)
                    {
                        std::cout << element << "\t";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
                std::cout << "\n";
            }
            return;
        }

        constexpr auto toString()
        {
            return this->transform([](ElementT element) { return std::to_string(element); });
        }
        
    private:
        std::vector<std::vector<std::vector<ElementT>>> data;

        template<class F>
        constexpr auto transform(const F& f)
        {
            return recursive_transform<3>(this->data, f);
        }
    };
}


#endif

