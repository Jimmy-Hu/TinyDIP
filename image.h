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
            this->image_data.resize(newHeight);
            for (size_t i = 0; i < newHeight; ++i) {
                this->image_data[i].resize(newWidth);
            }
            this->image_data = recursive_transform<2>(this->image_data, [initVal](ElementT element) { return initVal; });
            return;
        }

        Image(const std::vector<std::vector<ElementT>>& input)
        {
            this->image_data = recursive_transform<2>(input, [](ElementT element) {return element; } ); //  Deep copy
            return;
        }

        template<class OutputT>
        constexpr auto cast()
        {
            return this->transform([](ElementT element) { return static_cast<OutputT>(element); });
        }

        constexpr auto get(const unsigned int locationx, const unsigned int locationy)
        {
            return this->image_data[locationy][locationx];
        }

        constexpr auto set(const unsigned int locationx, const unsigned int locationy, const ElementT& element)
        {
            this->image_data[locationy][locationx] = element;
            return *this;
        }

        template<class InputT>
        constexpr auto set(const unsigned int locationx, const unsigned int locationy, const InputT& element)
        {
            this->image_data[locationy][locationx] = static_cast<ElementT>(element);
            return *this;
        }

        constexpr auto getSizeX()
        {
            return this->image_data[0].size();
        }

        constexpr auto getSizeY()
        {
            return this->image_data.size();
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

        constexpr auto bicubicInterpolation(const int& newSizeX, const int& newSizeY)
        {
            auto output = Image<ElementT>(newSizeX, newSizeY);
            auto ratiox = (float)this->getSizeX() / (float)newSizeX;
            auto ratioy = (float)this->getSizeY() / (float)newSizeY;
            
            for (size_t y = 0; y < newSizeY; y++)
            {
                for (size_t x = 0; x < newSizeX; x++)
                {
                    float xMappingToOrigin = (float)x * ratiox;
                    float yMappingToOrigin = (float)y * ratioy;
                    float xMappingToOriginFloor = floor(xMappingToOrigin);
                    float yMappingToOriginFloor = floor(yMappingToOrigin);
                    float xMappingToOriginFrac = xMappingToOrigin - xMappingToOriginFloor;
                    float yMappingToOriginFrac = yMappingToOrigin - yMappingToOriginFloor;
                    
                    ElementT ndata[4 * 4];
                    for (int ndatay = -1; ndatay <= 2; ndatay++)
                    {
                        for (int ndatax = -1; ndatax <= 2; ndatax++)
                        {
                            ndata[(ndatay + 1) * 4 + (ndatax + 1)] = this->get(
                                clip(xMappingToOriginFloor + ndatax, 0, this->getSizeX() - 1), 
                                clip(yMappingToOriginFloor + ndatay, 0, this->getSizeY() - 1));
                        }
                        
                    }
                    output.set(x, y, bicubicPolate(ndata, xMappingToOriginFrac, yMappingToOriginFrac));
                }
            }
            return output;
        }

        Image<ElementT>& operator=(Image<ElementT> const& input) = default;  //  Copy Assign

        Image<ElementT>& operator=(Image<ElementT>&& other) = default;       //  Move Assign

        Image(const Image<ElementT> &input)                      //  Copy Constructor
        {
            this->image_data = input.getData();
        }

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

