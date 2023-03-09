/* Developed by Jimmy Hu */

#ifndef TinyDIP_Image_H
#define TinyDIP_Image_H

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <complex>
#include <concepts>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <numeric>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
#include "image_operations.h"

#ifdef USE_BOOST_SERIALIZATION
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/vector.hpp>
#endif

namespace TinyDIP
{
    template <typename ElementT>
    class Image
    {
    public:
        Image() = default;

        Image(const std::size_t width, const std::size_t height):
            image_data(width * height)
            {
                size.reserve(2);
                size.emplace_back(width);
                size.emplace_back(height);
            }

        Image(const std::size_t width, const std::size_t height, const ElementT initVal):
            image_data(width * height, initVal)
            {
                size.reserve(2);
                size.emplace_back(width);
                size.emplace_back(height);
            }

        Image(const std::vector<ElementT>& input, std::size_t newWidth, std::size_t newHeight)
        {
            size.reserve(2);
            size.emplace_back(newWidth);
            size.emplace_back(newHeight);
            if (input.size() != newWidth * newHeight)
            {
                throw std::runtime_error("Image data input and the given size are mismatched!");
            }
            image_data = input;
        }

        Image(std::vector<ElementT>&& input, std::size_t newWidth, std::size_t newHeight)
        {
            size.reserve(2);
            size.emplace_back(newWidth);
            size.emplace_back(newHeight);
            if (input.size() != newWidth * newHeight)
            {
                throw std::runtime_error("Image data input and the given size are mismatched!");
            }
            image_data = std::move(input);              //  Reference: https://stackoverflow.com/a/51706522/6667035
        }

        Image(const std::vector<std::vector<ElementT>>& input)
        {
            size.reserve(2);
            size.emplace_back(input[0].size());
            size.emplace_back(input.size());
            for (auto& rows : input)
            {
                image_data.insert(image_data.end(), std::ranges::begin(input), std::ranges::end(input));    //  flatten
            }
            return;
        }

        template<typename... Args>
        constexpr ElementT& at(const Args... indexInput)
        {
            checkBoundary(indexInput...);
            auto x = get_from_variadic_template<1>(indexInput...);
            auto y = get_from_variadic_template<2>(indexInput...);
            return image_data[y * size[0] + x];
        }

        template<typename... Args>
        constexpr ElementT const& at(const Args... indexInput) const
        {
            checkBoundary(indexInput...);
            auto x = get_from_variadic_template<1>(indexInput...);
            auto y = get_from_variadic_template<2>(indexInput...);
            return image_data[y * size[0] + x];
        }

        constexpr std::size_t getWidth() const
        {
            return size[0];
        }

        constexpr std::size_t getHeight() const noexcept
        {
            return size[1];
        }

        constexpr auto getSize() noexcept
        {
            return std::make_tuple(size[0], size[1]);
        }

        std::vector<ElementT> const& getImageData() const noexcept { return image_data; }      //  expose the internal data

        void print(std::string separator = "\t", std::ostream& os = std::cout) const
        {
            for (std::size_t y = 0; y < size[1]; ++y)
            {
                for (std::size_t x = 0; x < size[0]; ++x)
                {
                    //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                    os << +at(x, y) << separator;
                }
                os << "\n";
            }
            os << "\n";
            return;
        }

        //  Enable this function if ElementT = RGB
        void print(std::string separator = "\t", std::ostream& os = std::cout) const
        requires(std::same_as<ElementT, RGB>)
        {
            for (std::size_t y = 0; y < size[1]; ++y)
            {
                for (std::size_t x = 0; x < size[0]; ++x)
                {
                    os << "( ";
                    for (std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                    {
                        //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                        os << +at(x, y).channels[channel_index] << separator;
                    }
                    os << ")" << separator;
                }
                os << "\n";
            }
            os << "\n";
            return;
        }

        friend std::ostream& operator<<(std::ostream& os, const Image<ElementT>& rhs)
        {
            const std::string separator = "\t";
            rhs.print(separator, os);
            return os;
        }

        Image<ElementT>& operator+=(const Image<ElementT>& rhs)
        {
            check_size_same(rhs, *this);
            std::transform(std::ranges::cbegin(image_data), std::ranges::cend(image_data), std::ranges::cbegin(rhs.image_data),
                   std::ranges::begin(image_data), std::plus<>{});
            return *this;
        }

        Image<ElementT>& operator-=(const Image<ElementT>& rhs)
        {
            check_size_same(rhs, *this);
            std::transform(std::ranges::cbegin(image_data), std::ranges::cend(image_data), std::ranges::cbegin(rhs.image_data),
                   std::ranges::begin(image_data), std::minus<>{});
            return *this;
        }

        Image<ElementT>& operator*=(const Image<ElementT>& rhs)
        {
            check_size_same(rhs, *this);
            std::transform(std::ranges::cbegin(image_data), std::ranges::cend(image_data), std::ranges::cbegin(rhs.image_data),
                   std::ranges::begin(image_data), std::multiplies<>{});
            return *this;
        }

        Image<ElementT>& operator/=(const Image<ElementT>& rhs)
        {
            check_size_same(rhs, *this);
            std::transform(std::ranges::cbegin(image_data), std::ranges::cend(image_data), std::ranges::cbegin(rhs.image_data),
                   std::ranges::begin(image_data), std::divides<>{});
            return *this;
        }

        friend bool operator==(Image<ElementT> const&, Image<ElementT> const&) = default;

        friend bool operator!=(Image<ElementT> const&, Image<ElementT> const&) = default;

        friend Image<ElementT> operator+(Image<ElementT> input1, const Image<ElementT>& input2)
        {
            return input1 += input2;
        }

        friend Image<ElementT> operator-(Image<ElementT> input1, const Image<ElementT>& input2)
        {
            return input1 -= input2;
        }

        friend Image<ElementT> operator*(Image<ElementT> input1, ElementT input2)
        {
            return multiplies(input1, input2);
        }

        friend Image<ElementT> operator*(ElementT input1, Image<ElementT> input2)
        {
            return multiplies(input2, input1);
        }
        
#ifdef USE_BOOST_SERIALIZATION

        void Save(std::string filename)
        {
            const std::string filename_with_extension = filename + ".dat";
            //	Reference: https://stackoverflow.com/questions/523872/how-do-you-serialize-an-object-in-c
            std::ofstream ofs(filename_with_extension, std::ios::binary);
            boost::archive::binary_oarchive ArchiveOut(ofs);
            //	write class instance to archive
            ArchiveOut << *this;
            //	archive and stream closed when destructors are called
            ofs.close();
        }
        
#endif
    private:
        std::vector<std::size_t> size;
        std::vector<ElementT> image_data;

        template<typename... Args>
        void checkBoundary(const Args... indexInput) const
        {
            if (get_from_variadic_template<1>(indexInput) >= size[0])
                throw std::out_of_range("Given x out of range!");
            if (get_from_variadic_template<2>(indexInput) >= size[1])
                throw std::out_of_range("Given y out of range!");
        }

#ifdef USE_BOOST_SERIALIZATION
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar& size;
            ar& image_data;
        }
        /*
        static bool is_file_exist(const char* file_name)
        {
            if (access(file_name, F_OK) != -1)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        */
#endif

    };

    template<typename T, typename ElementT>
    concept is_Image = std::is_same_v<T, Image<ElementT>>;
}


#endif

