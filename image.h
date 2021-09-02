/* Developed by Jimmy Hu */

#ifndef Image_H
#define Image_H

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <complex>
#include <concepts>
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
            width(width),
            height(height),
            image_data(width * height) { }

        Image(const std::size_t width, const std::size_t height, const ElementT initVal):
            width(width),
            height(height),
            image_data(width * height, initVal) {}

        Image(const std::vector<ElementT>& input, std::size_t newWidth, std::size_t newHeight):
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

        constexpr std::size_t getWidth() const
        {
            return this->width;
        }

        constexpr std::size_t getHeight() const
        {
            return this->height;
        }

        constexpr auto getSize()
        {
            return std::make_tuple(this->width, this->height);
        }

        std::vector<ElementT> const& getImageData() const { return this->image_data; }      //  expose the internal data

        void print(std::string separator = "\t", std::ostream& os = std::cout)
        {
            for (std::size_t y = 0; y < this->height; ++y)
            {
                for (std::size_t x = 0; x < this->width; ++x)
                {
                    //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                    os << +this->at(x, y) << separator;
                }
                os << "\n";
            }
            os << "\n";
            return;
        }

        //  Enable this function if ElementT = RGB
        void print(std::string separator = "\t", std::ostream& os = std::cout) requires(std::same_as<ElementT, RGB>)
        {
            for (std::size_t y = 0; y < this->height; ++y)
            {
                for (std::size_t x = 0; x < this->width; ++x)
                {
                    os << "( ";
                    for (std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                    {
                        //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                        os << +this->at(x, y).channels[channel_index] << separator;
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
            for (std::size_t y = 0; y < rhs.height; ++y)
            {
                for (std::size_t x = 0; x < rhs.width; ++x)
                {
                    //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                    os << +rhs.at(x, y) << separator;
                }
                os << "\n";
            }
            os << "\n";
            return os;
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

        bool operator==(const Image<ElementT>& rhs) const
        {
            /* do actual comparison */
            if (rhs.width != this->width ||
                rhs.height != this->height)
            {
                return false;
            }
            return rhs.image_data == this->image_data;
        }

        bool operator!=(const Image<ElementT>& rhs) const
        {
            return !(this == rhs);
        }

        Image<ElementT>& operator=(Image<ElementT> const& input) = default;  //  Copy Assign

        Image<ElementT>& operator=(Image<ElementT>&& other) = default;       //  Move Assign

        Image(const Image<ElementT> &input) = default;                       //  Copy Constructor

        Image(Image<ElementT> &&input) = default;                            //  Move Constructor
        
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
        std::size_t width;
        std::size_t height;
        std::vector<ElementT> image_data;

        void checkBoundary(const size_t x, const size_t y) const
        {
            assert(x < width);
            assert(y < height);
        }
    };
}


#endif

