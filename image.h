/* Developed by Jimmy Hu */

#ifndef TINYDIP_IMAGE_H       // image.h header guard, follow the suggestion from https://codereview.stackexchange.com/a/293832/231235
#define TINYDIP_IMAGE_H

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
#include "base_types.h"
#include "basic_functions.h"

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

        template<std::same_as<std::size_t>... Sizes>
        Image(Sizes... sizes): size{sizes...}, image_data((1 * ... * sizes))
        {}

        template<std::same_as<int>... Sizes>
        Image(Sizes... sizes)
        {
            size.reserve(sizeof...(sizes));
            (size.emplace_back(sizes), ...);
            image_data.resize(
                std::reduce(
                    std::ranges::cbegin(size),
                    std::ranges::cend(size),
                    std::size_t{1},
                    std::multiplies<>()
                    )
            );
        }

        //  Image constructor
        template<std::ranges::input_range Sizes>
        requires(std::same_as<std::ranges::range_value_t<Sizes>, std::size_t>)
        Image(const Sizes& sizes)
        {
            if (sizes.empty())
            {
                throw std::runtime_error("Image size vector is empty!");
            }
            size.resize(sizes.size());
            std::transform(std::ranges::cbegin(sizes), std::ranges::cend(sizes), std::ranges::begin(size), [&](auto&& element) { return element; });
            image_data.resize(
                std::reduce(
                    std::ranges::cbegin(sizes),
                    std::ranges::cend(sizes),
                    std::size_t{1},
                    std::multiplies<>()
                    ));
        }

        //  Image constructor
        #ifdef __cpp_lib_containers_ranges
            template<std::ranges::input_range Range,
                     std::same_as<std::size_t>... Sizes>
            Image(const Range&& input, Sizes... sizes):
                size{sizes...}, image_data(std::from_range_t, std::forward<Range>(input))
            {
                if (image_data.size() != (1 * ... * sizes)) {
                    throw std::runtime_error("Image data input and the given size are mismatched!");
                }
            }
        #else
            template<std::ranges::input_range Range,
                     std::same_as<std::size_t>... Sizes>
            Image(const Range&& input, Sizes... sizes):
                size{sizes...}, image_data(input.begin(), input.end())
            {
                if (image_data.size() != (1 * ... * sizes)) {
                    throw std::runtime_error("Image data input and the given size are mismatched!");
                }
            }
        #endif

        //  Image constructor
        #ifdef __cpp_lib_containers_ranges
            template<std::ranges::input_range Range, std::same_as<std::size_t>... Sizes>
            Image(const Range& input, Sizes... sizes) :
                size{ sizes... }
            {
                if (input.empty())
                {
                    throw std::runtime_error("Input vector is empty!");
                }
                image_data = std::vector(std::from_range_t, input);
                if (image_data.size() != (1 * ... * sizes)) {
                    throw std::runtime_error("Image data input and the given size are mismatched!");
                }
            }
        #else
            template<std::ranges::input_range Range, std::same_as<std::size_t>... Sizes>
            Image(const Range& input, Sizes... sizes):
                size{sizes...}
            {
                if (input.empty())
                {
                    throw std::runtime_error("Input vector is empty!");
                }
                image_data = std::vector(input.begin(), input.end());
                if (image_data.size() != (1 * ... * sizes)) {
                    throw std::runtime_error("Image data input and the given size are mismatched!");
                }
            }
        #endif

        //  Image constructor
        template<std::ranges::input_range Range, std::ranges::input_range Sizes>
        requires(std::same_as<std::ranges::range_value_t<Sizes>, std::size_t>)
        Image(const Range& input, const Sizes& sizes)
        {
            if (input.empty())
            {
                throw std::runtime_error("Input vector is empty!");
            }
            size.resize(sizes.size());
            std::transform(std::ranges::cbegin(sizes), std::ranges::cend(sizes), std::ranges::begin(size), [&](auto&& element) { return element; });
            image_data = std::vector(std::ranges::cbegin(input), std::ranges::cend(input));
            auto count = std::reduce(std::ranges::cbegin(sizes), std::ranges::cend(sizes), 1, std::multiplies());
            if (image_data.size() != count) {
                throw std::runtime_error("Image data input and the given size are mismatched!");
            }
        }

        Image(const std::vector<std::vector<ElementT>>& input)
        {
            if (input.empty())
            {
                throw std::runtime_error("Input vector is empty!");
            }
            size.reserve(2);
            size.emplace_back(input[0].size());
            size.emplace_back(input.size());
            for (auto& rows : input)
            {
                image_data.insert(image_data.end(), std::ranges::begin(rows), std::ranges::end(rows));    //  flatten
            }
            return;
        }

        //  at template function implementation
        template<std::same_as<std::size_t>... Args>
        constexpr ElementT& at(const Args... indexInput)
        {
            return const_cast<ElementT&>(static_cast<const Image &>(*this).at(indexInput...));
        }

        //  at template function implementation
        //  Reference: https://codereview.stackexchange.com/a/288736/231235
        template<std::same_as<std::size_t>... Args>
        constexpr ElementT const& at(const Args... indexInput) const
        {
            checkBoundary(indexInput...);
            return at_without_boundary_check(indexInput...);
        }

        //  at template function implementation
        template<std::same_as<int>... Args>
        constexpr ElementT& at(const Args... indexInput)
        {
            return at(static_cast<std::size_t>(indexInput)...);
        }

        //  at template function implementation
        template<std::same_as<int>... Args>
        constexpr ElementT const& at(const Args... indexInput) const
        {
            return at(static_cast<std::size_t>(indexInput)...);
        }

        //  at_without_boundary_check template function implementation
        template<std::same_as<std::size_t>... Args>
        constexpr ElementT& at_without_boundary_check(const Args... indexInput)
        {
            return const_cast<ElementT&>(static_cast<const Image &>(*this).at_without_boundary_check(indexInput...));
        }

        //  at_without_boundary_check template function implementation
        template<std::same_as<std::size_t>... Args>
        constexpr ElementT const& at_without_boundary_check(const Args... indexInput) const
        {
            constexpr std::size_t n = sizeof...(Args);
            if (n != size.size()) {
                throw std::runtime_error("Dimensionality mismatched!");
            }
            std::size_t index = calculateIndex(indexInput...);
            return image_data[index];
        }

        //  at_without_boundary_check template function implementation
        template<std::same_as<int>... Args>
        constexpr ElementT& at_without_boundary_check(const Args... indexInput)
        {
            return at_without_boundary_check(static_cast<std::size_t>(indexInput)...);
        }

        //  at_without_boundary_check template function implementation
        template<std::same_as<int>... Args>
        constexpr ElementT const& at_without_boundary_check(const Args... indexInput) const
        {
            return at_without_boundary_check(static_cast<std::size_t>(indexInput)...);
        }

        //  at_without_boundary_check template function implementation (std::ranges::input_range case)
        template<std::ranges::input_range Indices>
        requires(std::same_as<std::ranges::range_value_t<Indices>, std::size_t> ||
                 std::same_as<std::ranges::range_value_t<Indices>, int>)
        constexpr ElementT const& at_without_boundary_check(const Indices indexInput) const
        {
            std::vector<std::size_t> index_size_t(indexInput.size());
            std::transform(
                std::ranges::cbegin(indexInput),
                std::ranges::cend(indexInput),
                std::ranges::begin(index_size_t),
                [&](auto&& element) {return static_cast<std::size_t>(element); }        //  casting to std::size_t
            );
            return image_data[calculateIndex(index_size_t)];
        }

        //  at_without_boundary_check template function implementation (std::ranges::input_range case)
        template<std::ranges::input_range Indices>
        requires(std::same_as<std::ranges::range_value_t<Indices>, std::size_t> ||
                 std::same_as<std::ranges::range_value_t<Indices>, int>)
        constexpr ElementT& at_without_boundary_check(const Indices indexInput)
        {
            return const_cast<ElementT&>(static_cast<const Image&>(*this).at_without_boundary_check(indexInput));
        }

        //  get function implementation
        constexpr ElementT get(std::size_t index) const noexcept
        {
            return image_data[index];
        }

        //  set function implementation
        constexpr ElementT& set(const std::size_t index) noexcept
        {
            return image_data[index];
        }

        //  set template function implementation
        template<class TupleT>
        requires(is_tuple<TupleT>::value and
                 check_tuple_element_type<std::size_t, TupleT>::value)
        constexpr bool set(const TupleT location, const ElementT draw_value)
        {
            if (checkBoundaryTuple(location))
            {
                image_data[tuple_location_to_index(location)] = draw_value;
                return true;
            }
            return false;
        }

        //  cast template function implementation
        template<typename TargetT>
        constexpr Image<TargetT> cast()
        {
            std::vector<TargetT> output_data;
            output_data.resize(image_data.size());
            std::transform(
                std::ranges::cbegin(image_data),
                std::ranges::cend(image_data),
                std::ranges::begin(output_data),
                [](auto& input){ return static_cast<TargetT>(input); }
                );
            Image<TargetT> output(output_data, size);
            return output;
        }

        constexpr std::size_t count() const noexcept
        {
            return std::reduce(std::ranges::cbegin(size), std::ranges::cend(size), std::size_t{ 1 }, std::multiplies());
        }

        //  count member function implementation
        template<class ExPo>
        requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
        constexpr std::size_t count(ExPo execution_policy)
        {
            return std::reduce(execution_policy, std::ranges::cbegin(size), std::ranges::cend(size), std::size_t{ 1 }, std::multiplies());
        }
  
        constexpr std::size_t getDimensionality() const noexcept
        {
            return size.size();
        }

        constexpr std::size_t getWidth() const noexcept
        {
            return size[0];
        }

        constexpr std::size_t getHeight() const noexcept
        {
            return size[1];
        }

        //  getSize function implementation
        constexpr auto getSize() const noexcept
        {
            return size;
        }

        //  getSize function implementation
        constexpr auto getSize(std::size_t index) const noexcept
        {
            return size[index];
        }

        //  getStride function implementation
        constexpr std::size_t getStride(std::size_t index) const noexcept
        {
            if(index == 0)
            {
                return std::size_t{1};
            }
            std::size_t output = std::size_t{1};
            for(std::size_t i = 0; i < index; ++i)
            {
                output *= size[i];
            }
            return output;
        }

        std::vector<ElementT> const& getImageData() const noexcept { return image_data; }      //  expose the internal data

        //  print function implementation
        void print(std::string separator = "\t", std::ostream& os = std::cout) const
        {
            if constexpr (is_MultiChannel<ElementT>::value)
            {
                if (size.size() == 1)
                {
                    for (std::size_t x = 0; x < size[0]; ++x)
                    {
                        os << at(x) << separator;
                    }
                    os << "\n";
                }
                else if (size.size() == 2)
                {
                    for (std::size_t y = 0; y < size[1]; ++y)
                    {
                        for (std::size_t x = 0; x < size[0]; ++x)
                        {
                            os << at(x, y) << separator;
                        }
                        os << "\n";
                    }
                    os << "\n";
                }
                else if (size.size() == 3)
                {
                    for (std::size_t z = 0; z < size[2]; ++z)
                    {
                        for (std::size_t y = 0; y < size[1]; ++y)
                        {
                            for (std::size_t x = 0; x < size[0]; ++x)
                            {
                                os << at(x, y, z) << separator;
                            }
                            os << "\n";
                        }
                        os << "\n";
                    }
                    os << "\n";
                }
                else if (size.size() == 4)
                {
                    for (std::size_t a = 0; a < size[3]; ++a)
                    {
                        os << "group = " << a << "\n";
                        for (std::size_t z = 0; z < size[2]; ++z)
                        {
                            for (std::size_t y = 0; y < size[1]; ++y)
                            {
                                for (std::size_t x = 0; x < size[0]; ++x)
                                {
                                    os << at(x, y, z, a) << separator;
                                }
                                os << "\n";
                            }
                            os << "\n";
                        }
                        os << "\n";
                    }
                    os << "\n";
                }
            }
            else
            {
                if (size.size() == 1)
                {
                    for (std::size_t x = 0; x < size[0]; ++x)
                    {
                        //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                        os << +at(x) << separator;
                    }
                    os << "\n";
                }
                else if (size.size() == 2)
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
                }
                else if (size.size() == 3)
                {
                    for (std::size_t z = 0; z < size[2]; ++z)
                    {
                        for (std::size_t y = 0; y < size[1]; ++y)
                        {
                            for (std::size_t x = 0; x < size[0]; ++x)
                            {
                                //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                                os << +at(x, y, z) << separator;
                            }
                            os << "\n";
                        }
                        os << "\n";
                    }
                    os << "\n";
                }
                else if (size.size() == 4)
                {
                    for (std::size_t a = 0; a < size[3]; ++a)
                    {
                        os << "group = " << a << "\n";
                        for (std::size_t z = 0; z < size[2]; ++z)
                        {
                            for (std::size_t y = 0; y < size[1]; ++y)
                            {
                                for (std::size_t x = 0; x < size[0]; ++x)
                                {
                                    //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                                    os << +at(x, y, z, a) << separator;
                                }
                                os << "\n";
                            }
                            os << "\n";
                        }
                        os << "\n";
                    }
                    os << "\n";
                }
            }
        }

        //  Enable this function if ElementT = RGB or RGB_DOUBLE or HSV
        void print(std::string separator = "\t", std::ostream& os = std::cout) const
        requires(std::same_as<ElementT, RGB> or std::same_as<ElementT, RGB_DOUBLE> or std::same_as<ElementT, HSV> or is_MultiChannel<ElementT>::value)
        {
            if (size.size() == 1)
            {
                for (std::size_t x = 0; x < size[0]; ++x)
                {
                    os << "( ";
                    for (std::size_t channel_index = 0; channel_index < 3; ++channel_index)
                    {
                        //  Ref: https://isocpp.org/wiki/faq/input-output#print-char-or-ptr-as-number
                        os << +at(x).channels[channel_index] << separator;
                    }
                    os << ")" << separator;
                }
                os << "\n";
            }
            else if (size.size() == 2)
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
            }
            return;
        }

        Image<ElementT>& setAllValue(const ElementT input)
        {
            std::fill(std::ranges::begin(image_data), std::ranges::end(image_data), input);
            return *this;
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

        // calculateIndex template function implementation
        template<std::same_as<std::size_t>... Args>
        constexpr auto calculateIndex(const Args... indices) const
        {
            std::size_t index = 0;
            std::size_t stride = 1;
            std::size_t i = 0;
            auto update_index = [&](std::size_t idx) {
                index += idx * stride;
                stride *= size[i++];
                };
            (update_index(indices), ...);
            return index;
        }

        template<typename... Args>
        void checkBoundary(const Args... indexInput) const
        {
            constexpr std::size_t n = sizeof...(Args);
            if(n != size.size())
            {
                throw std::runtime_error("Dimensionality mismatched!");
            }
            std::size_t parameter_pack_index = 0;
            auto function = [&](auto index) {
                if (index >= size[parameter_pack_index])
                    throw std::out_of_range("Given index out of range!");
                parameter_pack_index = parameter_pack_index + 1;
            };

            (function(indexInput), ...);
        }

        //  checkBoundaryTuple template function implementation
        template<class TupleT>
        requires(TinyDIP::is_tuple<TupleT>::value)
        constexpr bool checkBoundaryTuple(const TupleT location)
        {
            constexpr std::size_t n = std::tuple_size<TupleT>{};
            if(n != size.size())
            {
                throw std::runtime_error("Dimensionality mismatched!");
            }
            std::size_t parameter_pack_index = 0;
            auto function = [&](auto index) {
                if (std::cmp_greater_equal(index, size[parameter_pack_index]))
                    return false;
                parameter_pack_index = parameter_pack_index + 1;
                return true;
            };
            return std::apply([&](auto&&... args) { return ((function(args))&& ...);}, location);
        }

        //  tuple_location_to_index template function implementation
        template<class TupleT>
        requires(TinyDIP::is_tuple<TupleT>::value)
        constexpr std::size_t tuple_location_to_index(TupleT location)
        {
            std::size_t i = 0;
            std::size_t stride = 1;
            std::size_t position = 0;
            auto update_position = [&](auto index) {
                    position += index * stride;
                    stride *= size[i++];
                };
            std::apply([&](auto&&... args) {((update_position(args)), ...);}, location);
            return position;
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

    template <typename T>
    struct is_Image : std::false_type {};

    template <typename T>
    struct is_Image<Image<T>> : std::true_type {};
}


#endif

