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
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <numeric>
#include <print>
#include <ranges>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
#include <version>
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
    //  is_streamable concept
    //  Concept to check if a type supports the << operator with std::ostream
    template<typename T>
    concept is_streamable = requires(std::ostream & os, const T & val)
    {
        { os << val } -> std::same_as<std::ostream&>;
    };

    //  Polyfill concept for std::formattable to support compilers lacking complete C++23 features
#if defined(__cpp_lib_format) && __cpp_lib_format >= 202207L
    // Use standard std::formattable for modern compilers in C++23 mode
    template<typename T, typename CharT = char>
    concept is_formattable_compat = std::formattable<T, CharT>;
#else
    // Fallback for older compilers or C++20 mode lacking std::formattable
    template<typename T, typename CharT = char>
    concept is_formattable_compat = requires(std::formatter<std::remove_cvref_t<T>, CharT> f, std::basic_format_parse_context<CharT> pc)
    {
        f.parse(pc);
    };
#endif

    template <typename ElementT>
    class Image
    {
    public:
        Image() = default;

        template<std::same_as<std::size_t>... Sizes>
        Image(Sizes... sizes): size{sizes...}, image_data((1 * ... * sizes))
        {
        }

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
            if (std::ranges::empty(sizes))
            {
                throw std::runtime_error("Image size vector is empty!");
            }
            size.resize(std::ranges::size(sizes));
            std::transform(std::ranges::cbegin(sizes), std::ranges::cend(sizes), std::ranges::begin(size), [&](auto&& element) { return element; });
            image_data.resize(
                std::reduce(
                    std::ranges::cbegin(sizes),
                    std::ranges::cend(sizes),
                    std::size_t{1},
                    std::multiplies<>()
                )
            );
        }

        //  Image constructor
        #ifdef __cpp_lib_containers_ranges
            template<std::ranges::input_range Range,
                     std::same_as<std::size_t>... Sizes>
            requires(std::convertible_to<std::ranges::range_value_t<Range>, ElementT>)
            Image(Range&& input, Sizes... sizes):
                size{sizes...}, image_data(std::from_range, std::forward<Range>(input))
            {
                if (std::ranges::size(image_data) != (1 * ... * sizes)) 
                {
                    throw std::runtime_error("Image data input and the given size are mismatched!");
                }
            }
        #else
            template<std::ranges::input_range Range,
                     std::same_as<std::size_t>... Sizes>
            requires(std::convertible_to<std::ranges::range_value_t<Range>, ElementT>)
            Image(const Range&& input, Sizes... sizes):
                size{sizes...}, image_data(std::ranges::begin(input), std::ranges::end(input))
            {
                if (std::ranges::size(image_data) != (1 * ... * sizes)) 
                {
                    throw std::runtime_error("Image data input and the given size are mismatched!");
                }
            }
        #endif

        //  Image constructor
        #ifdef __cpp_lib_containers_ranges
            template<std::ranges::input_range Range, std::same_as<std::size_t>... Sizes>
            requires(std::convertible_to<std::ranges::range_value_t<Range>, ElementT>)
            Image(const Range& input, Sizes... sizes) :
                size{ sizes... }
            {
                if (std::ranges::empty(input))
                {
                    throw std::runtime_error("Input vector is empty!");
                }
                image_data = std::vector(std::from_range, input);
                if (std::ranges::size(image_data) != (1 * ... * sizes)) 
                {
                    throw std::runtime_error("Image data input and the given size are mismatched!");
                }
            }
        #else
            template<std::ranges::input_range Range, std::same_as<std::size_t>... Sizes>
            requires(std::convertible_to<std::ranges::range_value_t<Range>, ElementT>)
            Image(const Range& input, Sizes... sizes):
                size{sizes...}
            {
                if (std::ranges::empty(input))
                {
                    throw std::runtime_error("Input vector is empty!");
                }
                image_data = std::vector(std::ranges::begin(input), std::ranges::end(input));
                if (std::ranges::size(image_data) != (1 * ... * sizes)) 
                {
                    throw std::runtime_error("Image data input and the given size are mismatched!");
                }
            }
        #endif

        //  Image constructor
        template<std::ranges::input_range Range, std::ranges::input_range Sizes>
        requires(std::convertible_to<std::ranges::range_value_t<Range>, ElementT>) and
                ((std::same_as<std::ranges::range_value_t<Sizes>, std::size_t>) or
                 (std::same_as<std::ranges::range_value_t<Sizes>, int>))
        Image(const Range& input, const Sizes& sizes)
        {
            if (std::ranges::empty(input))
            {
                throw std::runtime_error("Input vector is empty!");
            }
            size.resize(std::ranges::size(sizes));
            std::transform(std::ranges::cbegin(sizes), std::ranges::cend(sizes), std::ranges::begin(size), [&](auto&& element) { return static_cast<std::size_t>(element); });
            image_data = std::vector(std::ranges::cbegin(input), std::ranges::cend(input));
            auto count = std::reduce(std::ranges::cbegin(sizes), std::ranges::cend(sizes), std::size_t{1}, std::multiplies());
            if (std::ranges::size(image_data) != count) 
            {
                throw std::runtime_error("Image data input and the given size are mismatched!");
            }
        }

        Image(const std::vector<std::vector<ElementT>>& input)
        {
            if (std::ranges::empty(input))
            {
                throw std::runtime_error("Input vector is empty!");
            }
            size.reserve(2);
            size.emplace_back(std::ranges::size(input[0]));
            size.emplace_back(std::ranges::size(input));
            for (auto& rows : input)
            {
                image_data.insert(std::ranges::end(image_data), std::ranges::begin(rows), std::ranges::end(rows));    //  flatten
            }
            return;
        }

        //  at template function implementation
        //  Consolidated variadic template using C++20 std::integral to accept any integer or size_t natively.
        //  decltype(auto) guarantees perfect forwarding of vector reference proxies (e.g., std::vector<bool>::reference).
        template<std::integral... Args>
        constexpr decltype(auto) at(const Args... indexInput)
        {
            checkBoundary(static_cast<std::size_t>(indexInput)...);
            return at_without_boundary_check(static_cast<std::size_t>(indexInput)...);
        }

        //  at template function implementation
        template<std::integral... Args>
        constexpr decltype(auto) at(const Args... indexInput) const
        {
            checkBoundary(static_cast<std::size_t>(indexInput)...);
            return at_without_boundary_check(static_cast<std::size_t>(indexInput)...);
        }

        //  at template function implementation (std::ranges::input_range case)
        template<std::ranges::input_range Indices>
        requires(std::integral<std::ranges::range_value_t<Indices>>)
        constexpr decltype(auto) at(const Indices& indexInput) const
        {
            std::size_t i = 0;
            // Iterate using ranges to universally support both continuous (std::vector) and discrete (std::list) ranges
            for (const auto& idx : indexInput)
            {
                // Boundary check MUST be >= to prevent array overflow segmentation faults
                if (static_cast<std::size_t>(idx) >= size[i])
                {
                    throw std::out_of_range("Given index out of range!");
                }
                ++i;
            }
            return at_without_boundary_check(indexInput);
        }

        //  at template function implementation (std::ranges::input_range case)
        template<std::ranges::input_range Indices>
        requires(std::integral<std::ranges::range_value_t<Indices>>)
        constexpr decltype(auto) at(const Indices& indexInput)
        {
            std::size_t i = 0;
            for (const auto& idx : indexInput)
            {
                if (static_cast<std::size_t>(idx) >= size[i])
                {
                    throw std::out_of_range("Given index out of range!");
                }
                ++i;
            }
            return at_without_boundary_check(indexInput);
        }

        //  at_without_boundary_check template function implementation
        template<std::integral... Args>
        constexpr decltype(auto) at_without_boundary_check(const Args... indexInput)
        {
            constexpr std::size_t n = sizeof...(Args);
            if (n != std::ranges::size(size)) 
            {
                throw std::runtime_error("Dimensionality mismatched!");
            }
            return image_data[calculateIndex(static_cast<std::size_t>(indexInput)...)];
        }

        //  at_without_boundary_check template function implementation
        template<std::integral... Args>
        constexpr decltype(auto) at_without_boundary_check(const Args... indexInput) const
        {
            constexpr std::size_t n = sizeof...(Args);
            if (n != std::ranges::size(size)) 
            {
                throw std::runtime_error("Dimensionality mismatched!");
            }
            return image_data[calculateIndex(static_cast<std::size_t>(indexInput)...)];
        }

        //  at_without_boundary_check template function implementation (std::ranges::input_range case)
        template<std::ranges::input_range Indices>
        requires(std::integral<std::ranges::range_value_t<Indices>>)
        constexpr decltype(auto) at_without_boundary_check(const Indices& indexInput) const
        {
            // Direct injection to calculateIndex removes the severe dynamic memory allocation overhead
            return image_data[calculateIndex(indexInput)];
        }

        //  at_without_boundary_check template function implementation (std::ranges::input_range case)
        template<std::ranges::input_range Indices>
        requires(std::integral<std::ranges::range_value_t<Indices>>)
        constexpr decltype(auto) at_without_boundary_check(const Indices& indexInput)
        {
            return image_data[calculateIndex(indexInput)];
        }

        //  get function implementation
        constexpr ElementT get(std::size_t index) const noexcept
        {
            return static_cast<ElementT>(image_data[index]);
        }

        //  set function implementation
        //  decltype(auto) ensures that modifying the return element safely propagates 
        //  into the proxy objects associated with bitwise structures (e.g., vector<bool>)
        constexpr decltype(auto) set(const std::size_t index, std::ostream& os = std::cout)
        {
            if (index >= count())
            {
                if constexpr (is_formattable_compat<std::size_t, char>)
                {
                    std::print(os, "index = {}, count = {}\n", index, count());
                }
                else
                {
                    os << "index = " << index << ", count = " << count() << '\n';
                }
                throw std::out_of_range("Given index out of range!");
            }
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
        constexpr Image<TargetT> cast() const
        {
            std::vector<TargetT> output_data;
            output_data.resize(std::ranges::size(image_data));
            std::transform(
                std::ranges::cbegin(image_data),
                std::ranges::cend(image_data),
                std::ranges::begin(output_data),
                [&](const auto& input){ return static_cast<TargetT>(input); }
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
        constexpr std::size_t count(ExPo&& execution_policy) const
        {
            if (std::ranges::empty(size)) return 0;
            return std::reduce(std::forward<ExPo>(execution_policy), std::ranges::cbegin(size), std::ranges::cend(size), std::size_t{ 1 }, std::multiplies());
        }
  
        constexpr std::size_t getDimensionality() const noexcept
        {
            return std::ranges::size(size);
        }

        constexpr std::size_t getWidth() const noexcept
        {
            return size[0];
        }

        constexpr std::size_t getHeight() const noexcept
        {
            return (getDimensionality() > 1) ? size[1] : 0;
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

        /**
         * print function implementation
         * @brief Prints the image content to an output stream.
         * This function is generic and supports printing N-dimensional images.
         * @param separator The separator to use between elements.
         * @param os The output stream to write to.
         */
        void print(std::string_view separator = "\t", std::ostream& os = std::cout) const
        {
            if (getDimensionality() == 0)
            {
                return;
            }

            // A generic lambda perfectly accepts primitive copies, proxy instances, or direct references safely
            auto element_printer = [&](const auto& value) 
            {
                if constexpr (is_MultiChannel<ElementT>::value)
                {
                    os << "( ";
                    for (std::size_t i = 0; i < std::size(value.channels); ++i) 
                    {
                        if constexpr (is_formattable_compat<decltype(+value.channels[i]), char>)
                        {
                            std::print(os, "{}", +value.channels[i]);
                        }
                        else
                        {
                            os << +value.channels[i];
                        }
                        
                        if (i != std::size(value.channels) - 1)
                        {
                            os << " ";
                        }
                    }
                    os << ") ";
                }
                else if constexpr (is_streamable<ElementT> && !std::is_fundamental_v<ElementT>)
                {
                    // For non-fundamental types (like custom structs), try modern formatting first, 
                    // otherwise fall back to standard stream insertion safely.
                    if constexpr (is_formattable_compat<ElementT, char>)
                    {
                        std::print(os, "{}", value);
                    }
                    else
                    {
                        os << value;
                    }
                }
                else
                {
                    // Use unary '+' to ensure char types are printed as numbers
                    if constexpr (is_formattable_compat<decltype(+value), char>)
                    {
                        std::print(os, "{}", +value);
                    }
                    else
                    {
                        os << +value;
                    }
                }
            };

            std::vector<std::size_t> indices(getDimensionality(), 0);
            print_recursive_helper(indices, getDimensionality() - 1, element_printer, separator, os);
        }

        Image<ElementT>& setAllValue(const ElementT& input)
        {
            std::fill(std::ranges::begin(image_data), std::ranges::end(image_data), input);
            return *this;
        }

        //  setAllValue template function implementation (with Execution Policy)
        template<class ExecutionPolicy>
        requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
        Image<ElementT>& setAllValue(ExecutionPolicy&& execution_policy, const ElementT& input)
        {
            std::fill(std::forward<ExecutionPolicy>(execution_policy), std::ranges::begin(image_data), std::ranges::end(image_data), input);
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

        // --- Start of C++23 operator[] ---
        template<std::integral... Args>
        constexpr decltype(auto) operator[](Args... indices)&
        {
            return at_without_boundary_check(static_cast<std::size_t>(indices)...);
        }

        template<std::integral... Args>
        constexpr decltype(auto) operator[](Args... indices) const&
        {
            return at_without_boundary_check(static_cast<std::size_t>(indices)...);
        }
        // --- End of C++23 operator[] ---

        // Nested class for the custom iterator
        // Refactored to utilize std::vector iterators directly to strictly prevent proxy errors on Image<bool>
        class pixel_iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = std::tuple<typename std::vector<ElementT>::reference, std::size_t, std::size_t>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type*;
            using reference = value_type;

            pixel_iterator() = default;

            pixel_iterator(typename std::vector<ElementT>::iterator iter, std::size_t x, std::size_t y, std::size_t width)
                : current_iter(iter), px(x), py(y), image_width(width)
            {
            }

            reference operator*() const
            {
                return {*current_iter, px, py};
            }

            pixel_iterator& operator++()
            {
                ++current_iter;
                ++px;
                if (px == image_width)
                {
                    px = 0;
                    ++py;
                }
                return *this;
            }
            
            pixel_iterator operator++(int)
            {
                pixel_iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            bool operator==(const pixel_iterator& other) const = default;

        private:
            typename std::vector<ElementT>::iterator current_iter{};
            std::size_t px = 0;
            std::size_t py = 0;
            std::size_t image_width = 0;
        };

        // Nested class for the custom const iterator
        class const_pixel_iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = std::tuple<typename std::vector<ElementT>::const_reference, std::size_t, std::size_t>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type*;
            using reference = value_type;

            const_pixel_iterator() = default;

            const_pixel_iterator(typename std::vector<ElementT>::const_iterator iter, std::size_t x, std::size_t y, std::size_t width)
                : current_iter(iter), px(x), py(y), image_width(width)
            {
            }

            reference operator*() const
            {
                return {*current_iter, px, py};
            }

            const_pixel_iterator& operator++()
            {
                ++current_iter;
                ++px;
                if (px == image_width)
                {
                    px = 0;
                    ++py;
                }
                return *this;
            }

            const_pixel_iterator operator++(int)
            {
                const_pixel_iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            bool operator==(const const_pixel_iterator& other) const = default;

        private:
            typename std::vector<ElementT>::const_iterator current_iter{};
            std::size_t px = 0;
            std::size_t py = 0;
            std::size_t image_width = 0;
        };

        // Nested class for the range proxy object
        class pixel_proxy
        {
        public:
            pixel_proxy(Image<ElementT>& image) : img(image)
            {
                if (img.getDimensionality() != 2)
                {
                    throw std::logic_error("pixels_with_coordinates is only supported for 2D images.");
                }
            }

            [[nodiscard]] auto begin()
            {
                return pixel_iterator(std::ranges::begin(img.image_data), 0, 0, img.getWidth());
            }

            [[nodiscard]] auto end()
            {
                return pixel_iterator(std::ranges::end(img.image_data), 0, img.getHeight(), img.getWidth());
            }

        private:
            Image<ElementT>& img;
        };
        
        // Nested class for the const range proxy object
        class const_pixel_proxy
        {
        public:
            const_pixel_proxy(const Image<ElementT>& image) : img(image)
            {
                if (img.getDimensionality() != 2)
                {
                    throw std::logic_error("pixels_with_coordinates is only supported for 2D images.");
                }
            }

            [[nodiscard]] auto begin() const
            {
                return const_pixel_iterator(std::ranges::cbegin(img.image_data), 0, 0, img.getWidth());
            }

            [[nodiscard]] auto end() const
            {
                return const_pixel_iterator(std::ranges::cend(img.image_data), 0, img.getHeight(), img.getWidth());
            }

        private:
            const Image<ElementT>& img;
        };

        /**
         * @brief Returns a range-like object for iterating over pixels with their 2D coordinates.
         * @details This is intended for use in range-based for loops with structured bindings:
         * for (auto& [value, x, y] : image.pixels_with_coordinates()) { ... }
         * @note This function is only valid for 2D images and will throw an exception otherwise.
         * @return A proxy object with begin() and end() methods.
         */
        [[nodiscard]] auto pixels_with_coordinates()
        {
            return pixel_proxy(*this);
        }

        /**
         * @brief Returns a const range-like object for iterating over pixels with their 2D coordinates.
         * @details This is intended for use in range-based for loops with structured bindings:
         * for (const auto& [value, x, y] : image.pixels_with_coordinates()) { ... }
         * @note This function is only valid for 2D images and will throw an exception otherwise.
         * @return A proxy object with begin() and end() methods.
         */
        [[nodiscard]] auto pixels_with_coordinates() const
        {
            return const_pixel_proxy(*this);
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


        /**
         * print_recursive_helper template function implementation
         * @brief Recursively iterates through dimensions to print the image.
         *
         * @tparam PrintFunc A callable type for printing a single element.
         * @param indices A vector to hold the current N-dimensional index.
         * @param current_dim The current dimension being iterated over.
         * @param print_element The function that prints a single element.
         * @param separator The separator string.
         * @param os The output stream.
         */
        template <typename PrintFunc>
        void print_recursive_helper(
            std::vector<std::size_t>& indices,
            std::size_t current_dim,
            const PrintFunc& print_element,
            std::string_view separator,
            std::ostream& os) const
        {
            if (current_dim == 0)                                   // Base case: The innermost dimension
            {
                for (std::size_t i = 0; i < getSize(current_dim); ++i)
                {
                    indices[current_dim] = i;
                    print_element(at_without_boundary_check(indices));
                    os << separator;
                }
            }
            else                                                    // Recursive step: Outer dimensions
            {
                for (std::size_t i = 0; i < getSize(current_dim); ++i)
                {
                    indices[current_dim] = i;
                    print_recursive_helper(indices, current_dim - 1, print_element, separator, os);
                    // After a full "row" of the lower dimension, add a newline.
                    os << '\n';
                }
                // Add an extra newline to separate higher-dimensional "planes".
                 os << '\n';
            }
        }

        // calculateIndex template function implementation
        template<std::integral... Args>
        constexpr auto calculateIndex(const Args... indices) const
        {
            std::size_t index = 0;
            std::size_t stride = 1;
            std::size_t i = 0;
            auto update_index = [&](std::size_t idx) 
            {
                index += idx * stride;
                stride *= size[i++];
            };
            (update_index(static_cast<std::size_t>(indices)), ...);
            return index;
        }

        //  calculateIndex template function implementation
        template<std::ranges::input_range Indices>
        requires (std::integral<std::ranges::range_value_t<Indices>>)
        constexpr std::size_t calculateIndex(const Indices& indices) const
        {
            std::size_t index = 0;
            std::size_t stride = 1;
            std::size_t i = 0;
            for (const auto& idx : indices) 
            {
                index += static_cast<std::size_t>(idx) * stride;
                stride *= size[i++];
            }
            return index;
        }

        template<std::integral... Args>
        void checkBoundary(const Args... indexInput) const
        {
            constexpr std::size_t n = sizeof...(Args);
            if(n != std::ranges::size(size))
            {
                throw std::runtime_error("Dimensionality mismatched!");
            }
            std::size_t parameter_pack_index = 0;
            auto function = [&](auto index) 
            {
                if (static_cast<std::size_t>(index) >= size[parameter_pack_index])
                {
                    throw std::out_of_range("Given index out of range!");
                }
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
            if(n != std::ranges::size(size))
            {
                throw std::runtime_error("Dimensionality mismatched!");
            }
            std::size_t parameter_pack_index = 0;
            auto function = [&](auto index) 
            {
                if (std::cmp_greater_equal(index, size[parameter_pack_index]))
                {
                    return false;
                }
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
            auto update_position = [&](auto index) 
            {
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

