/* Developed by Jimmy Hu */
/* Refactored for CLI Application capability */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  main.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o main -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./main && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall main.cpp -o main 


//#define USE_BOOST_ITERATOR
//#define USE_BOOST_SERIALIZATION

//  Standard Library Headers
#include <algorithm>
#include <any>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <charconv>
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <valarray>
#include <vector>

//  Local Headers
#include "basic_functions.h"
#include "image_io.h"
#include "image_operations.h"
#include "timer.h"


//#define BOOST_TEST_DYN_LINK

//#define BOOST_TEST_MODULE image_elementwise_tests

#ifdef BOOST_TEST_MODULE
#include <boost/test/included/unit_test.hpp>


#ifdef BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#else
#include <boost/test/included/unit_test.hpp>
#endif // BOOST_TEST_DYN_LINK

#include <boost/mpl/list.hpp>
#include <boost/mpl/vector.hpp>
#include <tao/tuple/tuple.hpp>

typedef boost::mpl::list<
    byte, char, int, short, long, long long int,
    unsigned int, unsigned short int, unsigned long int, unsigned long long int,
    float, double, long double> test_types;

//  [TODO] Avoid code duplication (https://codereview.stackexchange.com/a/267709/231235)
BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_add_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T increment = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test += TinyDIP::Image<T>(size_x, size_y, increment);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal + increment));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_add_test_zero_dimensions, T, test_types)
{
    std::size_t size_x = 0;                         //  Test images with both of the dimensions having size zero.
    std::size_t size_y = 0;                         //  Test images with both of the dimensions having size zero.
    T initVal = 10;
    T increment = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test += TinyDIP::Image<T>(size_x, size_y, increment);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal + increment));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_add_test_large_dimensions, T, test_types)
{
    std::size_t size_x = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    std::size_t size_y = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    T initVal = 10;
    T increment = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test += TinyDIP::Image<T>(size_x, size_y, increment);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal + increment));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_minus_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T difference = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test -= TinyDIP::Image<T>(size_x, size_y, difference);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal - difference));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_minus_test_zero_dimensions, T, test_types)
{
    std::size_t size_x = 0;                         //  Test images with both of the dimensions having size zero.
    std::size_t size_y = 0;                         //  Test images with both of the dimensions having size zero.
    T initVal = 10;
    T difference = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test -= TinyDIP::Image<T>(size_x, size_y, difference);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal - difference));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_minus_test_large_dimensions, T, test_types)
{
    std::size_t size_x = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    std::size_t size_y = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    T initVal = 10;
    T difference = 1;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test -= TinyDIP::Image<T>(size_x, size_y, difference);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal - difference));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_multiplies_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T multiplier = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test *= TinyDIP::Image<T>(size_x, size_y, multiplier);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal * multiplier));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_multiplies_test_zero_dimensions, T, test_types)
{
    std::size_t size_x = 0;                         //  Test images with both of the dimensions having size zero.
    std::size_t size_y = 0;                         //  Test images with both of the dimensions having size zero.
    T initVal = 10;
    T multiplier = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test *= TinyDIP::Image<T>(size_x, size_y, multiplier);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal * multiplier));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_multiplies_test_large_dimensions, T, test_types)
{
    std::size_t size_x = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    std::size_t size_y = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    T initVal = 10;
    T multiplier = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test *= TinyDIP::Image<T>(size_x, size_y, multiplier);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal * multiplier));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_divides_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T divider = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test /= TinyDIP::Image<T>(size_x, size_y, divider);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal / divider));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_divides_test_zero_dimensions, T, test_types)
{
    std::size_t size_x = 0;                         //  Test images with both of the dimensions having size zero.
    std::size_t size_y = 0;                         //  Test images with both of the dimensions having size zero.
    T initVal = 10;
    T divider = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test /= TinyDIP::Image<T>(size_x, size_y, divider);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal / divider));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_divides_test_large_dimensions, T, test_types)
{
    std::size_t size_x = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    std::size_t size_y = 18446744073709551615;       //  Test images with very large dimensions (std::numeric_limits<std::size_t>::max()).
    T initVal = 10;
    T divider = 2;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test /= TinyDIP::Image<T>(size_x, size_y, divider);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal / divider));
}
/*
BOOST_AUTO_TEST_CASE_TEMPLATE(image_elementwise_divides_zero_test, T, test_types)
{
    std::size_t size_x = 10;
    std::size_t size_y = 10;
    T initVal = 10;
    T divider = 0;
    auto test = TinyDIP::Image<T>(size_x, size_y, initVal);
    test /= TinyDIP::Image<T>(size_x, size_y, divider);
    BOOST_TEST(test == TinyDIP::Image<T>(size_x, size_y, initVal / divider));       //  dividing by zero test
}
*/
#endif

void difference_and_enhancement(std::string input_path1, std::string input_path2, double enhancement_times)
{
    if (input_path1.empty())
    {
        std::cerr << "Input path is empty!";
    }

    std::filesystem::path input1 = input_path1;
    std::filesystem::path input2 = input_path2;
    
}

#ifndef BOOST_TEST_MODULE
void addLeadingZeros(std::string input_path, std::string output_path);

//  parse_arg template function implementation
//  Helper for converting string to numeric types safely
template <typename T>
T parse_arg(const std::string_view sv)
{
    T result{};
    if constexpr (std::is_arithmetic_v<T>)
    {
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + std::ranges::size(sv), result);
        if (ec != std::errc())
        {
            throw std::invalid_argument(std::string("Error parsing argument: ") + std::string(sv));
        }
    }
    else
    {
        //  Fallback for non-arithmetic types (unlikely to be used with this function in current context)
        //  This path forces allocation, but is rarely hit for numeric parsing
        std::string temp(sv);
        std::stringstream ss(temp);
        if (!(ss >> result))
        {
            throw std::invalid_argument(std::string("Error parsing argument: ") + temp);
        }
    }
    return result;
}

void print(auto comment, auto const& seq, char term = ' ') {
    for (std::cout << comment << '\n'; auto const& elem : seq)
        std::cout << elem << term;
    std::cout << '\n';
}


auto myHighLightRegion_parameters(const std::size_t index = 0)
{
    std::vector<std::tuple<
        std::string,    //  filenames
        std::size_t,    //  start_index
        std::size_t,    //  end_index
        std::size_t,    //  startx
        std::size_t,    //  endx
        std::size_t,    //  starty
        std::size_t,    //  endy
        std::string     //  output_location
        >> collection;
}

// ------------------------------------------------------------------------------------
//  Iterable Container Detection Traits
// ------------------------------------------------------------------------------------

//  is_vector template struct implementation
template <typename T> struct is_vector : std::false_type {};
template <typename T, typename A> struct is_vector<std::vector<T, A>> : std::true_type {};
template <typename T> inline constexpr bool is_vector_v = is_vector<T>::value;

template <typename T> struct is_deque : std::false_type {};
template <typename T, typename A> struct is_deque<std::deque<T, A>> : std::true_type {};
template <typename T> inline constexpr bool is_deque_v = is_deque<T>::value;

template <typename T> struct is_list : std::false_type {};
template <typename T, typename A> struct is_list<std::list<T, A>> : std::true_type {};
template <typename T> inline constexpr bool is_list_v = is_list<T>::value;

template <typename T> struct is_std_array : std::false_type {};
template <typename T, std::size_t N> struct is_std_array<std::array<T, N>> : std::true_type {};
template <typename T> inline constexpr bool is_std_array_v = is_std_array<T>::value;

//  match_any_type template function implementation
template <typename TupleT, class FunT>
constexpr bool match_any_type(FunT&& func)
{
    return [&]<template <typename...> class TupleLike, typename... Ts>(std::type_identity<TupleLike<Ts...>>)
    {
        return (... || std::forward<FunT>(func).template operator()<Ts>());
    }(std::type_identity<TupleT>{});
}

// ------------------------------------------------------------------------------------
//  Advanced Metaprogramming Type Generation Registries
// ------------------------------------------------------------------------------------

// Core Fundamental Types
using core_numeric_types = std::tuple<
    bool, char, signed char, unsigned char,
    short, unsigned short, int, unsigned int,
    long, unsigned long, long long, unsigned long long,
    std::int8_t, std::int16_t, std::int32_t, std::int64_t,
    std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t,
    float, double, long double, std::size_t, std::ptrdiff_t
>;

using core_floating_point_types = std::tuple<float, double, long double>;

// Metaprogramming Mapping Tools
template <template <typename...> class Wrapper, typename Tuple>
struct tuple_map;

template <template <typename...> class Wrapper, typename... Ts>
struct tuple_map<Wrapper, std::tuple<Ts...>>
{
    using type = std::tuple<Wrapper<Ts>...>;
};

template <template <typename...> class Wrapper, typename Tuple>
using tuple_map_t = typename tuple_map<Wrapper, Tuple>::type;

template <typename... Tuples>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Tuples>()...));

// -----------------------------------------------------------------------------
// Advanced NTTP Metaprogramming: Dynamic Array Size Generation
// -----------------------------------------------------------------------------

// Generate an index sequence representing [Min, Max]
template <std::size_t Min, std::size_t Max, std::size_t... Is>
constexpr auto make_range_sequence_impl(std::index_sequence<Is...>)
{
    return std::index_sequence<(Min + Is)...>{};
}

template <std::size_t Min, std::size_t Max>
requires (Min <= Max)
using make_range_sequence = decltype(make_range_sequence_impl<Min, Max>(std::make_index_sequence<Max - Min + 1>{}));

// Map an entire Tuple of types to std::array<T, N> for a fixed size N
template <typename Tuple, std::size_t N>
struct make_array_tuple;

template <typename... Ts, std::size_t N>
struct make_array_tuple<std::tuple<Ts...>, N>
{
    using type = std::tuple<std::array<Ts, N>...>;
};

// Perform a cartesian product: Concatenate make_array_tuple for all Ns in the sequence
template <typename Tuple, typename IndexSeq>
struct generate_arrays_impl;

template <typename Tuple, std::size_t... Ns>
struct generate_arrays_impl<Tuple, std::index_sequence<Ns...>>
{
    // Expands to tuple_cat_t< std::tuple<std::array<Ts, 3>...>, std::tuple<std::array<Ts, 4>...>, ... >
    using type = tuple_cat_t<typename make_array_tuple<Tuple, Ns>::type...>;
};

// User-friendly alias
template <typename Tuple, std::size_t Min, std::size_t Max>
using generate_arrays_t = typename generate_arrays_impl<Tuple, make_range_sequence<Min, Max>>::type;

// Helper aliases to bridge TinyDIP's Non-Type Template Parameters (NTTP) for tuple mapping
template <typename T>
using multichannel_t = TinyDIP::MultiChannel<T>;

template <typename T>
using image_t = TinyDIP::Image<T>;

// Exhaustive Derived Type Auto-Generation
using all_multichannel_types = tuple_map_t<multichannel_t, core_numeric_types>;
using all_complex_types = tuple_map_t<std::complex, core_floating_point_types>;
using all_vector_types = tuple_map_t<std::vector, core_numeric_types>;
using all_deque_types = tuple_map_t<std::deque, core_numeric_types>;
using all_list_types = tuple_map_t<std::list, core_numeric_types>;
using all_array_types = generate_arrays_t<core_numeric_types, 3, 4>;
using all_custom_scalar_types = std::tuple<TinyDIP::RGB, TinyDIP::RGB_DOUBLE, TinyDIP::HSV>;

// Master Scalar Tuple (Exhaustively includes ALL valid scalar and container output types)
using master_scalar_types = tuple_cat_t<
    core_numeric_types,
    all_custom_scalar_types,
    all_multichannel_types,
    all_complex_types,
    all_vector_types,
    all_deque_types,
    all_list_types,
    all_array_types
>;

// Master Image Tuple (Exhaustively includes ALL valid image structures)
using master_image_types = tuple_cat_t<
    tuple_map_t<image_t, core_numeric_types>,
    tuple_map_t<image_t, all_custom_scalar_types>,
    tuple_map_t<image_t, all_multichannel_types>
>;

// Master Data Tuple (Exhaustively includes ALL valid image structures AND containers)
using master_data_types = tuple_cat_t<
    master_image_types,
    all_vector_types,
    all_deque_types,
    all_list_types,
    all_array_types
>;

// Distinct tuple exclusively tailored for segregating complex formatting logic natively
using complex_scalar_types_for_printing = tuple_cat_t<
    all_custom_scalar_types,
    all_multichannel_types,
    all_complex_types,
    all_vector_types,
    all_deque_types,
    all_list_types,
    all_array_types
>;

//  get_type_name template function implementation
//  Generic compile-time helper to automatically extract exact human-readable string views for any type.
//  This utilizes compile-time SFINAE reflection over compiler signature macros.
template <typename T>
constexpr std::string_view get_type_name()
{
#if defined(__clang__)
    constexpr std::string_view name = __PRETTY_FUNCTION__;
    constexpr std::size_t start = name.find("T = ") + 4;
    constexpr std::size_t end = name.find_last_of(']');
    return name.substr(start, end - start);
#elif defined(__GNUC__)
    constexpr std::string_view name = __PRETTY_FUNCTION__;
    constexpr std::size_t start = name.find("with T = ") + 9;
    constexpr std::size_t semi_colon_pos = name.find(';', start);
    constexpr std::size_t end = (semi_colon_pos != std::string_view::npos) ? semi_colon_pos : name.find_last_of(']');
    return name.substr(start, end - start);
#elif defined(_MSC_VER)
    constexpr std::string_view name = __FUNCSIG__;
    constexpr std::size_t start = name.find("get_type_name<") + 14;
    constexpr std::size_t end = name.rfind(">(void)");
    return name.substr(start, end - start);
#else
    return "Unknown Type";
#endif
}

//  execute_type_action template function implementation
template <typename TargetT, typename TupleT, typename FallbackFun, std::size_t I = 0>
constexpr decltype(auto) execute_type_action(TupleT&& action_map, FallbackFun&& fallback)
{
    if constexpr (I < std::tuple_size_v<std::remove_cvref_t<TupleT>>)
    {
        using CurrentPair = std::tuple_element_t<I, std::remove_cvref_t<TupleT>>;
        if constexpr (std::is_same_v<TargetT, typename CurrentPair::type>)
        {
            return std::get<I>(std::forward<TupleT>(action_map)).action();
        }
        else
        {
            return execute_type_action<TargetT, TupleT, FallbackFun, I + 1>(
                std::forward<TupleT>(action_map), std::forward<FallbackFun>(fallback));
        }
    }
    else
    {
        return std::forward<FallbackFun>(fallback)();
    }
}

//  Workspace struct implementation
//  In-Memory Workspace for REPL session state
struct Workspace
{
    std::map<std::string, std::any> memory_store;

    template <typename T>
    void store(const std::string_view name, T&& item)
    {
        memory_store[std::string(name)] = std::forward<T>(item);
    }

    template <typename T>
    const T* retrieve(const std::string_view name) const
    {
        if (auto it = memory_store.find(std::string(name)); it != std::ranges::end(memory_store))
        {
            if (it->second.type() == typeid(T))
            {
                return std::any_cast<T>(&(it->second));
            }
        }
        return nullptr;
    }

    //  remove function implementation
    bool remove(const std::string_view name)
    {
        const std::string key = std::string(name);
        if (auto it = memory_store.find(key); it != std::ranges::end(memory_store))
        {
            memory_store.erase(it);
            return true;
        }
        return false;
    }

    //  rename function implementation
    bool rename(const std::string_view old_name, const std::string_view new_name)
    {
        const std::string old_key(old_name);
        if (auto it = memory_store.find(old_key); it != std::ranges::end(memory_store))
        {
            //  Use std::move to natively transfer ownership of the type-erased object with zero-copy
            memory_store[std::string(new_name)] = std::move(it->second);
            memory_store.erase(it);
            return true;
        }
        return false;
    }

    //  Clear all elements in the workspace memory store
    void clear()
    {
        memory_store.clear();
    }

    //  list_variables function implementation
    void list_variables(std::ostream& os) const
    {
        if (std::ranges::empty(memory_store))
        {
            os << "  (Workspace is empty)\n";
            return;
        }

        //  print_size lambda implementation
        // Generic lambda to cleanly format and print image dimensions
        auto print_size = [&os](const std::ranges::random_access_range auto& size_range)
        {
            auto it = std::ranges::begin(size_range);
            const auto end = std::ranges::end(size_range);
            if (it != end)
            {
                os << +(*it);
                ++it;
                for (; it != end; ++it)
                {
                    os << " x " << +(*it);
                }
            }
        };

        for (const auto& [name, value] : memory_store)
        {
            auto print_prefix = [&]<typename T>()
            {
                os << "  $" << std::left << std::setw(15) << name << " : [" << get_type_name<T>() << "]";
            };

            // Polymorphic lambda returning true if the image type matched
            auto try_print_image = [&]<typename T>() -> bool
            {
                if (value.type() == typeid(T))
                {
                    print_prefix.template operator()<T>();
                    os << ", size = ";
                    const auto* image_ptr = std::any_cast<T>(&value);
                    print_size(image_ptr->getSize());
                    return true;
                }
                return false;
            };

            // Polymorphic lambda returning true if the complex custom scalar type matched
            auto try_print_complex_scalar = [&]<typename T>() -> bool
            {
                if (value.type() == typeid(T))
                {
                    print_prefix.template operator()<T>();
                    if constexpr (is_vector_v<T> || is_deque_v<T> || is_list_v<T> || is_std_array_v<T>)
                    {
                        os << ", container value = {";
                        bool first = true;
                        const auto* container_ptr = std::any_cast<T>(&value);
                        for (const auto& elem : *container_ptr)
                        {
                            if (!first)
                            {
                                os << ", ";
                            }
                            os << +elem;
                            first = false;
                        }
                        os << "}";
                    }
                    else
                    {
                        os << ", scalar value = " << std::any_cast<T>(value);
                    }
                    return true;
                }
                return false;
            };

            if (match_any_type<master_image_types>(try_print_image))
            {
                // Handled successfully by try_print_image short-circuit logic
            }
            else if (match_any_type<complex_scalar_types_for_printing>(try_print_complex_scalar))
            {
                // Handled successfully by try_print_complex_scalar short-circuit logic
            }
            else
            {
                // Polymorphic lambda returning true if the numeric type matched
                auto try_print_numeric = [&]<typename T>() -> bool
                {
                    if (value.type() == typeid(T))
                    {
                        print_prefix.template operator()<T>();
                        if constexpr (sizeof(T) == 1 && std::is_integral_v<T>) // Safely print 8-bit integer types as numbers, not unprintable chars
                        {
                            os << ", scalar value = " << +std::any_cast<T>(value);
                        }
                        else
                        {
                            os << ", scalar value = " << std::any_cast<T>(value);
                        }
                        return true;
                    }
                    return false;
                };

                if (!match_any_type<core_numeric_types>(try_print_numeric))
                {
                    os << "  $" << std::left << std::setw(15) << name 
                       << " : [Type Hash: " << value.type().hash_code() << "] (Unsupported serialization type), type is " << value.type().name();
                }
            }
            os << '\n';
        }
    }
};

//  MetaImageIO struct implementation
//  Generic struct to deal with Workspace memory mapping and direct File I/O operations dynamically
struct MetaImageIO
{
public:
    struct Loader
    {
        template <typename ImageType = TinyDIP::Image<TinyDIP::RGB>>
        constexpr ImageType operator()(const std::string_view arg, const std::shared_ptr<Workspace>& ws) const
        {
            if (arg.starts_with('$'))
            {
                const std::string_view var_name = arg.substr(1);
                if (const ImageType* img_ptr = ws->retrieve<ImageType>(var_name))
                {
                    return *img_ptr;
                }
                throw std::invalid_argument(std::string("Memory variable not found or type mismatch: ") + std::string(var_name));
            }

            const std::filesystem::path input_path = std::string(arg);
            if (!std::filesystem::exists(input_path))
            {
                throw std::invalid_argument(std::string("File not found: ") + input_path.string());
            }

            if constexpr (std::is_same_v<ImageType, TinyDIP::Image<TinyDIP::RGB>>)
            {
                return TinyDIP::bmp_read(input_path.string().c_str(), true);
            }
            else if constexpr (std::is_same_v<ImageType, TinyDIP::Image<double>>)
            {
                return TinyDIP::double_image::read(input_path.string().c_str(), true);
            }
            else if constexpr (
                std::is_same_v<ImageType, TinyDIP::Image<TinyDIP::RGB_DOUBLE>> ||
                std::is_same_v<ImageType, TinyDIP::Image<TinyDIP::HSV>> ||
                std::is_same_v<ImageType, TinyDIP::Image<TinyDIP::MultiChannel<double>>>
            )
            {
                throw std::invalid_argument("Direct file reading is not implemented for this complex/high-precision image type.");
            }
            else
            {
                throw std::invalid_argument("Direct file reading is not explicitly implemented for this abstract/complex image type.");
            }
        }
    };

    struct Saver
    {
        template <typename ImageType>
        constexpr void operator()(const std::string_view arg, const std::shared_ptr<Workspace>& ws, ImageType&& img) const
        {
            if (arg.starts_with('$'))
            {
                const std::string_view var_name = arg.substr(1);
                ws->store(var_name, std::forward<ImageType>(img));
            }
            else
            {
                const std::filesystem::path output_filepath = std::string(arg);
                const std::filesystem::path path_without_extension = output_filepath.parent_path() / output_filepath.stem();
                
                if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<double>>)
                {
                    TinyDIP::double_image::write(path_without_extension.string().c_str(), std::forward<ImageType>(img));
                }
                else if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::RGB>>)
                {
                    TinyDIP::bmp_write(path_without_extension.string().c_str(), std::forward<ImageType>(img));
                }
                else if constexpr (
                    std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::RGB_DOUBLE>> ||
                    std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::HSV>> ||
                    std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::MultiChannel<double>>>
                )
                {
                    throw std::invalid_argument("Direct file writing is not implemented for this complex/high-precision image type.");
                }
                else
                {
                    throw std::invalid_argument("Direct file writing is not explicitly implemented for this abstract/complex image type.");
                }
            }
        }
    };
};

//  dispatch_data_operation template function implementation
//  Generic helper to dynamically load and dispatch data (from memory or disk) to a processor lambda
template <typename CheckingTypes = master_image_types, typename ProcessorFun, typename ImageLoaderFun>
requires (std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&> &&
          std::invocable<ProcessorFun, std::invoke_result_t<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&>>)
constexpr bool dispatch_data_operation(
    const std::string_view input_arg,
    const std::shared_ptr<Workspace>& workspace,
    ImageLoaderFun&& image_loader,
    ProcessorFun&& processor)
{
    if (input_arg.starts_with('$'))
    {
        const std::string_view var_name = input_arg.substr(1);

        auto try_process = [&]<typename T>() -> bool
        {
            if (workspace->template retrieve<T>(var_name))
            {
                processor(image_loader.template operator()<T>(input_arg, workspace));
                return true;
            }
            return false;
        };

        return match_any_type<CheckingTypes>(try_process);
    }
    else
    {
        const std::filesystem::path input_path = std::string(input_arg);
        if (input_path.extension() == ".dbmp")
        {
            processor(image_loader.template operator()<TinyDIP::Image<double>>(input_arg, workspace));
        }
        else
        {
            processor(image_loader.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace));
        }
        return true;
    }
}

//  Custom Type-Erasure Wrapper (Concept-Model Idiom)
//  This acts like std::any/std::function but enforces a highly optimized span boundary internally
class CommandHandler
{
private:
    //  The abstract interface (Concept)
    struct Concept
    {
        virtual ~Concept() = default;
        //  Using std::span provides a zero-allocation, type-erased boundary for any contiguous range
        virtual void call(std::span<const std::string_view> args, std::ostream& os) const = 0;
        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    //  The concrete implementation wrapper (Model)
    template <typename HandlerT>
    struct Model final : Concept
    {
        HandlerT handler_;

        constexpr explicit Model(HandlerT handler) : handler_(std::move(handler))
        {
        }

        void call(std::span<const std::string_view> args, std::ostream& os) const override
        {
            //  Forward the span argument to the generic operator() of the encapsulated handler
            handler_(args, os);
        }

        std::unique_ptr<Concept> clone() const override
        {
            return std::make_unique<Model>(*this);
        }
    };

    std::unique_ptr<Concept> pimpl_;

public:
    //  Default constructor
    constexpr CommandHandler() noexcept : pimpl_(nullptr)
    {
    }

    //  Generic constructor for absolutely any callable
    template <typename HandlerT>
    requires (!std::same_as<std::decay_t<HandlerT>, CommandHandler>)
    constexpr CommandHandler(HandlerT&& handler)
        : pimpl_(std::make_unique<Model<std::decay_t<HandlerT>>>(std::forward<HandlerT>(handler)))
    {
    }

    //  Copy constructor (Deep copy of the type-erased object)
    CommandHandler(const CommandHandler& other)
        : pimpl_(other.pimpl_ ? other.pimpl_->clone() : nullptr)
    {
    }

    //  Move constructor
    constexpr CommandHandler(CommandHandler&&) noexcept = default;

    //  Copy assignment
    CommandHandler& operator=(const CommandHandler& other)
    {
        if (this != &other)
        {
            pimpl_ = other.pimpl_ ? other.pimpl_->clone() : nullptr;
        }
        return *this;
    }

    //  Move assignment
    constexpr CommandHandler& operator=(CommandHandler&&) noexcept = default;

    //  Execution operator
    void operator()(std::span<const std::string_view> args, std::ostream& os) const
    {
        if (pimpl_)
        {
            pimpl_->call(args, os);
        }
        else
        {
            throw std::bad_function_call();
        }
    }
};

//  IOSchema struct implementation
//  Schema defining implicit argument positions for the pipeline engine to auto-inject memory variables
struct IOSchema
{
    int in_idx = -1;
    int out_idx = -1;
};

//  Define human-readable pipeline schema routing constants globally
constexpr auto GeneratorSchema = IOSchema{ -1, 1 };
constexpr auto TerminatorSchema = IOSchema{ 0, -1 };
constexpr auto TransformerSchema = IOSchema{ 0, 1 };
constexpr auto IndependentSchema = IOSchema{ -1, -1 };

//  CommandRegistry class implementation
class CommandRegistry
{
public:
    struct CommandInfo
    {
        std::string description;
        IOSchema schema;
        CommandHandler handler;
    };

private:
    std::map<std::string, CommandInfo> commands;

public:
    void register_command(const std::string_view name, const std::string_view description, const IOSchema schema, CommandHandler handler)
    {
        commands.emplace(std::string(name), CommandInfo{std::string(description), schema, std::move(handler)});
    }

    //  Fallback for commands without pipeline routing specifications
    void register_command(const std::string_view name, const std::string_view description, CommandHandler handler)
    {
        commands.emplace(std::string(name), CommandInfo{std::string(description), IOSchema{-1, -1}, std::move(handler)});
    }

    std::optional<IOSchema> get_schema(const std::string_view command_name) const
    {
        if (auto it = commands.find(std::string(command_name)); it != std::ranges::end(commands))
        {
            return it->second.schema;
        }
        return std::nullopt;
    }

    void list_commands(std::ostream& os = std::cout) const
    {
        os << "Available Commands:\n";
        for (const auto& [name, info] : commands)
        {
            os << "  " << std::left << std::setw(15) << name << " : " << info.description << "\n";
        }
        os << "\nUsage: ./tinydip <command> [args...]\n";
        os << "Tip: Use '$name' to read/write from in-memory variables.\n";
        os << "Tip: Chain commands with '|' pipelines. (e.g. read file.bmp | bicubic_resize 512 512 | $out)\n";
    }

    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    void execute(const std::string& command_name, const ArgsT& args, std::ostream& os = std::cout) const
    {
        if (auto it = commands.find(command_name); it != std::ranges::end(commands))
        {
            try
            {
                if constexpr (std::ranges::contiguous_range<ArgsT> && std::same_as<std::ranges::range_value_t<ArgsT>, std::string_view>)
                {
                    it->second.handler(std::span<const std::string_view>{std::ranges::data(args), std::ranges::size(args)}, os);
                }
                else
                {
                    std::vector<std::string_view> contiguous_args;
                    contiguous_args.reserve(std::ranges::size(args));
                    for (const auto& arg : args)
                    {
                        contiguous_args.emplace_back(arg);
                    }
                    it->second.handler(std::span<const std::string_view>{std::ranges::data(contiguous_args), std::ranges::size(contiguous_args)}, os);
                }
            }
            catch (const std::exception& e)
            {
                os << "Error executing command '" << command_name << "': " << e.what() << "\n";
            }
        }
        else
        {
            os << "Unknown command: " << command_name << "\n";
            list_commands(os); 
        }
    }
};

//  --------------------------------------------------------------------------
//  Workspace Memory Operation Handlers
//  --------------------------------------------------------------------------

//  MetaTransformHandler template struct implementation
//  Generic Meta Handler strictly refactoring transform commands like abs, bicubic_resize, dct2, idct2, and lanczos_resample
template <std::size_t MinArgs, typename SetupFun, typename CheckingTypes = master_image_types>
struct MetaTransformHandler
{
    std::string_view usage_string_;
    std::shared_ptr<Workspace> workspace_;
    SetupFun setup_fun_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = MetaImageIO::Loader,
        typename ImageSaverFun = MetaImageIO::Saver
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<double>&&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        std::string_view policy_str = "";
        std::vector<std::string_view> filtered_args;
        filtered_args.reserve(std::ranges::size(args));

        for (const auto& arg : args)
        {
            const std::string_view sv_arg = arg;
            if (sv_arg == "seq" || sv_arg == "par" || sv_arg == "par_unseq" || sv_arg == "unseq")
            {
                policy_str = sv_arg;
            }
            else
            {
                filtered_args.emplace_back(sv_arg);
            }
        }

        if (std::ranges::size(filtered_args) < MinArgs)
        {
            os << "Usage: " << usage_string_ << "\n";
            if (usage_string_.find("[execution_policy]") != std::string_view::npos)
            {
                os << "       Optional Execution policies: seq, par, par_unseq, unseq\n";
            }
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view output_arg = filtered_args[1];

        // Parse trailing args, output initial message, and retrieve dedicated transformation process
        auto core_processor = setup_fun_(filtered_args, policy_str, os);

        std::optional<std::any> final_result_opt;

        auto process_wrapper = [&]<typename ImageType>(ImageType&& input_img)
        {
            final_result_opt = core_processor(std::forward<ImageType>(input_img));
        };

        if (!dispatch_data_operation<CheckingTypes>(input_arg, workspace_, image_loader_fun, process_wrapper))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
            return;
        }

        if (final_result_opt.has_value())
        {
            std::any output_any = std::move(*final_result_opt);

            bool handled = false;
            auto try_save_output = [&]<typename OutT>() -> bool
            {
                if (output_any.type() == typeid(OutT))
                {
                    image_saver_fun(output_arg, workspace_, std::move(std::any_cast<OutT&>(output_any)));
                    os << "Saved to " << output_arg << "\n";
                    handled = true;
                    return true;
                }
                return false;
            };

            if (!match_any_type<CheckingTypes>(try_save_output))
            {
                os << "Error: Output type from processor is unknown or unsupported. Type Name: [" 
                   << output_any.type().name() << "]\n";
            }
        }
    }
};

//  make_meta_transform_handler template function implementation
template <std::size_t MinArgs, typename CheckingTypes = master_image_types, typename SetupFun>
constexpr auto make_meta_transform_handler(std::string_view usage, std::shared_ptr<Workspace> ws, SetupFun&& setup)
{
    return MetaTransformHandler<MinArgs, std::remove_cvref_t<SetupFun>, CheckingTypes>{
        usage, std::move(ws), std::forward<SetupFun>(setup)
    };
}

//  MetaScalarHandler template struct implementation
//  Generic Meta Handler strictly refactoring scalar reduction commands like max, min, and sum
template <std::size_t MinArgs, typename SetupFun>
struct MetaScalarHandler
{
    std::string_view usage_string_;
    std::string_view op_name_;
    std::string_view capitalized_op_name_;
    std::shared_ptr<Workspace> workspace_;
    SetupFun setup_fun_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}) const
    {
        std::string_view policy_str = "";
        std::vector<std::string_view> filtered_args;
        filtered_args.reserve(std::ranges::size(args));

        for (const auto& arg : args)
        {
            const std::string_view sv_arg = arg;
            if (sv_arg == "seq" || sv_arg == "par" || sv_arg == "par_unseq" || sv_arg == "unseq")
            {
                policy_str = sv_arg;
            }
            else
            {
                filtered_args.emplace_back(sv_arg);
            }
        }

        if (std::ranges::size(filtered_args) < MinArgs)
        {
            os << "Usage: " << usage_string_ << "\n";
            if (usage_string_.find("[execution_policy]") != std::string_view::npos)
            {
                os << "       Optional Execution policies: seq, par, par_unseq, unseq\n";
            }
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        std::string_view output_arg = "";
        
        if (std::ranges::size(filtered_args) > 1)
        {
            output_arg = filtered_args[1];
        }

        auto core_processor = setup_fun_(filtered_args, policy_str, os);

        // Polymorphic lambda to cleanly execute the algorithm dynamically independent of image type
        auto process_scalar = [&]<typename ImageType>(ImageType&& input_img)
        {
            std::any scalar_result_any = core_processor(std::forward<ImageType>(input_img));

            bool handled = false;
            auto handle_result = [&]<typename ScalarT>() -> bool
            {
                if (scalar_result_any.type() == typeid(ScalarT))
                {
                    auto& scalar_result = std::any_cast<ScalarT&>(scalar_result_any);
                    if (!std::ranges::empty(output_arg))
                    {
                        if (output_arg.starts_with('$'))
                        {
                            workspace_->store(output_arg.substr(1), scalar_result);
                            os << "Saved " << op_name_ << " result to " << output_arg << "\n";
                        }
                        else
                        {
                            os << "Error: Output must be a memory variable starting with '$'.\n";
                        }
                    }
                    else
                    {
                        if constexpr (requires { os << scalar_result; })
                        {
                            if constexpr (sizeof(ScalarT) == 1 && std::is_integral_v<ScalarT>)
                            {
                                os << capitalized_op_name_ << " result: " << +scalar_result << "\n";
                            }
                            else
                            {
                                os << capitalized_op_name_ << " result: " << scalar_result << "\n";
                            }
                        }
                        else if constexpr (is_vector_v<ScalarT> || is_deque_v<ScalarT> || is_list_v<ScalarT> || is_std_array_v<ScalarT>)
                        {
                            os << capitalized_op_name_ << " result: {";
                            bool first = true;
                            for (const auto& elem : scalar_result)
                            {
                                if (!first)
                                {
                                    os << ", ";
                                }
                                os << +elem;
                                first = false;
                            }
                            os << "}\n";
                        }
                        else
                        {
                            os << capitalized_op_name_ << " result evaluated successfully (Non-printable complex type).\n";
                        }
                    }
                    handled = true;
                    return true;
                }
                return false;
            };

            if (!match_any_type<master_scalar_types>(handle_result))
            {
                os << "Error: Output type from processor is unknown or unsupported. Type Name: [" 
                   << scalar_result_any.type().name() << "]\n";
            }
        };

        if (!dispatch_data_operation<master_data_types>(input_arg, workspace_, image_loader_fun, process_scalar))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }
};

//  make_meta_scalar_handler template function implementation
template <std::size_t MinArgs, typename SetupFun>
constexpr auto make_meta_scalar_handler(std::string_view usage, std::string_view op_name, std::string_view capitalized_op_name, std::shared_ptr<Workspace> ws, SetupFun&& setup)
{
    return MetaScalarHandler<MinArgs, std::remove_cvref_t<SetupFun>>{
        usage, op_name, capitalized_op_name, std::move(ws), std::forward<SetupFun>(setup)
    };
}

//  ReadHandler struct implementation
struct ReadHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT, 
        std::invocable<const std::string_view, const std::shared_ptr<Workspace>&> ImageLoaderFun = MetaImageIO::Loader,
        std::invocable<const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<TinyDIP::RGB>&&> ImageSaverFun = MetaImageIO::Saver
    >
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        if (std::ranges::empty(args))
        {
            os << "Usage: read <input_file> [$var]\n";
            return;
        }

        const std::string_view input_arg = args[0];
        std::string output_arg_str;
        std::string_view output_arg;

        if (std::ranges::size(args) > 1)
        {
            output_arg = args[1];
            if (!output_arg.starts_with('$'))
            {
                os << "Error: Output must be a memory variable starting with '$'.\n";
                return;
            }
        }
        else
        {
            // Dynamically assign variable name from origin file name stem
            const std::filesystem::path input_path = std::string(input_arg);
            output_arg_str = "$" + input_path.stem().string();
            output_arg = output_arg_str;
        }

        os << "Reading " << input_arg << " into memory as " << output_arg << "...\n";
        
        auto process_read = [&]<typename ImageType>(ImageType&& input_img)
        {
            image_saver_fun(output_arg, workspace_, std::forward<ImageType>(input_img));
        };

        if (!dispatch_data_operation(input_arg, workspace_, image_loader_fun, process_read))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
            return;
        }

        os << "Done.\n";
    }
};

//  RenameHandler struct implementation
struct RenameHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: rename <$old_var> <$new_var>\n";
            return;
        }

        const std::string_view old_arg = args[0];
        const std::string_view new_arg = args[1];

        if (!old_arg.starts_with('$') || !new_arg.starts_with('$'))
        {
            os << "Error: Both arguments must be memory variables starting with '$'.\n";
            return;
        }

        const std::string_view old_name = old_arg.substr(1);
        const std::string_view new_name = new_arg.substr(1);

        if (workspace_->rename(old_name, new_name))
        {
            os << "Renamed variable $" << old_name << " to $" << new_name << ".\n";
        }
        else
        {
            os << "Error: Memory variable $" << old_name << " not found.\n";
        }
    }
};

//  RemoveHandler struct implementation
struct RemoveHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        if (std::ranges::empty(args))
        {
            os << "Usage: remove <$var1> [$var2] ... OR remove all\n";
            return;
        }

        if (std::ranges::size(args) == 1 && std::string_view(args[0]) == "all")
        {
            workspace_->clear();
            os << "Removed all memory variables. Workspace is now empty.\n";
            return;
        }

        for (const auto& arg : args)
        {
            const std::string_view var_arg = arg;
            if (!var_arg.starts_with('$'))
            {
                os << "Error: Argument must be a memory variable starting with '$' or 'all'. Skipped " << var_arg << ".\n";
                continue;
            }

            const std::string_view var_name = var_arg.substr(1);
            if (workspace_->remove(var_name))
            {
                os << "Removed memory variable $" << var_name << ".\n";
            }
            else
            {
                os << "Warning: Memory variable $" << var_name << " not found.\n";
            }
        }
    }
};

//  WriteHandler struct implementation
struct WriteHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = MetaImageIO::Loader,
        typename ImageSaverFun = MetaImageIO::Saver
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<double>&&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: write <$var> <output_file>\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];

        if (!input_arg.starts_with('$'))
        {
            os << "Error: Input must be a memory variable starting with '$'.\n";
            return;
        }

        os << "Writing memory variable " << input_arg << " to file " << output_arg << "...\n";

        auto process_write = [&]<typename ImageType>(ImageType&& input_img)
        {
            image_saver_fun(output_arg, workspace_, std::forward<ImageType>(input_img));
        };

        if (!dispatch_data_operation(input_arg, workspace_, image_loader_fun, process_write))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
            return;
        }

        os << "Done.\n";
    }
};

//  VarsHandler struct implementation
struct VarsHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        (void)args;
        os << "Current Workspace Variables:\n";
        workspace_->list_variables(os);
    }
};

//  SaveWorkspaceHandler struct implementation
struct SaveWorkspaceHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        if (std::ranges::size(args) < 1)
        {
            os << "Usage: save_workspace <directory_bundle_path>\n";
            return;
        }

        const std::filesystem::path dir_path = std::string(args[0]);
        std::filesystem::create_directories(dir_path);

        os << "Saving workspace bundle to " << dir_path.string() << "...\n";

        for (const auto& [name, value] : workspace_->memory_store)
        {
            auto try_save_image = [&]<typename T>() -> bool
            {
                if (value.type() == typeid(T))
                {
                    const auto* img_ptr = std::any_cast<T>(&value);
                    const std::filesystem::path file_path = dir_path / (name);
                    
                    if constexpr (std::is_same_v<T, TinyDIP::Image<TinyDIP::RGB>>)
                    {
                        TinyDIP::bmp_write(file_path.string().c_str(), *img_ptr); 
                        os << "  Saved $" << name << " -> " << file_path.string() << ".bmp\n";
                    }
                    else if constexpr (std::is_same_v<T, TinyDIP::Image<double>>)
                    {
                        TinyDIP::double_image::write(file_path.string().c_str(), *img_ptr); 
                        os << "  Saved $" << name << " -> " << file_path.string() << ".dbmp\n";
                    }
                    return true;
                }
                return false;
            };

            using saveable_image_types = std::tuple<
                TinyDIP::Image<TinyDIP::RGB>,
                TinyDIP::Image<double>
            >;

            if (match_any_type<saveable_image_types>(try_save_image))
            {
                // Saved successfully
            }
            else
            {
                using unsupported_image_types = std::tuple<
                    TinyDIP::Image<TinyDIP::RGB_DOUBLE>,
                    TinyDIP::Image<TinyDIP::HSV>,
                    TinyDIP::Image<TinyDIP::MultiChannel<double>>
                >;

                auto try_skip_image = [&]<typename T>() -> bool
                {
                    if (value.type() == typeid(T))
                    {
                        os << "  Skipped $" << name << " (Serialization not implemented for this complex image type)\n";
                        return true;
                    }
                    return false;
                };

                if (!match_any_type<unsupported_image_types>(try_skip_image))
                {
                    os << "  Skipped $" << name << " (Unsupported serialization type)\n";
                }
            }
        }
        os << "Workspace saved successfully.\n";
    }
};

//  LoadWorkspaceHandler struct implementation
struct LoadWorkspaceHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        if (std::ranges::size(args) < 1)
        {
            os << "Usage: load_workspace <directory_bundle_path>\n";
            return;
        }

        const std::filesystem::path dir_path = std::string(args[0]);

        if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path))
        {
            os << "Error: Workspace directory bundle does not exist: " << dir_path.string() << "\n";
            return;
        }

        os << "Loading workspace bundle from " << dir_path.string() << "...\n";

        for (const auto& entry : std::filesystem::directory_iterator(dir_path))
        {
            if (entry.is_regular_file())
            {
                const std::string name = entry.path().stem().string();
                const std::string ext = entry.path().extension().string();

                if (ext == ".bmp")
                {
                    auto img = TinyDIP::bmp_read(entry.path().string().c_str(), true);
                    workspace_->store(name, std::move(img));
                    os << "  Loaded " << entry.path().filename().string() << " -> $" << name << "\n";
                }
                else if (ext == ".dbmp")
                {
                    auto img = TinyDIP::double_image::read(entry.path().string().c_str(), true);
                    workspace_->store(name, std::move(img));
                    os << "  Loaded " << entry.path().filename().string() << " -> $" << name << "\n";
                }
            }
        }
        os << "Workspace loaded successfully.\n";
    }
};

//  HelpHandler struct implementation
//  Wrapper for the 'help' functionality inside the REPL
struct HelpHandler
{
    const CommandRegistry& registry_;

    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        //  Silencing unused parameter warning
        (void)args;
        registry_.list_commands(os);
    }
};

//  InfoHandler struct implementation
//  Wrapper for 'info' functionality
//  Args: input_path
struct InfoHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}) const
    {
        if (std::ranges::empty(args))
        {
            os << "Usage: info <input_bmp | $var>\n";
            return;
        }

        const std::string_view input_arg = args[0];

        // Polymorphic lambda to cleanly print dimensions dynamically independent of image type
        auto process_info = [&]<typename ImageType>(const ImageType& img)
            requires (TinyDIP::is_Image<std::remove_cvref_t<ImageType>>::value)
        {
            os << "Image Info:\n";
            os << "  Source: " << input_arg << "\n";
            os << "  Width:  " << img.getWidth() << "\n";
            os << "  Height: " << img.getHeight() << "\n";
        };

        if (!dispatch_data_operation(input_arg, workspace_, image_loader_fun, process_info))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }
};


//  PrintHandler struct implementation
struct PrintHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}) const
    {
        if (std::ranges::empty(args))
        {
            os << "Usage: print <input_bmp | $var>\n";
            return;
        }

        const std::string_view input_arg = args[0];

        // Polymorphic lambda to cleanly print image content dynamically independent of image type
        auto process_print = [&]<typename ImageType>(const ImageType& img)
            requires (TinyDIP::is_Image<std::remove_cvref_t<ImageType>>::value)
        {
            os << "Printing image content for " << input_arg << ":\n";
            img.print(",");
            os << "Done.\n";
        };

        if (!dispatch_data_operation(input_arg, workspace_, image_loader_fun, process_print))
        {
            // If dispatch_data_operation returns false, it must be a $ variable holding a scalar or unsupported type
            const std::string_view var_name = input_arg.substr(1);
            
            // Polymorphic lambda returning true if the complex custom scalar type matched
            auto try_print_complex_scalar = [&]<typename T>() -> bool
            {
                if (workspace_->retrieve<T>(var_name))
                {
                    os << "Printing scalar value for " << input_arg << ":\n";
                    if constexpr (is_vector_v<T> || is_deque_v<T> || is_list_v<T> || is_std_array_v<T>)
                    {
                        os << "container value = {";
                        bool first = true;
                        const auto* container_ptr = workspace_->retrieve<T>(var_name);
                        for (const auto& elem : *container_ptr)
                        {
                            if (!first)
                            {
                                os << ", ";
                            }
                            os << +elem;
                            first = false;
                        }
                        os << "}\nDone.\n";
                    }
                    else
                    {
                        os << *workspace_->retrieve<T>(var_name) << "\nDone.\n";
                    }
                    return true;
                }
                return false;
            };

            if (match_any_type<complex_scalar_types_for_printing>(try_print_complex_scalar))
            {
                // Handled successfully by try_print_complex_scalar short-circuit logic
            }
            else
            {
                // Polymorphic lambda returning true if the numeric type matched
                auto try_print_numeric = [&]<typename T>() -> bool
                {
                    if (workspace_->retrieve<T>(var_name))
                    {
                        os << "Printing scalar value for " << input_arg << ":\n";
                        if constexpr (sizeof(T) == 1 && std::is_integral_v<T>) // Safely print 8-bit integer types as numbers, not unprintable chars
                        {
                            os << +(*workspace_->retrieve<T>(var_name)) << "\nDone.\n";
                        }
                        else
                        {
                            os << *workspace_->retrieve<T>(var_name) << "\nDone.\n";
                        }
                        return true;
                    }
                    return false;
                };

                if (!match_any_type<core_numeric_types>(try_print_numeric))
                {
                    os << "Error: Memory variable not found or unsupported type.\n";
                }
            }
        }
    }
};

//  RandHandler struct implementation
//  Wrapper for 'rand' functionality
//  Args: urbg_type output_path dim1 [dim2] [dim3] ...
struct RandHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <typename Urbg, typename Dist>
    requires (std::uniform_random_bit_generator<std::remove_reference_t<Urbg>> &&
              std::invocable<Dist&, Urbg&>)
    struct RandomGenerator
    {
        Urbg& urbg_;
        Dist& dist_;

        constexpr auto operator()()
        {
            return dist_(urbg_);
        }
    };

    template <
        std::ranges::random_access_range ArgsT,
        std::invocable<const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<double>&&> ImageSaverFun = MetaImageIO::Saver
    >
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        auto dispatch_generation = [&]
        <std::ranges::random_access_range SzArgsT>
        requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
        (std::uniform_random_bit_generator auto&& urbg, const std::string_view& out_path, const SzArgsT& sz)
        {
            std::uniform_real_distribution<double> dist{};
            using UrbgType = std::remove_cvref_t<decltype(urbg)>;
            using DistType = decltype(dist);

            RandomGenerator<UrbgType, DistType> gen{urbg, dist};

            //  Calling the dynamic range-based generate overload directly from TinyDIP.
            auto output_img = TinyDIP::generate(gen, sz);

            // Dynamically save image via the injected saver abstraction
            image_saver_fun(out_path, workspace_, std::move(output_img));
            os << "Saved to " << out_path << "\n";
        };

        std::map<std::string_view, std::function<void(const std::string_view&, std::span<const std::size_t>)>> urbg_mapping = {
            {"knuth_b",       [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::knuth_b{std::random_device{}()}, out_path, sz); }
            },
            {"minstd_rand",   [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::minstd_rand{std::random_device{}()}, out_path, sz); }
            },
            {"minstd_rand0",  [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::minstd_rand0{std::random_device{}()}, out_path, sz); }
            },
            {"mt19937",       [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::mt19937{std::random_device{}()}, out_path, sz); }
            },
            {"mt19937_64",    [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::mt19937_64{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux24",      [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::ranlux24{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux24_base", [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::ranlux24_base{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux48",      [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::ranlux48{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux48_base", [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::string_view& out_path, const SzArgsT& sz)
                { dispatch_generation(std::ranlux48_base{std::random_device{}()}, out_path, sz); }
            }
        };

        auto print_available_urbgs = [&]()
        {
            os << "Available URBGs: ";
            for (auto it = std::ranges::begin(urbg_mapping); it != std::ranges::end(urbg_mapping); ++it)
            {
                os << it->first;
                if (std::next(it) != std::ranges::end(urbg_mapping))
                {
                    os << ", ";
                }
            }
            os << '\n';
        };

        if (std::ranges::size(args) < 3)
        {
            os << "Usage: rand <urbg_type> <output_bmp | $var> <dim1> [dim2] [dim3] ...\n";
            print_available_urbgs();
            return;
        }

        const std::string_view urbg_type = args[0];
        const std::string_view output_arg = args[1];
        
        std::vector<std::size_t> sizes;
        sizes.reserve(std::ranges::size(args) - 2);
        
        for (std::size_t i = 2; i < std::ranges::size(args); ++i)
        {
            sizes.emplace_back(parse_arg<std::size_t>(args[i]));
        }

        os << "Generating random image with dimensions: ";
        for (const auto& size : sizes)
        {
            os << size << " ";
        }
        os << "using URBG '" << urbg_type << "'...\n";

        if (auto it = urbg_mapping.find(urbg_type); it != std::ranges::end(urbg_mapping))
        {
            it->second(output_arg, sizes);
        }
        else
        {
            os << "Error: Unknown URBG type '" << urbg_type << "'.\n";
            print_available_urbgs();
        }
    }
};

//  run_legacy_tests function implementation
//  Legacy test function wrapper
template <std::ranges::random_access_range ArgsT>
requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
void run_legacy_tests(const ArgsT& args, std::ostream& os = std::cout)
{
    os << "Running legacy integration tests...\n";
    
    TinyDIP::Timer timer1;
    std::string file_path = "InputImages/1";
    auto bmp1 = TinyDIP::bmp_read(file_path.c_str(), false);
    std::size_t N1 = 8, N2 = 8;
    auto block_count_x = bmp1.getWidth() / N1;
    auto block_count_y = bmp1.getHeight() / N2;

    auto hsv_image = TinyDIP::rgb2hsv(TinyDIP::im2double(bmp1));
    auto h_plane = TinyDIP::getHplane(hsv_image);
    auto s_plane = TinyDIP::getSplane(hsv_image);
    auto v_plane = TinyDIP::getVplane(hsv_image);
    auto split_v_plane = TinyDIP::split(v_plane, block_count_x, block_count_y);
    auto block_maximum = TinyDIP::recursive_transform<2>(
        //std::execution::par,
        [](auto&& element)
        {
            return TinyDIP::max(element);
        }, split_v_plane);
    auto block_minimum = TinyDIP::recursive_transform<2>(
        //std::execution::par,
        [](auto&& element)
        {
            return TinyDIP::min(element);
        }, split_v_plane);
    v_plane = TinyDIP::concat(TinyDIP::recursive_transform<2>(
        //std::execution::par,
        [](auto&& element)
        {
            auto v_block_dct = TinyDIP::dct2(element);
            return TinyDIP::idct2(v_block_dct);
        }, split_v_plane));
    bmp1 = copyResizeBicubic(bmp1, bmp1.getWidth() * 2, bmp1.getHeight() * 2);
    //bmp1 = gaussian_fisheye(bmp1, 800.0);
    auto SIFT_keypoints = TinyDIP::SIFT_impl::get_potential_keypoint(std::execution::par, v_plane);
    os << "SIFT_keypoints = " << SIFT_keypoints.size() << "\n";
    bmp1 = TinyDIP::draw_points(bmp1, SIFT_keypoints);
    for (auto&& each_SIFT_keypoint : SIFT_keypoints)
    {
        TinyDIP::SIFT_impl::get_keypoint_descriptor(v_plane, each_SIFT_keypoint);
    }
    #ifdef USE_OPENCV
    auto cv_mat = TinyDIP::to_cv_mat(bmp1);
    cv::imshow("Image", cv_mat);
    cv::waitKey(0);
    #endif
    TinyDIP::bmp_write("test20241026", bmp1);
    return;
}

//  CommandBundle template struct implementation
//  Helper struct to bundle command metadata with its functor for variadic registration
//  Utilizing std::string_view eliminates dynamic allocations on string literal input.
template <typename FunT>
struct CommandBundle
{
    std::string_view name;
    std::string_view description;
    IOSchema schema;
    FunT handler;
};

//  Deduction guide to guarantee smooth C++20 CTAD with string literals
template <typename FunT>
CommandBundle(const char*, const char*, IOSchema, FunT) -> CommandBundle<FunT>;

//  command_registration template function implementation
//  command_registration utilizes C++20 constrained variadic templates and C++17 Fold Expressions
//  to automatically iterate and register an arbitrary number of handlers gracefully.
template <std::invocable<std::span<const std::string_view>, std::ostream&>... Funs>
constexpr CommandRegistry command_registration(CommandBundle<Funs>&&... bundles)
{
    CommandRegistry registry;

    //  Unpack and register all provided command bundles automatically using a C++17 fold expression
    (registry.register_command(bundles.name, bundles.description, bundles.schema, std::forward<Funs>(bundles.handler)), ...);

    //  Internal / Anonymous Handlers can still be registered statically here
    registry.register_command("test", "Run internal integration tests.", 
        [](std::span<const std::string_view> args, std::ostream& os)
        {
            run_legacy_tests(args, os);
        }
    );

    registry.register_command("batch_add_zeros", "Add leading zeros to filenames in a directory.",
        [](std::span<const std::string_view> args, std::ostream& os)
        {
            if (std::ranges::size(args) < 2)
            {
                os << "Usage: batch_add_zeros <input_dir> <output_dir>\n";
                return;
            }
            os << "Batch processing from " << args[0] << " to " << args[1] << "\n";
        }
    );

    return registry;
}

//  run_interactive_mode function implementation
//  Interactive REPL loop implementation with dynamic Pipeline '|' argument injection
void run_interactive_mode(const CommandRegistry& registry, std::ostream& os = std::cout)
{
    os << "TinyDIP Interactive Interpreter\n";
    os << "Type 'help' to list commands, or 'exit' / 'quit' to terminate.\n";

    std::string line;
    while (true)
    {
        os << "tinydip> ";
        if (!std::getline(std::cin, line))
        {
            break;
        }

        if (std::ranges::empty(line))
        {
            continue;
        }

        std::vector<std::string> segments_raw;
        std::stringstream ss_line(line);
        std::string segment;
        
        while (std::getline(ss_line, segment, '|'))
        {
            segments_raw.emplace_back(std::move(segment));
        }

        std::vector<std::vector<std::string>> segments_tokens;
        for (const auto& seg_raw : segments_raw)
        {
            std::vector<std::string> tokens;
            std::istringstream iss(seg_raw);
            std::string token;
            while (iss >> token)
            {
                tokens.emplace_back(std::move(token));
            }
            if (!std::ranges::empty(tokens))
            {
                segments_tokens.emplace_back(std::move(tokens));
            }
        }

        if (std::ranges::empty(segments_tokens))
        {
            continue;
        }

        std::string final_output_var = "";
        if (std::ranges::size(segments_tokens) > 1 && 
            std::ranges::size(segments_tokens.back()) == 1 && 
            segments_tokens.back()[0].starts_with('$'))
        {
            final_output_var = segments_tokens.back()[0];
            segments_tokens.pop_back();
        }

        std::string prev_pipe_var = "";
        for (std::size_t i = 0; i < std::ranges::size(segments_tokens); ++i)
        {
            const auto& tokens = segments_tokens[i];
            const std::string command_name = tokens[0];

            if (command_name == "exit" || command_name == "quit")
            {
                return;
            }

            std::optional<IOSchema> schema_opt = registry.get_schema(command_name);
            if (!schema_opt)
            {
                os << "Unknown command: " << command_name << "\n";
                break;
            }
            const IOSchema schema = *schema_opt;

            std::string next_pipe_var = "";
            if (i < std::ranges::size(segments_tokens) - 1)
            {
                next_pipe_var = "$__pipe_" + std::to_string(i);
            }
            else if (!std::ranges::empty(final_output_var))
            {
                next_pipe_var = final_output_var;
            }

            std::vector<std::string> args_str;
            for (std::size_t j = 1; j < std::ranges::size(tokens); ++j)
            {
                args_str.emplace_back(tokens[j]);
            }

            // Inject intermediate variables into the correct argument index
            if (schema.in_idx != -1 && !std::ranges::empty(prev_pipe_var))
            {
                const int insert_pos = std::min(schema.in_idx, static_cast<int>(std::ranges::size(args_str)));
                args_str.insert(std::ranges::begin(args_str) + insert_pos, prev_pipe_var);
            }

            if (schema.out_idx != -1 && !std::ranges::empty(next_pipe_var))
            {
                const int insert_pos = std::min(schema.out_idx, static_cast<int>(std::ranges::size(args_str)));
                args_str.insert(std::ranges::begin(args_str) + insert_pos, next_pipe_var);
            }

            std::vector<std::string_view> args_sv;
            args_sv.reserve(std::ranges::size(args_str));
            for (const auto& arg : args_str)
            {
                args_sv.emplace_back(arg);
            }

            registry.execute(command_name, args_sv, os);

            prev_pipe_var = next_pipe_var;
        }
    }
}

//  dispatch_policy_string template function implementation
//  Helper to dispatch execution policy string to std::execution policies
template <typename PolicyFun, std::invocable DefaultFun>
constexpr std::any dispatch_policy_string(
    const std::string_view policy_str,
    PolicyFun&& policy_fun,
    DefaultFun&& default_fun,
    std::ostream& os)
{
    if (policy_str == "par")
    {
        return std::forward<PolicyFun>(policy_fun)(std::execution::par);
    }
    else if (policy_str == "par_unseq")
    {
        return std::forward<PolicyFun>(policy_fun)(std::execution::par_unseq);
    }
    else if (policy_str == "unseq")
    {
        return std::forward<PolicyFun>(policy_fun)(std::execution::unseq);
    }
    else if (policy_str == "seq")
    {
        return std::forward<PolicyFun>(policy_fun)(std::execution::seq);
    }
    else
    {
        if (!std::ranges::empty(policy_str))
        {
            os << "Warning: Unknown execution policy '" << policy_str << "'. Falling back to default.\n";
        }
        return std::forward<DefaultFun>(default_fun)();
    }
}

//  AbsOp struct implementation
struct AbsOp
{
    template <typename... Args>
    requires requires { TinyDIP::abs(std::declval<Args>()...); }
    static constexpr auto exec(Args&&... args) { return TinyDIP::abs(std::forward<Args>(args)...); }

    template <typename T>
    requires (!requires { TinyDIP::abs(std::declval<T>()); } && 
              !std::ranges::input_range<std::remove_cvref_t<T>> && 
              requires { TinyDIP::generic_abs(std::declval<T>()); })
    static constexpr auto exec(T&& arg) { return TinyDIP::generic_abs(std::forward<T>(arg)); }
};

//  Dct2Op struct implementation
struct Dct2Op
{
    template <typename... Args> requires requires { TinyDIP::dct2(std::forward<Args>()...); }
    static constexpr auto exec(Args&&... args) { return TinyDIP::dct2(std::forward<Args>(args)...); }
};

//  SumOp struct implementation
struct SumOp
{
    template <typename ValT>
    struct SumAccumulator
    {
        ValT& sum_ref;

        template <class T>
        constexpr auto operator()(const T& element) const
        {
            sum_ref += element;
            return element;
        }
    };

    template <typename ValT>
    struct ParallelSumAccumulator
    {
        ValT& sum_ref;
        std::mutex& mtx_ref;

        template <class T>
        auto operator()(const T& element) const
        {
            std::lock_guard<std::mutex> lock(mtx_ref);
            sum_ref += element;
            return element;
        }
    };

    template <typename... Args> requires requires { TinyDIP::sum(std::declval<Args>()...); }
    static constexpr auto exec(Args&&... args) { return TinyDIP::sum(std::forward<Args>(args)...); }

    template <typename ContainerT> requires (!requires { TinyDIP::sum(std::declval<ContainerT>()); } && std::ranges::input_range<std::remove_cvref_t<ContainerT>>)
    static constexpr auto exec(ContainerT&& data)
    {
        using DecayedT = std::remove_cvref_t<ContainerT>;
        if constexpr (TinyDIP::recursive_depth<DecayedT>() > 1)
        {
            using ValT = decltype(exec(*std::ranges::begin(data)));
            ValT total_sum{};
            
            TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedT>()>(
                SumAccumulator<ValT>{total_sum},
                std::forward<ContainerT>(data)
            );
            
            return total_sum;
        }
        else
        {
            using ValT = std::ranges::range_value_t<ContainerT>;
            return std::accumulate(std::ranges::begin(data), std::ranges::end(data), ValT{});
        }
    }

    template <typename ExecPolicy, typename ContainerT> requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
    static constexpr auto exec(ExecPolicy&& policy, ContainerT&& data) requires (!requires { TinyDIP::sum(std::declval<ExecPolicy>(), std::declval<ContainerT>()); } && std::ranges::input_range<std::remove_cvref_t<ContainerT>>)
    {
        using DecayedT = std::remove_cvref_t<ContainerT>;
        if constexpr (TinyDIP::recursive_depth<DecayedT>() > 1)
        {
            using ValT = decltype(exec(std::forward<ExecPolicy>(policy), *std::ranges::begin(data)));
            ValT total_sum{};
            std::mutex mtx;
            
            TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedT>()>(
                std::forward<ExecPolicy>(policy),
                ParallelSumAccumulator<ValT>{total_sum, mtx},
                std::forward<ContainerT>(data)
            );
            
            return total_sum;
        }
        else
        {
            using ValT = std::ranges::range_value_t<ContainerT>;
            return std::reduce(std::forward<ExecPolicy>(policy), std::ranges::begin(data), std::ranges::end(data), ValT{});
        }
    }
};

//  make_unary_transform_bundle template function implementation
//  Generic Factory Builder for Unary Transformations
template <typename CoreOp>
constexpr auto make_unary_transform_bundle(
    const std::string_view name,
    const std::string_view description,
    std::shared_ptr<Workspace> workspace)
{
    auto handler = make_meta_transform_handler<2, master_data_types>(
        std::string(name) + " [execution_policy] <input_img | $var> <output_img | $var>",
        workspace,
        [name](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
        {
            os << "Executing " << name << " on " << filtered_args[0] << (!std::ranges::empty(policy_str) ? std::string(" (Policy: ") + std::string(policy_str) + ")" : "") << "...\n";

            return [policy_str, name, &os]<typename DataT>(DataT&& data) -> std::any
            {
                using DecayedDataT = std::remove_cvref_t<DataT>;

                // Create a generic, tightly constrained lambda for robust SFINAE-friendly execution
                auto transform_lambda = [](auto&& element) requires requires { CoreOp::exec(std::forward<decltype(element)>(element)); }
                {
                    return CoreOp::exec(std::forward<decltype(element)>(element));
                };

                auto exec_default = [&]() -> std::any
                {
                    if constexpr (requires { CoreOp::exec(std::forward<DataT>(data)); })
                    {
                        return CoreOp::exec(std::forward<DataT>(data));
                    }
                    else if constexpr (std::ranges::input_range<DecayedDataT> && requires { TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(transform_lambda, std::forward<DataT>(data)); })
                    {
                        return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                            transform_lambda,
                            std::forward<DataT>(data)
                        );
                    }
                    else
                    {
                        throw std::invalid_argument(std::string("Input type does not support ") + std::string(name));
                    }
                };

                auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                    requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                {
                    if constexpr (requires { CoreOp::exec(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                    {
                        return CoreOp::exec(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
                    }
                    else if constexpr (std::ranges::input_range<DecayedDataT> && requires { TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(std::forward<ExecPolicy>(exec_policy), transform_lambda, std::forward<DataT>(data)); })
                    {
                        return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                            std::forward<ExecPolicy>(exec_policy),
                            transform_lambda,
                            std::forward<DataT>(data)
                        );
                    }
                    else
                    {
                        if (!std::ranges::empty(policy_str))
                        {
                            os << "Warning: Execution policy requested but not supported for " << name << ". Falling back to default.\n";
                        }
                        return exec_default();
                    }
                };

                return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
            };
        }
    );

    return CommandBundle<decltype(handler)>{name, description, TransformerSchema, std::move(handler)};
}

//  make_scalar_reduction_bundle template function implementation
//  Generic Factory Builder for Scalar Reductions
template <typename CoreOp>
constexpr auto make_scalar_reduction_bundle(
    const std::string_view name,
    const std::string_view description,
    const std::string_view cap_name,
    std::shared_ptr<Workspace> ws)
{
    auto handler = make_meta_scalar_handler<1>(
        std::string(name) + " [execution_policy] <input_data | $var> [output_var | $var]",
        name, cap_name, ws,
        [name](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
        {
            os << "Calculating " << name << " of " << filtered_args[0] << (!std::ranges::empty(policy_str) ? std::string(" (Policy: ") + std::string(policy_str) + ")" : "") << "...\n";

            return [policy_str, name, &os]<typename DataT>(DataT && raw_data) -> std::any
            {
                // Helper to address conversion requirements before CoreOp dynamically
                auto execute_inner = [&]<typename TargetT>(TargetT && data) -> std::any
                {
                    auto exec_default = [&]() -> std::any { return CoreOp::exec(std::forward<TargetT>(data)); };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (requires { CoreOp::exec(std::forward<ExecPolicy>(exec_policy), std::forward<TargetT>(data)); })
                        {
                            return CoreOp::exec(std::forward<ExecPolicy>(exec_policy), std::forward<TargetT>(data));
                        }
                        else
                        {
                            if (!std::ranges::empty(policy_str)) os << "Warning: Execution policy requested but not supported. Falling back to default.\n";
                            return exec_default();
                        }
                    };
                    return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                };

                if constexpr (std::same_as<std::remove_cvref_t<DataT>, TinyDIP::Image<TinyDIP::RGB>> && std::same_as<CoreOp, SumOp>)
                {
                    return execute_inner(TinyDIP::im2double(std::forward<DataT>(raw_data)));
                }
                else
                {
                    return execute_inner(std::forward<DataT>(raw_data));
                }
            };
        }
    );

    return CommandBundle<decltype(handler)>{name, description, TransformerSchema, std::move(handler)};
}

//  Main Entry Point
int main(int argc, char* argv[])
{
    // Configure the shared state memory workspace
    auto workspace = std::make_shared<Workspace>();
    
    // Register commands directly with context-injected instances using generic variadic bundles
    CommandRegistry registry = command_registration(
        make_unary_transform_bundle<AbsOp>("abs", "Calculate the absolute value of an image or container.", workspace),
        CommandBundle{"bicubic_resize", "Resize an image using Bicubic interpolation.", TransformerSchema,
            make_meta_transform_handler<4>(
                "bicubic_resize [execution_policy] <input_img | $var> <output_img | $var> <width> <height>",
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    const std::size_t width = parse_arg<std::size_t>(filtered_args[2]);
                    const std::size_t height = parse_arg<std::size_t>(filtered_args[3]);
                    os << "Resizing " << filtered_args[0] << " to " << width << "x" << height << "...\n";

                    return [width, height, policy_str, &os]<typename ImageType>(ImageType && img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any { return TinyDIP::copyResizeBicubic(std::forward<ImageType>(img), width, height); };
                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::copyResizeBicubic(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height); })
                                return TinyDIP::copyResizeBicubic(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height);
                            else
                                return exec_default();
                        };
                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            )
        },
        CommandBundle{"dct2", "Calculate Discrete Cosine Transformation for an image.", TransformerSchema,
            make_meta_transform_handler<2>(
                "dct2 [execution_policy] <input_img | $var> <output_img | $var>", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Calculating DCT-2 for " << filtered_args[0] << " (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Calculating DCT-2 for " << filtered_args[0] << "...\n";
                    }

                    return [policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            return TinyDIP::dct2(std::forward<ImageType>(img));
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::dct2(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img)); })
                            {
                                return TinyDIP::dct2(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img));
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };
                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            )
        },
        CommandBundle{"getBplane", "Extract the Blue plane (channel 2) from a multi-channel image.", TransformerSchema, 
            make_meta_transform_handler<2>(
                "getBplane [execution_policy] <input_img | $var> <output_img | $var>", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Extracting Blue plane (channel 2) of " << filtered_args[0] << " (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Extracting Blue plane (channel 2) of " << filtered_args[0] << "...\n";
                    }

                    return [policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (requires { TinyDIP::getPlane(std::forward<ImageType>(img), 2); })
                            {
                                return TinyDIP::getPlane(std::forward<ImageType>(img), 2);
                            }
                            else
                            {
                                throw std::invalid_argument("Input image does not support multi-channel plane extraction.");
                                return {};
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::getPlane(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), 2); })
                            {
                                return TinyDIP::getPlane(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), 2);
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };
                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            )
        },
        CommandBundle{"getGplane", "Extract the Green plane (channel 1) from a multi-channel image.", TransformerSchema, 
            make_meta_transform_handler<2>(
                "getGplane [execution_policy] <input_img | $var> <output_img | $var>", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Extracting Green plane (channel 1) of " << filtered_args[0] << " (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Extracting Green plane (channel 1) of " << filtered_args[0] << "...\n";
                    }

                    return [policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (requires { TinyDIP::getPlane(std::forward<ImageType>(img), 1); })
                            {
                                return TinyDIP::getPlane(std::forward<ImageType>(img), 1);
                            }
                            else
                            {
                                throw std::invalid_argument("Input image does not support multi-channel plane extraction.");
                                return {};
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::getPlane(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), 1); })
                            {
                                return TinyDIP::getPlane(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), 1);
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };

                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            )
        },
        CommandBundle{"getRplane", "Extract the Red plane (channel 0) from a multi-channel image.", TransformerSchema, 
            make_meta_transform_handler<2>(
                "getRplane [execution_policy] <input_img | $var> <output_img | $var>", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Extracting Red plane (channel 0) of " << filtered_args[0] << " (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Extracting Red plane (channel 0) of " << filtered_args[0] << "...\n";
                    }

                    return [policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (requires { TinyDIP::getPlane(std::forward<ImageType>(img), 0); })
                            {
                                return TinyDIP::getPlane(std::forward<ImageType>(img), 0);
                            }
                            else
                            {
                                throw std::invalid_argument("Input image does not support multi-channel plane extraction.");
                                return {};
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::getPlane(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), 0); })
                            {
                                return TinyDIP::getPlane(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), 0);
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };

                        if (policy_str == "par")
                        {
                            return exec_policy(std::execution::par);
                        }
                        else if (policy_str == "par_unseq")
                        {
                            return exec_policy(std::execution::par_unseq);
                        }
                        else if (policy_str == "unseq")
                        {
                            return exec_policy(std::execution::unseq);
                        }
                        else if (policy_str == "seq")
                        {
                            return exec_policy(std::execution::seq);
                        }
                        else
                        {
                            return exec_default();
                        }
                    };
                }
            )
        },
        CommandBundle{"hsv2rgb", "Convert an HSV image or container to RGB color space.", TransformerSchema,
            make_meta_transform_handler<2, master_data_types>(
                "hsv2rgb [execution_policy] <input_data | $var> <output_var | $var>",
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Converting " << filtered_args[0] << " to RGB (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Converting " << filtered_args[0] << " to RGB...\n";
                    }

                    return [policy_str, &os]<typename DataT>(DataT && data) -> std::any
                    {
                        using DecayedDataT = std::remove_cvref_t<DataT>;

                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                            {
                                if constexpr (requires { TinyDIP::hsv2rgb(std::forward<DataT>(data)); })
                                {
                                    return TinyDIP::hsv2rgb(std::forward<DataT>(data));
                                }
                                else
                                {
                                    throw std::invalid_argument("Input image type does not support hsv2rgb conversion.");
                                    return {};
                                }
                            }
                            else if constexpr (std::ranges::input_range<DecayedDataT>)
                            {
                                if constexpr (requires { TinyDIP::hsv2rgb(*std::ranges::begin(data)); })
                                {
                                    return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                        [](auto&& element)
                                        {
                                            return TinyDIP::hsv2rgb(std::forward<decltype(element)>(element));
                                        },
                                        std::forward<DataT>(data)
                                    );
                                }
                                else
                                {
                                    throw std::invalid_argument("Input container type does not support hsv2rgb conversion.");
                                    return {};
                                }
                            }
                            else
                            {
                                throw std::invalid_argument("Input data type does not support hsv2rgb conversion.");
                                return {};
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                            {
                                if constexpr (requires { TinyDIP::hsv2rgb(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                                {
                                    return TinyDIP::hsv2rgb(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
                                }
                                else
                                {
                                    if (!std::ranges::empty(policy_str))
                                    {
                                        os << "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n";
                                    }
                                    return exec_default();
                                }
                            }
                            else if constexpr (std::ranges::input_range<DecayedDataT>)
                            {
                                if constexpr (requires { TinyDIP::hsv2rgb(*std::ranges::begin(data)); })
                                {
                                    return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                        std::forward<ExecPolicy>(exec_policy),
                                        [](auto&& element)
                                        {
                                            return TinyDIP::hsv2rgb(std::forward<decltype(element)>(element));
                                        },
                                        std::forward<DataT>(data)
                                    );
                                }
                                else
                                {
                                    if (!std::ranges::empty(policy_str))
                                    {
                                        os << "Warning: Execution policy requested but not supported for this data type/operation. Falling back to default.\n";
                                    }
                                    return exec_default();
                                }
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this data type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };

                        if (policy_str == "par")
                        {
                            return exec_policy(std::execution::par);
                        }
                        else if (policy_str == "par_unseq")
                        {
                            return exec_policy(std::execution::par_unseq);
                        }
                        else if (policy_str == "unseq")
                        {
                            return exec_policy(std::execution::unseq);
                        }
                        else if (policy_str == "seq")
                        {
                            return exec_policy(std::execution::seq);
                        }
                        else
                        {
                            return exec_default();
                        }
                    };
                }
            )
        },
        CommandBundle{"idct2", "Calculate Inverse Discrete Cosine Transformation for an image.", TransformerSchema, 
            make_meta_transform_handler<2>(
                "idct2 [execution_policy] <input_img | $var> <output_img | $var>", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Calculating Inverse DCT-2 for " << filtered_args[0] << " (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Calculating Inverse DCT-2 for " << filtered_args[0] << "...\n";
                    }

                    return [policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            return TinyDIP::idct2(std::forward<ImageType>(img));
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::idct2(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img)); })
                            {
                                return TinyDIP::idct2(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img));
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };

                        if (policy_str == "par") return exec_policy(std::execution::par);
                        else if (policy_str == "par_unseq") return exec_policy(std::execution::par_unseq);
                        else if (policy_str == "unseq") return exec_policy(std::execution::unseq);
                        else if (policy_str == "seq") return exec_policy(std::execution::seq);
                        else return exec_default();
                    };
                }
            )
        },
        CommandBundle{"info", "Display basic information about an image.", TerminatorSchema, InfoHandler{workspace}},
        CommandBundle{"lanczos_resample", "Resize an image using Lanczos resampling.", TransformerSchema, 
            make_meta_transform_handler<4>(
                "lanczos_resample [execution_policy] <input_img | $var> <output_img | $var> <width> <height> [a=3]", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    const std::size_t width = parse_arg<std::size_t>(filtered_args[2]);
                    const std::size_t height = parse_arg<std::size_t>(filtered_args[3]);
                    std::size_t a = 3;
                    
                    if (std::ranges::size(filtered_args) >= 5)
                    {
                        a = parse_arg<std::size_t>(filtered_args[4]);
                    }

                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Resizing " << filtered_args[0] << " to " << width << "x" << height << " with Lanczos radius " << a << " (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Resizing " << filtered_args[0] << " to " << width << "x" << height << " with Lanczos radius " << a << "...\n";
                    }

                    return [width, height, a, policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            return TinyDIP::lanczos_resample(std::forward<ImageType>(img), width, height, static_cast<int>(a));
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::lanczos_resample(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height, static_cast<int>(a)); })
                                return TinyDIP::lanczos_resample(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height, static_cast<int>(a));
                            else
                                return exec_default();
                        };
                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            )
        },
        CommandBundle{"load_workspace", "Load memory variables from a directory bundle.", IndependentSchema, LoadWorkspaceHandler{workspace}},
        CommandBundle{"max", "Calculate the maximum value of an image or container.", TransformerSchema, 
            make_meta_scalar_handler<1>(
                "max <input_data | $var> [output_var | $var]", 
                "max", "Max", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Warning: Execution policy '" << policy_str << "' is ignored for 'max'.\n";
                    }
                    os << "Calculating max of " << filtered_args[0] << "...\n";

                    return []<typename DataT>(DataT&& data) -> std::any
                    {
                        if constexpr (requires { TinyDIP::max(std::forward<DataT>(data)); })
                        {
                            return TinyDIP::max(std::forward<DataT>(data));
                        }
                        else
                        {
                            return std::ranges::max(std::forward<DataT>(data));
                        }
                    };
                }
            )
        },
        CommandBundle{"min", "Calculate the minimum value of an image or container.", TransformerSchema, 
            make_meta_scalar_handler<1>(
                "min <input_data | $var> [output_var | $var]", 
                "min", "Min", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Warning: Execution policy '" << policy_str << "' is ignored for 'min'.\n";
                    }
                    os << "Calculating min of " << filtered_args[0] << "...\n";

                    return []<typename DataT>(DataT&& data) -> std::any
                    {
                        if constexpr (requires { TinyDIP::min(std::forward<DataT>(data)); })
                        {
                            return TinyDIP::min(std::forward<DataT>(data));
                        }
                        else
                        {
                            return std::ranges::min(std::forward<DataT>(data));
                        }
                    };
                }
            )
        },
        CommandBundle{"print", "Print the contents of a memory variable.", TerminatorSchema, PrintHandler{workspace}},
        CommandBundle{"rand", "Generate random multi-dimensional image with specified URBG.", GeneratorSchema, RandHandler{workspace}},
        CommandBundle{"read", "Read an image from disk into a memory variable.", GeneratorSchema, ReadHandler{workspace}},
        CommandBundle{"remove", "Remove memory variables from the workspace (or 'all' to clear).", IndependentSchema, RemoveHandler{workspace}},
        CommandBundle{"rename", "Rename a memory variable in the workspace.", IndependentSchema, RenameHandler{workspace}},
        CommandBundle{"rgb2hsv", "Convert an RGB image or container to HSV color space.", TransformerSchema,
            make_meta_transform_handler<2, master_data_types>(
                "rgb2hsv [execution_policy] <input_data | $var> <output_var | $var>",
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Converting " << filtered_args[0] << " to HSV (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Converting " << filtered_args[0] << " to HSV...\n";
                    }

                    return[policy_str, &os]<typename DataT>(DataT && data) -> std::any
                    {
                        using DecayedDataT = std::remove_cvref_t<DataT>;

                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (requires { TinyDIP::rgb2hsv(std::forward<DataT>(data)); })
                            {
                                return TinyDIP::rgb2hsv(std::forward<DataT>(data));
                            }
                            else if constexpr (std::ranges::input_range<DecayedDataT> && requires { TinyDIP::rgb2hsv(*std::ranges::begin(data)); })
                            {
                                return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                    [](auto&& element)
                                    {
                                        return TinyDIP::rgb2hsv(std::forward<decltype(element)>(element));
                                    },
                                    std::forward<DataT>(data)
                                );
                            }
                            else
                            {
                                throw std::invalid_argument("Input data type does not support rgb2hsv conversion.");
                                return {};
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::rgb2hsv(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                            {
                                return TinyDIP::rgb2hsv(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
                            }
                            else if constexpr (std::ranges::input_range<DecayedDataT> && requires { TinyDIP::rgb2hsv(*std::ranges::begin(data)); })
                            {
                                return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                    std::forward<ExecPolicy>(exec_policy),
                                    [](auto&& element)
                                    {
                                        return TinyDIP::rgb2hsv(std::forward<decltype(element)>(element));
                                    },
                                    std::forward<DataT>(data)
                                );
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this data type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };

                        if (policy_str == "par")
                        {
                            return exec_policy(std::execution::par);
                        }
                        else if (policy_str == "par_unseq")
                        {
                            return exec_policy(std::execution::par_unseq);
                        }
                        else if (policy_str == "unseq")
                        {
                            return exec_policy(std::execution::unseq);
                        }
                        else if (policy_str == "seq")
                        {
                            return exec_policy(std::execution::seq);
                        }
                        else
                        {
                            return exec_default();
                        }
                    };
                }
            )
        },
        CommandBundle{"save_workspace", "Save all memory variables to a directory bundle.", IndependentSchema, SaveWorkspaceHandler{workspace}},
        CommandBundle{"sum", "Calculate the sum of all elements in an image or container.", TransformerSchema, 
            make_meta_scalar_handler<1>(
                "sum [execution_policy] <input_data | $var> [output_var | $var]", 
                "sum", "Sum", 
                workspace,
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Calculating sum of " << filtered_args[0] << " (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Calculating sum of " << filtered_args[0] << "...\n";
                    }

                    return [policy_str, &os]<typename DataT>(DataT&& img) -> std::any
                    {
                        // Helper to safely execute sum on the potentially casted image or generic container
                        auto process_sum_impl = [&]<typename T>(T&& actual_data) -> std::any
                        {
                            auto exec_default = [&]() -> std::any
                            {
                                if constexpr (requires { TinyDIP::sum(std::forward<T>(actual_data)); })
                                {
                                    return TinyDIP::sum(std::forward<T>(actual_data));
                                }
                                else
                                {
                                    using ValT = std::ranges::range_value_t<T>;
                                    return std::accumulate(std::ranges::begin(actual_data), std::ranges::end(actual_data), ValT{});
                                }
                            };

                            auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                                requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                            {
                                if constexpr (requires { TinyDIP::sum(std::forward<ExecPolicy>(exec_policy), std::forward<T>(actual_data)); })
                                {
                                    return TinyDIP::sum(std::forward<ExecPolicy>(exec_policy), std::forward<T>(actual_data));
                                }
                                else if constexpr (requires { std::reduce(std::forward<ExecPolicy>(exec_policy), std::ranges::begin(actual_data), std::ranges::end(actual_data)); })
                                {
                                    return std::reduce(std::forward<ExecPolicy>(exec_policy), std::ranges::begin(actual_data), std::ranges::end(actual_data));
                                }
                                else
                                {
                                    if (!std::ranges::empty(policy_str))
                                    {
                                        os << "Warning: Execution policy requested but not supported for this data type/operation. Falling back to default.\n";
                                    }
                                    return exec_default();
                                }
                            };

                            if (policy_str == "par") return exec_policy(std::execution::par);
                            else if (policy_str == "par_unseq") return exec_policy(std::execution::par_unseq);
                            else if (policy_str == "unseq") return exec_policy(std::execution::unseq);
                            else if (policy_str == "seq") return exec_policy(std::execution::seq);
                            else return exec_default();
                        };

                        if constexpr (std::same_as<std::remove_cvref_t<DataT>, TinyDIP::Image<TinyDIP::RGB>>)
                        {
                            // Explicitly cast to RGB_DOUBLE prior to calling process_sum_impl to prevent internal summation overflow
                            return process_sum_impl(TinyDIP::im2double(std::forward<DataT>(img)));
                        }
                        else
                        {
                            return process_sum_impl(std::forward<DataT>(img));
                        }
                    };
                }
            )
        },
        CommandBundle{"vars", "List all currently allocated memory variables.", IndependentSchema, VarsHandler{workspace}},
        CommandBundle{"write", "Write a memory variable out to a disk file.", TerminatorSchema, WriteHandler{workspace}}
    );

    // Register the help command dynamically to ensure it has access to the final mapped registry
    registry.register_command("help", "List all available commands.", IndependentSchema, HelpHandler{registry});

    if (argc < 2)
    {
        run_interactive_mode(registry);
        return EXIT_SUCCESS;
    }

    std::string command = argv[1];
    std::vector<std::string_view> args;

    if (argc > 2)
    {
        args.reserve(argc - 2);
        for (int i = 2; i < argc; ++i)
        {
            args.emplace_back(argv[i]);
        }
    }

    registry.execute(command, args);

    return EXIT_SUCCESS;
}

#endif


void test()
{
    constexpr int dims = 5;
    std::vector<std::string> test_vector1{ "1", "4", "7" };
    auto test1 = TinyDIP::n_dim_vector_generator<dims>(test_vector1, 3);
    std::vector<std::string> test_vector2{ "2", "5", "8" };
    auto test2 = TinyDIP::n_dim_vector_generator<dims>(test_vector2, 3);
    std::vector<std::string> test_vector3{ "3", "6", "9" };
    auto test3 = TinyDIP::n_dim_vector_generator<dims>(test_vector3, 3);
    std::vector<std::string> test_vector4{ "a", "b", "c" };
    auto test4 = TinyDIP::n_dim_vector_generator<dims>(test_vector4, 3);
    auto output = TinyDIP::recursive_transform<dims + 1>(
        [](auto element1, auto element2, auto element3, auto element4) { return element1 + element2 + element3 + element4; },
        test1, test2, test3, test4);
    std::cout << typeid(output).name() << std::endl;
    TinyDIP::recursive_print(output
    .at(0).at(0).at(0).at(0).at(0));
    return;   
}

//  bicubicInterpolationTest function implementation
void bicubicInterpolationTest()
{
    TinyDIP::Image<TinyDIP::GrayScale> image1(3, 3);
    image1.setAllValue(1);
    std::cout << "Width: " + std::to_string(image1.getWidth()) + "\n";
    std::cout << "Height: " + std::to_string(image1.getHeight()) + "\n";
    image1.at(static_cast<std::size_t>(1), static_cast<std::size_t>(1)) = 100;
    image1.print();

    auto image2 = TinyDIP::copyResizeBicubic(image1, 12, 12);
    image2.print();
}

void addLeadingZeros(std::string input_path, std::string output_path)
{
    for (std::size_t i = 1; i <= 96; ++i)
    {
        std::string filename = input_path + std::to_string(i) + ".bmp";
        std::cout << filename << "\n";
        auto bmpimage = TinyDIP::bmp_read(filename.c_str(), true);
        char buff[100];
        snprintf(buff, sizeof(buff), "%s%05ld", output_path.c_str(), i);
        TinyDIP::bmp_write(buff, bmpimage);
    }
}
