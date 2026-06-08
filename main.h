#ifndef TINYDIP_MAIN_H
#define TINYDIP_MAIN_H

//  Standard Library Headers
#include <algorithm>
#include <any>
#include <array>
#include <charconv>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <execution>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <numbers>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <vector>

//  Local Headers
#include "basic_functions.h"
#include "image_io.h"
#include "image_operations.h"
#include "timer.h"

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

// gaussian_params_t helper alias
// gaussian_params_t is a helper alias to bridge GaussianParameters2D's NTTP for tuple mapping.
// Helper alias to bridge GaussianParameters2D's NTTP for tuple mapping
template <typename T>
using gaussian_params_t = TinyDIP::GaussianParameters2D<T>;

// Exhaustive Derived Type Auto-Generation
using all_multichannel_types = tuple_map_t<multichannel_t, core_numeric_types>;
using all_complex_types = tuple_map_t<std::complex, core_floating_point_types>;
using all_complex_multichannel_types = tuple_map_t<multichannel_t, all_complex_types>;
using all_vector_types = tuple_map_t<std::vector, core_numeric_types>;
using all_deque_types = tuple_map_t<std::deque, core_numeric_types>;
using all_list_types = tuple_map_t<std::list, core_numeric_types>;
using all_array_types = generate_arrays_t<core_numeric_types, 3, 4>;
using all_custom_scalar_types = std::tuple<TinyDIP::RGB, TinyDIP::RGB_DOUBLE, TinyDIP::HSV>;
using all_gaussian_params_types = tuple_map_t<gaussian_params_t, core_floating_point_types>;

// Master Scalar Tuple (Exhaustively includes ALL valid scalar and container output types)
using master_scalar_types = tuple_cat_t<
    core_numeric_types,
    all_custom_scalar_types,
    all_multichannel_types,
    all_complex_types,
    all_complex_multichannel_types,
    all_vector_types,
    all_deque_types,
    all_list_types,
    all_array_types,
    all_gaussian_params_types
>;

// Master Image Tuple (Exhaustively includes ALL valid image structures)
using master_image_types = tuple_cat_t<
    tuple_map_t<image_t, core_numeric_types>,
    tuple_map_t<image_t, all_custom_scalar_types>,
    tuple_map_t<image_t, all_multichannel_types>,
    tuple_map_t<image_t, all_complex_types>,
    tuple_map_t<image_t, all_complex_multichannel_types>
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
    all_complex_multichannel_types,
    all_vector_types,
    all_deque_types,
    all_list_types,
    all_array_types,
    all_gaussian_params_types
>;

using all_vector_image_types = tuple_map_t<std::vector, master_image_types>;
using all_deque_image_types = tuple_map_t<std::deque, master_image_types>;
using all_list_image_types = tuple_map_t<std::list, master_image_types>;

//  master_image_container_types type is used to identify all container types 
// that hold images, which allows the workspace listing function to apply special
// formatting logic for these types (e.g. printing count and first image size) without having to check each container type separately.
using master_image_container_types = tuple_cat_t<
    all_vector_image_types,
    all_deque_image_types,
    all_list_image_types
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



#endif //TINYDIP_MAIN_H