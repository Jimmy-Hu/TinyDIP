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



#endif //TINYDIP_MAIN_H