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

//  sanitize_string_view function implementation
//  Helper for safely sanitizing cross-platform path strings and REPL tokens natively with zero allocations
constexpr std::string_view sanitize_string_view(std::string_view sv)
{
    auto is_junk = [](unsigned char c) { return std::isspace(c) || c == '\r' || c == '\n' || c == '\"' || c == '\''; };
    while (!sv.empty() && is_junk(static_cast<unsigned char>(sv.back())))
    {
        sv.remove_suffix(1);
    }
    while (!sv.empty() && is_junk(static_cast<unsigned char>(sv.front())))
    {
        sv.remove_prefix(1);
    }
    return sv;
}


//  parse_arg template function implementation
//  Helper for converting string to numeric types safely
template <typename T>
T parse_arg(const std::string_view sv)
{
    const std::string_view clean_sv = sanitize_string_view(sv);
    T result{};
    if constexpr (std::is_arithmetic_v<T>)
    {
        auto [ptr, ec] = std::from_chars(clean_sv.data(), clean_sv.data() + std::ranges::size(clean_sv), result);
        if (ec != std::errc())
        {
            throw std::invalid_argument(std::string("Error parsing argument: ") + std::string(clean_sv));
        }
    }
    else
    {
        //  Fallback for non-arithmetic types (unlikely to be used with this function in current context)
        //  This path forces allocation, but is rarely hit for numeric parsing
        std::string temp(clean_sv);
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
    mutable std::mutex mtx;

    template <typename T>
    void store(const std::string_view name, T&& item)
    {
        std::lock_guard<std::mutex> lock(mtx);
        memory_store[std::string(sanitize_string_view(name))] = std::forward<T>(item);
    }

    template <typename T>
    const T* retrieve(const std::string_view name) const
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (auto it = memory_store.find(std::string(sanitize_string_view(name))); it != std::ranges::end(memory_store))
        {
            if (it->second.type() == typeid(T))
            {
                return std::any_cast<T>(&(it->second));
            }
        }
        return nullptr;
    }

    //  retrieve_any function implementation
    std::optional<std::any> retrieve_any(const std::string_view name) const
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (auto it = memory_store.find(std::string(sanitize_string_view(name))); it != std::ranges::end(memory_store))
        {
            return it->second;
        }
        return std::nullopt;
    }

    //  remove function implementation
    bool remove(const std::string_view name)
    {
        std::lock_guard<std::mutex> lock(mtx);
        const std::string key = std::string(sanitize_string_view(name));
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
        std::lock_guard<std::mutex> lock(mtx);
        const std::string old_key(sanitize_string_view(old_name));
        if (auto it = memory_store.find(old_key); it != std::ranges::end(memory_store))
        {
            //  Use std::move to natively transfer ownership of the type-erased object with zero-copy
            memory_store[std::string(sanitize_string_view(new_name))] = std::move(it->second);
            memory_store.erase(it);
            return true;
        }
        return false;
    }

    //  Clear all elements in the workspace memory store
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        memory_store.clear();
    }

    //  list_variables function implementation
    void list_variables(std::ostream& os) const
    {
        std::lock_guard<std::mutex> lock(mtx);
        
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

            // Polymorphic lambda returning true if the image container type matched
            auto try_print_image_container = [&]<typename T>() -> bool
            {
                if (value.type() == typeid(T))
                {
                    print_prefix.template operator()<T>();
                    const auto* container_ptr = std::any_cast<T>(&value);
                    os << ", count = " << std::ranges::size(*container_ptr);
                    
                    if (!std::ranges::empty(*container_ptr))
                    {
                        os << " (first image size: ";
                        print_size(std::ranges::begin(*container_ptr)->getSize());
                        os << ")";
                    }
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
            else if (match_any_type<master_image_container_types>(try_print_image_container))
            {
                // Handled successfully by try_print_image_container short-circuit logic
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
        constexpr ImageType operator()(const std::string_view arg, Workspace& ws) const
        {
            if (arg.starts_with('$'))
            {
                const std::string_view var_name = arg.substr(1);
                if (const ImageType* img_ptr = ws.retrieve<ImageType>(var_name))
                {
                    return *img_ptr;
                }
                throw std::invalid_argument(std::string("Memory variable not found or type mismatch: ") + std::string(var_name));
            }

            const std::string_view clean_arg = sanitize_string_view(arg);
            const std::filesystem::path input_path = std::string(clean_arg);
            const bool has_ext = input_path.has_extension();
            std::string ext{};
            if (has_ext)
            {
                ext = input_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            }

            if constexpr (std::is_same_v<ImageType, TinyDIP::Image<TinyDIP::RGB>>)
            {
                if (ext == ".ppm")
                {
                    return TinyDIP::pnm::read(input_path);
                }
                
                return TinyDIP::bmp_read(input_path.string().c_str(), has_ext);
            }
            else if constexpr (std::is_same_v<ImageType, TinyDIP::Image<double>>)
            {
                if (ext == ".csv")
                {
                    return TinyDIP::double_image::read_from_csv(input_path.string().c_str());
                }
                
                return TinyDIP::double_image::read(input_path.string().c_str(), has_ext);
            }
            else if constexpr (std::is_same_v<ImageType, TinyDIP::Image<TinyDIP::HSV>>)
            {
                return TinyDIP::hsv_read(input_path.string().c_str(), has_ext);
            }
            else
            {
                throw std::invalid_argument("Direct file reading is not explicitly implemented for this abstract/complex data type.");
            }
        }
    };

    struct Saver
    {
        template <typename ImageType>
        constexpr void operator()(const std::string_view arg, Workspace& ws, ImageType&& img) const
        {
            if (arg.starts_with('$'))
            {
                const std::string_view var_name = arg.substr(1);
                ws.store(var_name, std::forward<ImageType>(img));
            }
            else
            {
                const std::string_view clean_arg = sanitize_string_view(arg);
                const std::filesystem::path output_filepath = std::string(clean_arg);
                const bool has_ext = output_filepath.has_extension();
                std::string ext{};
                if (has_ext)
                {
                    ext = output_filepath.extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
                }
                const std::filesystem::path base_path = has_ext ? (output_filepath.parent_path() / output_filepath.stem()) : output_filepath;
                
                if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<double>>)
                {
                    if (ext == ".csv")
                    {
                        TinyDIP::double_image::write_to_csv(output_filepath.string().c_str(), std::forward<ImageType>(img));
                    }
                    else
                    {
                        TinyDIP::double_image::write(base_path.string().c_str(), std::forward<ImageType>(img));
                    }
                }
                else if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::RGB>>)
                {
                    if (ext == ".ppm")
                    {
                        TinyDIP::pnm::write(std::forward<ImageType>(img), output_filepath);
                    }
                    else
                    {
                        TinyDIP::bmp_write(base_path.string().c_str(), std::forward<ImageType>(img));
                    }
                }
                else if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::HSV>>)
                {
                    TinyDIP::hsv_write(base_path.string().c_str(), std::forward<ImageType>(img));
                }
                else
                {
                    throw std::invalid_argument("Direct file writing is not explicitly implemented for this abstract/complex data type.");
                }
            }
        }
    };
};

//  dispatch_data_operation template function implementation
//  Generic helper to dynamically load and dispatch data (from memory or disk) to a processor lambda
template <typename CheckingTypes = master_image_types, typename ProcessorFun, typename ImageLoaderFun>
requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&> &&
          std::invocable<ProcessorFun, std::invoke_result_t<ImageLoaderFun, const std::string_view, Workspace&>>)
constexpr bool dispatch_data_operation(
    const std::string_view input_arg,
    Workspace& workspace,
    ImageLoaderFun&& image_loader,
    ProcessorFun&& processor)
{
    if (input_arg.starts_with('$'))
    {
        const std::string_view var_name = input_arg.substr(1);

        auto try_process = [&]<typename T>() -> bool
        {
            if (workspace.template retrieve<T>(var_name))
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
        const std::string_view clean_arg = sanitize_string_view(input_arg);
        const std::filesystem::path input_path = std::string(clean_arg);
        std::string ext{};
        if (input_path.has_extension())
        {
            ext = input_path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        }

        if (ext == ".dbmp" || ext == ".csv")
        {
            processor(image_loader.template operator()<TinyDIP::Image<double>>(clean_arg, workspace));
        }
        else if (ext == ".hsv")
        {
            processor(image_loader.template operator()<TinyDIP::Image<TinyDIP::HSV>>(clean_arg, workspace));
        }
        else
        {
            processor(image_loader.template operator()<TinyDIP::Image<TinyDIP::RGB>>(clean_arg, workspace));
        }
        
        return true;
    }
}


//  QueuedCommand struct implementation
struct QueuedCommand
{
    std::string name;
    std::vector<std::string> args;
};

//  match_any template function implementation
//  Helper function to flawlessly check if a target string matches any element in an input container
template <std::ranges::input_range RangeT>
requires (std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string_view> or
          std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string> or
          std::convertible_to<std::ranges::range_value_t<RangeT>, std::string_view> or
          std::convertible_to<std::ranges::range_value_t<RangeT>, std::string>)
constexpr bool match_any(const std::string_view target, const RangeT& container)
{
    return std::ranges::find(std::ranges::begin(container), std::ranges::end(container), target) != std::ranges::end(container);
}

//  PeepholeOptimizer template class implementation
template <std::ranges::input_range RangeT>
requires std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, QueuedCommand>
class PeepholeOptimizer
{
public:
    static void optimize(RangeT& pipeline, std::ostream& os)
    {
        if (std::ranges::empty(pipeline))
        {
            return;
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            RangeT optimized_pipeline;
            
            if constexpr (std::ranges::sized_range<RangeT> && requires { optimized_pipeline.reserve(1); })
            {
                optimized_pipeline.reserve(std::ranges::size(pipeline));
            }

            auto it = std::ranges::begin(pipeline);
            const auto end = std::ranges::end(pipeline);

            std::optional<QueuedCommand> pending_cmd;

            auto push_to_optimized = [&](QueuedCommand&& cmd)
            {
                if constexpr (requires { optimized_pipeline.emplace_back(std::move(cmd)); })
                {
                    optimized_pipeline.emplace_back(std::move(cmd));
                }
                else
                {
                    optimized_pipeline.push_back(std::move(cmd));
                }
            };
            
            auto get_io = [](const QueuedCommand& c) -> std::pair<std::string, std::string>
            {
                if (c.name == "copy" && std::ranges::size(c.args) >= 2)
                {
                    return {c.args[0], c.args[1]};
                }
                if (c.name == "dct2" || c.name == "idct2" || c.name == "abs" || c.name == "normalize" || c.name == "multiply")
                {
                    bool has_policy = (std::ranges::size(c.args) > 0 && (c.args[0] == "seq" || c.args[0] == "par" || c.args[0] == "par_unseq" || c.args[0] == "unseq"));
                    std::size_t offset = has_policy ? 1 : 0;
                    if (std::ranges::size(c.args) > offset + 1)
                    {
                        return {c.args[offset], c.args[offset + 1]};
                    }
                }
                return {"", ""};
            };

            auto update_in = [](QueuedCommand& c, const std::string& new_in) -> bool
            {
                if (c.name == "copy" && std::ranges::size(c.args) >= 2)
                {
                    c.args[0] = new_in;
                    return true;
                }
                if (c.name == "dct2" || c.name == "idct2" || c.name == "abs" || c.name == "normalize" || c.name == "multiply")
                {
                    bool has_policy = (std::ranges::size(c.args) > 0 && (c.args[0] == "seq" || c.args[0] == "par" || c.args[0] == "par_unseq" || c.args[0] == "unseq"));
                    std::size_t offset = has_policy ? 1 : 0;
                    if (std::ranges::size(c.args) > offset + 1)
                    {
                        c.args[offset] = new_in;
                        return true;
                    }
                }
                return false;
            };

            auto update_out = [](QueuedCommand& c, const std::string& new_out) -> bool
            {
                if (c.name == "copy" && std::ranges::size(c.args) >= 2)
                {
                    c.args[1] = new_out;
                    return true;
                }
                if (c.name == "dct2" || c.name == "idct2" || c.name == "abs" || c.name == "normalize" || c.name == "multiply")
                {
                    bool has_policy = (std::ranges::size(c.args) > 0 && (c.args[0] == "seq" || c.args[0] == "par" || c.args[0] == "par_unseq" || c.args[0] == "unseq"));
                    std::size_t offset = has_policy ? 1 : 0;
                    if (std::ranges::size(c.args) > offset + 1)
                    {
                        c.args[offset + 1] = new_out;
                        return true;
                    }
                }
                return false;
            };

            while (it != end || pending_cmd.has_value())
            {
                QueuedCommand cmd1;
                if (pending_cmd.has_value())
                {
                    cmd1 = std::move(pending_cmd.value());
                    pending_cmd.reset();
                }
                else
                {
                    cmd1 = std::move(*it);
                    ++it;
                }

                // 1. Identify Redundant Scalar Math (Multiply by 1.0)
                if (cmd1.name == "multiply" && std::ranges::size(cmd1.args) >= 3)
                {
                    const std::string_view factor = cmd1.args.back();
                    const std::string_view clean_factor = sanitize_string_view(factor);
                    double factor_val = 0.0;
                    
                    auto [ptr, ec] = std::from_chars(clean_factor.data(), clean_factor.data() + std::ranges::size(clean_factor), factor_val);
                    
                    if (ec == std::errc() && factor_val == 1.0)
                    {
                        auto [cmd1_in, cmd1_out] = get_io(cmd1);
                        
                        os << "  [Peephole Optimizer] Detected Identity operation: 'multiply' by 1.0.\n";
                        os << "  [Peephole Optimizer] Downgrading to zero-overhead 'copy'.\n";
                        
                        QueuedCommand copy_cmd;
                        copy_cmd.name = "copy";
                        copy_cmd.args = {cmd1_in, cmd1_out};
                        cmd1 = std::move(copy_cmd);
                        changed = true;
                    }
                }

                // Try to fetch cmd2 to check pairs
                if (!pending_cmd.has_value() && it != end)
                {
                    QueuedCommand cmd2 = std::move(*it);
                    ++it;

                    auto [cmd1_in, cmd1_out] = get_io(cmd1);
                    auto [cmd2_in, cmd2_out] = get_io(cmd2);

                    if (!cmd1_out.empty() && cmd1_out == cmd2_in)
                    {
                        // A. Inverses
                        if ((cmd1.name == "dct2" && cmd2.name == "idct2") ||
                            (cmd1.name == "idct2" && cmd2.name == "dct2"))
                        {
                            os << "  [Peephole Optimizer] Detected redundant inverse chain: '" << cmd1.name << "' -> '" << cmd2.name << "'.\n";
                            os << "  [Peephole Optimizer] Collapsing chain into zero-overhead 'copy' (" << cmd1_in << " -> " << cmd2_out << ").\n";
                            
                            QueuedCommand copy_cmd;
                            copy_cmd.name = "copy";
                            copy_cmd.args = {cmd1_in, cmd2_out};
                            push_to_optimized(std::move(copy_cmd));
                            changed = true;
                            continue;
                        }

                        // B. Forward Copy Propagation
                        if (cmd1.name == "copy" && cmd1_out.find("$__pipe_") == 0)
                        {
                            if (update_in(cmd2, cmd1_in))
                            {
                                os << "  [Peephole Optimizer] Forwarding 'copy' input natively.\n";
                                pending_cmd = std::move(cmd2);
                                changed = true;
                                continue;
                            }
                        }

                        // C. Backward Copy Propagation
                        if (cmd2.name == "copy" && cmd2_in.find("$__pipe_") == 0)
                        {
                            if (update_out(cmd1, cmd2_out))
                            {
                                os << "  [Peephole Optimizer] Collapsing trailing 'copy' natively.\n";
                                pending_cmd = std::move(cmd1);
                                changed = true;
                                continue;
                            }
                        }
                    }

                    // No match -> keep cmd2 for next pass
                    pending_cmd = std::move(cmd2);
                }

                // Push cmd1 if it didn't merge
                push_to_optimized(std::move(cmd1));
            }

            pipeline = std::move(optimized_pipeline);
        }
    }
};

//  supports_standard_execution_policies concept definition
//  Concept to rigorously enforce that a polymorphic lambda supports all C++ standard execution policies
template <typename FunT>
concept supports_standard_execution_policies = 
    std::invocable<FunT, decltype(std::execution::seq)> &&
    std::invocable<FunT, decltype(std::execution::par)> &&
    std::invocable<FunT, decltype(std::execution::par_unseq)> &&
    std::invocable<FunT, decltype(std::execution::unseq)>;

//  dispatch_policy_string template function implementation
//  Helper to dispatch execution policy string to std::execution policies
template <supports_standard_execution_policies PolicyFun, std::invocable DefaultFun>
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

//  CommandHandler type alias definition
//  Modern C++ Standard Function signature for highly robust, state-injected execution
using CommandHandler = std::function<void(Workspace&, std::span<const std::string_view>, std::ostream&)>;

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
constexpr auto CombinerSchema = IOSchema{ 0, 2 };
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

    void execute(Workspace& workspace, const std::string& command_name, std::span<const std::string_view> args, std::ostream& os = std::cout) const
    {
        if (auto it = commands.find(command_name); it != std::ranges::end(commands))
        {
            try
            {
                it->second.handler(workspace, args, os);
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
//  Generic Meta Handler strictly refactoring transform commands like abs, bicubic_resize, dct2, dct3, idct2, and lanczos_resample
template <
    std::size_t MinArgs,
    typename SetupFun,
    typename ArgsContainer = std::vector<std::string_view>,
    typename CheckingTypes = master_image_types
>
requires(std::invocable<SetupFun, const ArgsContainer&, const std::string_view, std::ostream&>)
struct MetaTransformHandler
{
    std::string_view usage_string_;
    SetupFun setup_fun_;

    template <
        typename ImageLoaderFun = MetaImageIO::Loader,
        typename ImageSaverFun = MetaImageIO::Saver
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&> &&
              std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<double>&&>)
    constexpr void operator()(Workspace& workspace, std::span<const std::string_view> args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        std::string_view policy_str = "";
        ArgsContainer filtered_args;
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

        if (!dispatch_data_operation<CheckingTypes>(input_arg, workspace, image_loader_fun, process_wrapper))
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
                    image_saver_fun(output_arg, workspace, std::move(std::any_cast<OutT&>(output_any)));
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
template <std::size_t MinArgs, typename CheckingTypes = master_image_types, typename SetupFun, typename ArgsContainer = std::vector<std::string_view>>
requires(std::invocable<SetupFun, ArgsContainer, const std::string_view, std::ostream&>)
constexpr auto make_meta_transform_handler(std::string_view usage, SetupFun&& setup)
{
    return MetaTransformHandler<MinArgs, std::remove_cvref_t<SetupFun>, ArgsContainer, CheckingTypes>{
        usage, std::forward<SetupFun>(setup)
    };
}

namespace handlers
{
    //  dct3 function implementation
    constexpr void dct3(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2, master_image_container_types>(
            "dct3 [execution_policy] <input_container | $var> <output_container | $var>",
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                os << "Calculating DCT-3 for " << filtered_args[0];
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [policy_str, &os]<typename ImageType>(ImageType && img) -> std::any
                {
                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (requires { TinyDIP::dct3(std::forward<ImageType>(img)); })
                        {
                            return TinyDIP::dct3(std::forward<ImageType>(img));
                        }
                        else
                        {
                            throw std::invalid_argument("Input image container type does not support dct3 operation.");
                        }
                        return std::any{};
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (requires { TinyDIP::dct3(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img)); })
                        {
                            return TinyDIP::dct3(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img));
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
        );

        transform_handler(workspace, args, os);
    }

    //  copy function implementation
    constexpr void copy(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: copy <$src_var> <$dst_var>\n";
            return;
        }

        const std::string_view src_arg = args[0];
        const std::string_view dst_arg = args[1];

        if (!src_arg.starts_with('$') || !dst_arg.starts_with('$'))
        {
            os << "Error: Both arguments must be memory variables starting with '$'.\n";
            return;
        }

        const std::string_view src_name = src_arg.substr(1);
        const std::string_view dst_name = dst_arg.substr(1);

        auto val_opt = workspace.retrieve_any(src_name);
        if (val_opt.has_value())
        {
            // std::any naturally performs a safe deep copy of the underlying type erased object
            workspace.store(dst_name, val_opt.value());
            os << "Copied variable $" << src_name << " to $" << dst_name << ".\n";
        }
        else
        {
            os << "Error: Memory variable $" << src_name << " not found.\n";
        }
    }

    //  erase_element template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void erase_element(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        if (std::ranges::size(args) < 3)
        {
            os << "Usage: erase_element <input_container | $var> <output_var | $var> <index>\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];

        if (!output_arg.starts_with('$'))
        {
            os << "Error: Output must be a memory variable starting with '$'.\n";
            return;
        }

        const std::size_t index = parse_arg<std::size_t>(args[2]);

        os << "Erasing element at index " << index << " from " << input_arg << "...\n";

        auto process_erase = [&]<typename CandidateType>(CandidateType&& candidate)
        {
            using DecayedT = std::remove_cvref_t<CandidateType>;

            // Mathematically verify that the type is a registered container that supports erase
            if constexpr (is_vector_v<DecayedT> || is_deque_v<DecayedT> || is_list_v<DecayedT>)
            {
                if (index >= std::ranges::size(candidate))
                {
                    os << "Error: Index " << index << " is out of bounds for container of size " << std::ranges::size(candidate) << ".\n";
                    return;
                }

                DecayedT new_container = candidate; // Copy original container completely independently
                
                // Natively advance iterator and execute erase to dynamically resize the sequence
                auto it = std::ranges::begin(new_container);
                std::ranges::advance(it, index);
                new_container.erase(it);
                
                workspace.store(output_arg.substr(1), std::move(new_container));
                os << "Saved updated container to " << output_arg << ".\n";
            }
            else
            {
                os << "Error: Input type [" << get_type_name<DecayedT>() << "] is not a supported container type that supports erasing.\n";
            }
        };

        // Leverage tuple_cat_t to flawlessly support both scalar containers and newly registered image containers!
        using AllContainerTypes = tuple_cat_t<master_data_types, master_image_container_types>;

        if (!dispatch_data_operation<AllContainerTypes>(input_arg, workspace, image_loader_fun, process_erase))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }

    //  load_workspace function implementation
    constexpr void load_workspace(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        if (std::ranges::size(args) < 1)
        {
            os << "Usage: load_workspace <directory_bundle_path>\n";
            return;
        }

        const std::string_view clean_arg = sanitize_string_view(args[0]);
        const std::filesystem::path dir_path = std::string(clean_arg);

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
                    workspace.store(name, std::move(img));
                    os << "  Loaded " << entry.path().filename().string() << " -> $" << name << "\n";
                }
                else if (ext == ".dbmp")
                {
                    auto img = TinyDIP::double_image::read(entry.path().string().c_str(), true);
                    workspace.store(name, std::move(img));
                    os << "  Loaded " << entry.path().filename().string() << " -> $" << name << "\n";
                }
                else if (ext == ".hsv")
                {
                    auto img = TinyDIP::hsv_read(entry.path().string().c_str(), true);
                    workspace.store(name, std::move(img));
                    os << "  Loaded " << entry.path().filename().string() << " -> $" << name << "\n";
                }
                else if (ext == ".csv")
                {
                    auto img = TinyDIP::double_image::read_from_csv(entry.path().string().c_str());
                    workspace.store(name, std::move(img));
                    os << "  Loaded " << entry.path().filename().string() << " -> $" << name << "\n";
                }
                else if (ext == ".ppm")
                {
                    auto img = TinyDIP::pnm::read(entry.path());
                    workspace.store(name, std::move(img));
                    os << "  Loaded " << entry.path().filename().string() << " -> $" << name << "\n";
                }
            }
        }
        os << "Workspace loaded successfully.\n";
    }

	//  manhattan_distance template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void manhattan_distance(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        std::string_view policy_str = "";
        std::vector<std::string_view> filtered_args{};
        filtered_args.reserve(std::ranges::size(args));

        for (const auto& arg : args)
        {
            if (arg == "seq" || arg == "par" || arg == "par_unseq" || arg == "unseq")
            {
                policy_str = arg;
            }
            else
            {
                filtered_args.emplace_back(arg);
            }
        }

        if (std::ranges::size(filtered_args) < 2)
        {
            os << "Usage: manhattan_distance [execution_policy] <input1_data | $var> <input2_data | $var> [output_var | $var]\n";
            return;
        }

        const std::string_view input1_arg = filtered_args[0];
        const std::string_view input2_arg = filtered_args[1];
        std::string_view output_arg = "";
        
        if (std::ranges::size(filtered_args) > 2)
        {
            output_arg = filtered_args[2];
        }

        os << "Calculating Manhattan distance between " << input1_arg << " and " << input2_arg;
        if (!std::ranges::empty(policy_str))
        {
            os << " (Policy: " << policy_str << ")";
        }
        os << "...\n";

        auto process_input1 = [&]<typename Data1T>(Data1T&& data1)
        {
            auto process_input2 = [&]<typename Data2T>(Data2T&& data2)
            {
                using Decayed1T = std::remove_cvref_t<Data1T>;
                using Decayed2T = std::remove_cvref_t<Data2T>;

                auto exec_default = [&]() -> std::any
                {
                    if constexpr (requires { TinyDIP::manhattan_distance(std::forward<Data1T>(data1), std::forward<Data2T>(data2)); })
                    {
                        return TinyDIP::manhattan_distance(std::forward<Data1T>(data1), std::forward<Data2T>(data2));
                    }
                    else
                    {
                        throw std::invalid_argument(std::string("Input types [") + std::string(get_type_name<Decayed1T>()) + "] and [" + std::string(get_type_name<Decayed2T>()) + "] do not support manhattan_distance.");
                        return std::any{};
                    }
                };

                auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                    requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                {
                    if constexpr (requires { TinyDIP::manhattan_distance(std::forward<ExecPolicy>(exec_policy), std::forward<Data1T>(data1), std::forward<Data2T>(data2)); })
                    {
                        return TinyDIP::manhattan_distance(std::forward<ExecPolicy>(exec_policy), std::forward<Data1T>(data1), std::forward<Data2T>(data2));
                    }
                    else
                    {
                        if (!std::ranges::empty(policy_str))
                        {
                            os << "Warning: Execution policy requested but not supported for these data types/operation. Falling back to default.\n";
                        }
                        return exec_default();
                    }
                };

                try
                {
                    std::any result = dispatch_policy_string(policy_str, exec_policy, exec_default, os);

                    // std::any{} indicates an error occurred and was caught safely in the default blocks
                    if (!result.has_value())
                    {
                        return;
                    }

                    bool handled = false;
                    auto handle_result = [&]<typename ScalarT>() -> bool
                    {
                        if (result.type() == typeid(ScalarT))
                        {
                            auto& scalar_result = std::any_cast<ScalarT&>(result);
                            if (!std::ranges::empty(output_arg))
                            {
                                if (output_arg.starts_with('$'))
                                {
                                    workspace.store(output_arg.substr(1), scalar_result);
                                    os << "Saved Manhattan Distance result to " << output_arg << "\n";
                                }
                                else
                                {
                                    os << "Error: Output must be a memory variable starting with '$'.\n";
                                }
                            }
                            else
                            {
                                if constexpr (is_vector_v<ScalarT> || is_deque_v<ScalarT> || is_list_v<ScalarT> || is_std_array_v<ScalarT>)
                                {
                                    os << "Manhattan Distance result: {";
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
                                else if constexpr (requires { os << scalar_result; })
                                {
                                    if constexpr (sizeof(ScalarT) == 1 && std::is_integral_v<ScalarT>)
                                    {
                                        os << "Manhattan Distance result: " << +scalar_result << "\n";
                                    }
                                    else
                                    {
                                        os << "Manhattan Distance result: " << scalar_result << "\n";
                                    }
                                }
                                else
                                {
                                    os << "Manhattan Distance result evaluated successfully (Non-printable complex type).\n";
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
                           << result.type().name() << "]\n";
                    }
                }
                catch (const std::exception& e)
                {
                    os << "Error calculating manhattan_distance: " << e.what() << '\n';
                }
            };

            if (!dispatch_data_operation<master_data_types>(input2_arg, workspace, image_loader_fun, process_input2))
            {
                os << "Error: Memory variable not found or unsupported type for input2: " << input2_arg << "\n";
            }
        };

        if (!dispatch_data_operation<master_data_types>(input1_arg, workspace, image_loader_fun, process_input1))
        {
            os << "Error: Memory variable not found or unsupported type for input1: " << input1_arg << "\n";
        }
    }

    //  normalize function implementation
    constexpr void normalize(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2, master_data_types>(
            "normalize [execution_policy] <input_data | $var> <output_var | $var>", 
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                os << "Normalizing " << filtered_args[0];
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [policy_str, &os]<typename DataT>(DataT && data) -> std::any
                {
                    using DecayedDataT = std::remove_cvref_t<DataT>;

                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            if constexpr (requires { TinyDIP::normalize(std::forward<DataT>(data)); })
                            {
                                return TinyDIP::normalize(std::forward<DataT>(data));
                            }
                            else
                            {
                                throw std::invalid_argument("Input image type does not support normalize operation.");
                                return std::any{};
                            }
                        }
                        else if constexpr (std::ranges::input_range<DecayedDataT>)
                        {
                            return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                [](auto&& element) 
                                { 
                                    if constexpr (requires { TinyDIP::normalize(std::forward<decltype(element)>(element)); })
                                    {
                                        return TinyDIP::normalize(std::forward<decltype(element)>(element));
                                    }
                                    else
                                    {
                                        throw std::invalid_argument("Input container element type does not support normalize operation.");
                                        using RetT = std::remove_cvref_t<decltype(element)>;
                                        return RetT{};
                                    }
                                },
                                std::forward<DataT>(data)
                            );
                        }
                        else
                        {
                            if constexpr (requires { TinyDIP::normalize(std::forward<DataT>(data)); })
                            {
                                return TinyDIP::normalize(std::forward<DataT>(data));
                            }
                            else
                            {
                                throw std::invalid_argument("Input data type does not support normalize operation.");
                                return std::any{};
                            }
                        }
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            if constexpr (requires { TinyDIP::normalize(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                            {
                                return TinyDIP::normalize(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
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
                            return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                std::forward<ExecPolicy>(exec_policy),
                                [](auto&& element) 
                                { 
                                    if constexpr (requires { TinyDIP::normalize(std::forward<decltype(element)>(element)); })
                                    {
                                        return TinyDIP::normalize(std::forward<decltype(element)>(element));
                                    }
                                    else
                                    {
                                        throw std::invalid_argument("Input container element type does not support normalize operation.");
                                        using RetT = std::remove_cvref_t<decltype(element)>;
                                        return RetT{};
                                    }
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

                    return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                };
            }
        );

        transform_handler(workspace, args, os);
    }

	//  read template function implementation
    template <
        std::invocable<const std::string_view, Workspace&> ImageLoaderFun = MetaImageIO::Loader,
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::RGB>&&> ImageSaverFun = MetaImageIO::Saver
    >
    constexpr void read(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{},
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        if (std::ranges::empty(args))
        {
            os << "Usage: read <input_file> [$var]\n";
            return;
        }

        const std::string_view input_arg = args[0];
        std::string output_arg_str{};
        std::string_view output_arg{};

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
            // Dynamically assign variable name from origin file name stem securely
            const std::string_view clean_input = sanitize_string_view(input_arg);
            const std::filesystem::path input_path = std::string(clean_input);
            output_arg_str = "$" + input_path.stem().string();
            output_arg = output_arg_str;
        }

        os << "Reading " << input_arg << " into memory as " << output_arg << "...\n";
        
        auto process_read = [&]<typename ImageType>(ImageType&& input_img)
        {
            image_saver_fun(output_arg, workspace, std::forward<ImageType>(input_img));
        };

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_read))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
            return;
        }

        os << "Done.\n";
    }

    //  save_workspace function implementation
    constexpr void save_workspace(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        if (std::ranges::size(args) < 1)
        {
            os << "Usage: save_workspace <directory_bundle_path>\n";
            return;
        }

        const std::string_view clean_arg = sanitize_string_view(args[0]);
        const std::filesystem::path dir_path = std::string(clean_arg);
        std::filesystem::create_directories(dir_path);

        os << "Saving workspace bundle to " << dir_path.string() << "...\n";

        for (const auto& [name, value] : workspace.memory_store)
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
                    else if constexpr (std::is_same_v<T, TinyDIP::Image<TinyDIP::HSV>>)
                    {
                        TinyDIP::hsv_write(file_path.string().c_str(), *img_ptr); 
                        os << "  Saved $" << name << " -> " << file_path.string() << ".hsv\n";
                    }
                    return true;
                }
                return false;
            };

            using saveable_image_types = std::tuple<
                TinyDIP::Image<TinyDIP::RGB>,
                TinyDIP::Image<double>,
                TinyDIP::Image<TinyDIP::HSV>
            >;

            if (match_any_type<saveable_image_types>(try_save_image))
            {
                // Saved successfully
            }
            else
            {
                auto try_skip_image = [&]<typename T>() -> bool
                {
                    if (value.type() == typeid(T))
                    {
                        os << "  Skipped $" << name << " (Serialization not implemented for this image type)\n";
                        return true;
                    }
                    return false;
                };

                if (!match_any_type<master_image_types>(try_skip_image))
                {
                    os << "  Skipped $" << name << " (Unsupported serialization type)\n";
                }
            }
        }
        os << "Workspace saved successfully.\n";
    }

    //  transform_container template function implementation
	template <typename ImageLoaderFun = MetaImageIO::Loader>
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void transform_container(
        Workspace& workspace,
        std::span<const std::string_view> args,
        const CommandRegistry& registry,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        std::string_view policy_str = "";
        std::vector<std::string_view> filtered_args{};
        filtered_args.reserve(std::ranges::size(args));

        for (const auto& arg : args)
        {
            if (arg == "seq" || arg == "par" || arg == "par_unseq" || arg == "unseq")
            {
                policy_str = arg;
            }
            else
            {
                filtered_args.emplace_back(arg);
            }
        }

        if (std::ranges::size(filtered_args) < 3)
        {
            os << "Usage: transform_container [execution_policy] <input_container | $var> <output_container | $var> <command_name> [command_args...]\n";
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view output_arg = filtered_args[1];
        const std::string_view cmd_name = filtered_args[2];
        
        std::vector<std::string_view> sub_args{};
        for(std::size_t i = 3; i < std::ranges::size(filtered_args); ++i)
        {
            sub_args.emplace_back(filtered_args[i]);
        }

        os << "Applying command '" << cmd_name << "' to elements of " << input_arg << "...\n";

        auto process_container = [&]<typename CandidateType>(CandidateType&& candidate) -> std::any
        {
            using DecayedT = std::remove_cvref_t<CandidateType>;

            if constexpr (is_vector_v<DecayedT> || is_deque_v<DecayedT> || is_list_v<DecayedT>)
            {
                if (std::ranges::empty(candidate))
                {
                    os << "Warning: Input container is empty. Output will be an empty container.\n";
                    workspace.store(output_arg.substr(1), std::vector<typename DecayedT::value_type>{});
                    return std::any{};
                }
                
                const std::size_t total_elements = std::ranges::size(candidate);
                std::vector<std::any> result_anys(total_elements);
                
                auto exec_default = [&]() -> std::any
                {
                    std::size_t idx = 0;
                    for(const auto& element : candidate)
                    {
                        const std::string temp_in = "__tc_in_" + std::to_string(idx);
                        const std::string temp_out = "__tc_out_" + std::to_string(idx);
                        
                        workspace.store(temp_in, element); 
                        
                        std::vector<std::string> local_args_str{};
                        if (!std::ranges::empty(policy_str)) 
                        {
                            local_args_str.emplace_back(std::string(policy_str));
                        }
                        local_args_str.emplace_back("$" + temp_in);
                        local_args_str.emplace_back("$" + temp_out);
                        for (const auto& sa : sub_args) 
                        {
                            local_args_str.emplace_back(std::string(sa));
                        }
                        
                        std::vector<std::string_view> local_args_sv{};
                        for (const auto& s : local_args_str) 
                        {
                            local_args_sv.emplace_back(s);
                        }
                        
                        std::stringstream ss{};
                        registry.execute(workspace, std::string(cmd_name), local_args_sv, ss);
                        
                        auto out_any = workspace.retrieve_any(temp_out);
                        if (out_any.has_value())
                        {
                            result_anys[idx] = out_any.value();
                        }
                        else
                        {
                            throw std::runtime_error("Command failed to produce output variable: $" + temp_out + "\nLog: " + ss.str());
                        }
                        
                        workspace.remove(temp_in);
                        workspace.remove(temp_out);
                        ++idx;
                    }
                    return std::any{};
                };

                auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                    requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                {
                    std::vector<typename DecayedT::value_type> elements(std::ranges::begin(candidate), std::ranges::end(candidate));
                    auto indices = std::views::iota(std::size_t{0}, total_elements);
                    
                    std::for_each(
                        std::forward<ExecPolicy>(exec_policy),
                        std::ranges::begin(indices),
                        std::ranges::end(indices),
                        [&](const std::size_t idx)
                        {
                            const std::string temp_in = "__tc_in_" + std::to_string(idx);
                            const std::string temp_out = "__tc_out_" + std::to_string(idx);
                            
                            workspace.store(temp_in, elements[idx]); 
                            
                            std::vector<std::string> local_args_str{};
                            if (!std::ranges::empty(policy_str)) 
                            {
                                local_args_str.emplace_back(std::string(policy_str));
                            }
                            local_args_str.emplace_back("$" + temp_in);
                            local_args_str.emplace_back("$" + temp_out);
                            for (const auto& sa : sub_args) 
                            {
                                local_args_str.emplace_back(std::string(sa));
                            }
                            
                            std::vector<std::string_view> local_args_sv{};
                            for (const auto& s : local_args_str) 
                            {
                                local_args_sv.emplace_back(s);
                            }
                            
                            std::stringstream ss{};
                            registry.execute(workspace, std::string(cmd_name), local_args_sv, ss);
                            
                            auto out_any = workspace.retrieve_any(temp_out);
                            if (out_any.has_value())
                            {
                                result_anys[idx] = out_any.value();
                            }
                            else
                            {
                                throw std::runtime_error("Command failed to produce output variable: $" + temp_out + "\nLog: " + ss.str());
                            }
                            
                            workspace.remove(temp_in);
                            workspace.remove(temp_out);
                        }
                    );
                    return std::any{};
                };

                try
                {
                    dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                }
                catch (const std::exception& e)
                {
                    os << "Error during transform_container execution: " << e.what() << "\n";
                    return std::any{};
                }
                
                if (std::ranges::empty(result_anys) || !result_anys[0].has_value())
                {
                    os << "Error: Output results are empty or invalid.\n";
                    return std::any{};
                }

                bool reconstructed = false;
                auto try_reconstruct = [&]<typename OutT>() -> bool
                {
                    if (result_anys[0].type() == typeid(OutT))
                    {
                        std::vector<OutT> final_vec{};
                        final_vec.reserve(total_elements);
                        for (std::size_t i = 0; i < total_elements; ++i)
                        {
                            if (result_anys[i].type() != typeid(OutT))
                            {
                                throw std::runtime_error("Inconsistent output types detected during transformation.");
                            }
                            // Leverage std::move and std::any_cast<OutT&> to natively preserve zero-copy performance!
                            final_vec.emplace_back(std::move(std::any_cast<OutT&>(result_anys[i])));
                        }
                        workspace.store(output_arg.substr(1), std::move(final_vec));
                        os << "Successfully transformed container. Saved to " << output_arg << ".\n";
                        reconstructed = true;
                        return true;
                    }
                    return false;
                };

                using ReconstructibleTypes = tuple_cat_t<master_data_types, all_gaussian_params_types>; 
                
                if (!match_any_type<ReconstructibleTypes>(try_reconstruct))
                {
                    os << "Error: Unrecognized output type from transformation. Hash: " << result_anys[0].type().hash_code() << "\n";
                }

                return std::any{};
            }
            else
            {
                os << "Error: Input type [" << get_type_name<DecayedT>() << "] is not a supported container type for transformation.\n";
                return std::any{};
            }
        };

        using AllContainerTypes = tuple_cat_t<master_data_types, master_image_container_types>;

        if (!dispatch_data_operation<AllContainerTypes>(input_arg, workspace, image_loader_fun, process_container))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }
}

#endif //TINYDIP_MAIN_H