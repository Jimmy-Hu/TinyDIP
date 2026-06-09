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
        memory_store[std::string(name)] = std::forward<T>(item);
    }

    template <typename T>
    const T* retrieve(const std::string_view name) const
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (auto it = memory_store.find(std::string(name)); it != std::ranges::end(memory_store))
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
        if (auto it = memory_store.find(std::string(name)); it != std::ranges::end(memory_store))
        {
            return it->second;
        }
        return std::nullopt;
    }

    //  remove function implementation
    bool remove(const std::string_view name)
    {
        std::lock_guard<std::mutex> lock(mtx);
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
        std::lock_guard<std::mutex> lock(mtx);
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

            const std::filesystem::path input_path = std::string(arg);
            const bool has_ext = input_path.has_extension();

            if constexpr (std::is_same_v<ImageType, TinyDIP::Image<TinyDIP::RGB>>)
            {
                if (has_ext && input_path.extension() == ".ppm")
                {
                    return TinyDIP::pnm::read(input_path);
                }
                
                return TinyDIP::bmp_read(input_path.string().c_str(), has_ext);
            }
            else if constexpr (std::is_same_v<ImageType, TinyDIP::Image<double>>)
            {
                if (has_ext && input_path.extension() == ".csv")
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
                const std::filesystem::path output_filepath = std::string(arg);
                const std::filesystem::path path_without_extension = output_filepath.parent_path() / output_filepath.stem();
                
                if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<double>>)
                {
                    if (output_filepath.extension() == ".csv")
                    {
                        TinyDIP::double_image::write_to_csv(output_filepath.string().c_str(), std::forward<ImageType>(img));
                    }
                    else
                    {
                        TinyDIP::double_image::write(path_without_extension.string().c_str(), std::forward<ImageType>(img));
                    }
                }
                else if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::RGB>>)
                {
                    if (output_filepath.extension() == ".ppm")
                    {
                        TinyDIP::pnm::write(std::forward<ImageType>(img), output_filepath);
                    }
                    else
                    {
                        TinyDIP::bmp_write(path_without_extension.string().c_str(), std::forward<ImageType>(img));
                    }
                }
                else if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::HSV>>)
                {
                    TinyDIP::hsv_write(path_without_extension.string().c_str(), std::forward<ImageType>(img));
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

//  CommandHandler type alias definition
//  Modern C++ Standard Function signature for highly robust, state-injected execution
using CommandHandler = std::function<void(Workspace&, std::span<const std::string_view>, std::ostream&)>;


#endif //TINYDIP_MAIN_H