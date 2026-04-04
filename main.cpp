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

//  match_any_type template function implementation
template <typename TupleT, class FunT>
constexpr bool match_any_type(FunT&& func)
{
    return [&]<template <typename...> class TupleLike, typename... Ts>(std::type_identity<TupleLike<Ts...>>)
    {
        return (... || std::forward<FunT>(func).template operator()<Ts>());
    }(std::type_identity<TupleT>{});
}

//  TypeActionPair struct implementation
//  A compile-time key-value pair associating a strictly defined Type with a specific Action (Callable)
template <typename TargetT, typename ActionFun>
struct TypeActionPair
{
    using type = TargetT;
    ActionFun action;
};

//  make_type_action template function implementation
template <typename TargetT, typename ActionFun>
constexpr auto make_type_action(ActionFun&& action)
{
    return TypeActionPair<TargetT, std::remove_cvref_t<ActionFun>>{std::forward<ActionFun>(action)};
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
        //  Helper lambda to print size information for image types
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
            // Note: value.type().name() will print the mangled compiler name, 
            // but is helpful enough for debugging type information dynamically.
            os << "  $" << std::left << std::setw(15) << name << " : [Type Hash: " << value.type().hash_code() << "]";

            using image_types = std::tuple<
                TinyDIP::Image<TinyDIP::RGB>, 
                TinyDIP::Image<double>, 
                TinyDIP::Image<TinyDIP::RGB_DOUBLE>,
                TinyDIP::Image<TinyDIP::HSV>,
                TinyDIP::Image<TinyDIP::MultiChannel<double>>
            >;

            // Polymorphic lambda returning true if the image type matched
            auto try_print_image = [&]<typename T>() -> bool
            {
                if (value.type() == typeid(T))
                {
                    os << ", size = ";
                    const auto* image_ptr = std::any_cast<T>(&value);
                    print_size(image_ptr->getSize());
                    return true;
                }
                return false;
            };

            using complex_scalar_types = std::tuple<
                TinyDIP::RGB_DOUBLE,
                TinyDIP::HSV,
                TinyDIP::MultiChannel<double>
            >;

            // Polymorphic lambda returning true if the complex custom scalar type matched
            auto try_print_complex_scalar = [&]<typename T>() -> bool
            {
                if (value.type() == typeid(T))
                {
                    os << ", scalar value = " << std::any_cast<T>(value);
                    return true;
                }
                return false;
            };

            if (match_any_type<image_types>(try_print_image))
            {
                // Handled successfully by try_print_image short-circuit logic
            }
            else if (match_any_type<complex_scalar_types>(try_print_complex_scalar))
            {
                // Handled successfully by try_print_complex_scalar short-circuit logic
            }
            else
            {
                using numeric_types = std::tuple<
                    bool, char, signed char, unsigned char,
                    short, unsigned short, int, unsigned int,
                    long, unsigned long, long long, unsigned long long,
                    float, double, long double, std::size_t
                >;

                // Polymorphic lambda returning true if the numeric type matched
                auto try_print_numeric = [&]<typename T>() -> bool
                {
                    if (value.type() == typeid(T))
                    {
                        if constexpr (sizeof(T) == 1) // Safely print 8-bit integer types as numbers, not unprintable chars
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

                if (!match_any_type<numeric_types>(try_print_numeric))
                {
                    os << " (Unsupported serialization type), type is " << value.type().name();
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
private:
    //  Generic helper to dynamically validate supported types and execute type actions
    template <typename ImageType, typename TupleT>
    static constexpr decltype(auto) execute_file_io(TupleT&& action_map, const std::string_view operation_name)
    {
        // Using declval to perfectly deduce the true return type of the matched action mapping
        using ReturnT = decltype(std::get<0>(std::declval<TupleT>()).action());

        auto fallback_action = [operation_name]() -> ReturnT
        {
            using unsupported_types = std::tuple<
                TinyDIP::Image<TinyDIP::RGB_DOUBLE>,
                TinyDIP::Image<TinyDIP::HSV>,
                TinyDIP::Image<TinyDIP::MultiChannel<double>>
            >;

            constexpr auto is_target_type = []<typename T>() 
            { 
                return std::is_same_v<std::decay_t<ImageType>, T>; 
            };

            if constexpr (match_any_type<unsupported_types>(is_target_type))
            {
                throw std::invalid_argument(std::string(operation_name) + " is not implemented for this complex/high-precision image type.");
            }
            else
            {
                throw std::invalid_argument(std::string(operation_name) + " is not explicitly implemented for this abstract image type.");
            }
        };

        return execute_type_action<std::decay_t<ImageType>>(std::forward<TupleT>(action_map), fallback_action);
    }

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

            // Construct a compile-time map linking concrete types directly to their loading lambdas
            auto action_map = std::make_tuple(
                make_type_action<TinyDIP::Image<TinyDIP::RGB>>([&]() { return TinyDIP::bmp_read(input_path.string().c_str(), true); }),
                make_type_action<TinyDIP::Image<double>>([&]() { return TinyDIP::double_image::read(input_path.string().c_str(), true); })
            );

            // Recursively executes the matching action from the map tuple, entirely generated at compile-time
            return execute_file_io<ImageType>(action_map, "Direct file reading");
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
                
                // Construct a compile-time map linking concrete types directly to their saving lambdas
                auto action_map = std::make_tuple(
                    make_type_action<TinyDIP::Image<double>>([&]() { TinyDIP::double_image::write(path_without_extension.string().c_str(), std::forward<ImageType>(img)); }),
                    make_type_action<TinyDIP::Image<TinyDIP::RGB>>([&]() { TinyDIP::bmp_write(path_without_extension.string().c_str(), std::forward<ImageType>(img)); })
                );

                // Recursively executes the matching action from the map tuple, entirely generated at compile-time
                execute_file_io<std::decay_t<ImageType>>(action_map, "Direct file writing");
            }
        }
    };
};

//  dispatch_image_operation template function implementation
//  Generic helper to dynamically load and dispatch an image (from memory or disk) to a processor lambda
template <typename ProcessorFun, typename ImageLoaderFun>
requires (std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&> &&
          std::invocable<ProcessorFun, std::invoke_result_t<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&>>)
constexpr bool dispatch_image_operation(
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

        using checking_types = std::tuple<
            TinyDIP::Image<TinyDIP::RGB>, 
            TinyDIP::Image<double>, 
            TinyDIP::Image<TinyDIP::RGB_DOUBLE>,
            TinyDIP::Image<TinyDIP::HSV>,
            TinyDIP::Image<TinyDIP::MultiChannel<double>>
        >;

        return match_any_type<checking_types>(try_process);
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
//  Generic Meta Handler strictly refactoring transform commands like bicubic_resize, dct2, idct2
template <std::size_t MinArgs, typename SetupFun>
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
            os << "       Optional Execution policies: seq, par, par_unseq, unseq\n";
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view output_arg = filtered_args[1];

        // Parse trailing args, output initial message, and retrieve dedicated transformation process
        auto core_processor = setup_fun_(filtered_args, policy_str, os);

        auto process_wrapper = [&]<typename ImageType>(ImageType&& input_img)
        {
            auto output_img = core_processor(std::forward<ImageType>(input_img));
            image_saver_fun(output_arg, workspace_, std::move(output_img));
            os << "Saved to " << output_arg << "\n";
        };

        if (!dispatch_image_operation(input_arg, workspace_, image_loader_fun, process_wrapper))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }
};

//  make_meta_transform_handler template function implementation
template <std::size_t MinArgs, typename SetupFun>
constexpr auto make_meta_transform_handler(std::string_view usage, std::shared_ptr<Workspace> ws, SetupFun&& setup)
{
    return MetaTransformHandler<MinArgs, std::remove_cvref_t<SetupFun>>{
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
            os << "       Optional Execution policies: seq, par, par_unseq, unseq\n";
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
            auto handle_result = [&](auto&& scalar_result)
            {
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
                        if constexpr (sizeof(decltype(scalar_result)) == 1 && std::is_integral_v<std::decay_t<decltype(scalar_result)>>)
                        {
                            os << capitalized_op_name_ << " result: " << +scalar_result << "\n";
                        }
                        else
                        {
                            os << capitalized_op_name_ << " result: " << scalar_result << "\n";
                        }
                    }
                    else
                    {
                        os << capitalized_op_name_ << " result evaluated successfully (Non-printable complex type).\n";
                    }
                }
            };

            handle_result(core_processor(std::forward<ImageType>(input_img)));
        };

        if (!dispatch_image_operation(input_arg, workspace_, image_loader_fun, process_scalar))
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

        if (!dispatch_image_operation(input_arg, workspace_, image_loader_fun, process_read))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
            return;
        }

        os << "Done.\n";
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

        if (!dispatch_image_operation(input_arg, workspace_, image_loader_fun, process_write))
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
            using saveable_image_types = std::tuple<
                TinyDIP::Image<TinyDIP::RGB>,
                TinyDIP::Image<double>
            >;

            auto try_save_image = [&]<typename T>() -> bool
            {
                if (value.type() == typeid(T))
                {
                    const auto* img_ptr = std::any_cast<T>(&value);
                    const std::filesystem::path file_path = dir_path / (name);
                    
                    auto action_map = std::make_tuple(
                        make_type_action<TinyDIP::Image<TinyDIP::RGB>>(
                            [&]() 
                            { 
                                TinyDIP::bmp_write(file_path.string().c_str(), *img_ptr); 
                                os << "  Saved $" << name << " -> " << file_path.string() << ".bmp\n";
                            }),
                        make_type_action<TinyDIP::Image<double>>(
                            [&]() 
                            { 
                                TinyDIP::double_image::write(file_path.string().c_str(), *img_ptr); 
                                os << "  Saved $" << name << " -> " << file_path.string() << ".dbmp\n";
                            })
                    );

                    auto fallback_action = []() {};

                    execute_type_action<T>(action_map, fallback_action);
                    return true;
                }
                return false;
            };

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
        {
            os << "Image Info:\n";
            os << "  Source: " << input_arg << "\n";
            os << "  Width:  " << img.getWidth() << "\n";
            os << "  Height: " << img.getHeight() << "\n";
        };

        if (!dispatch_image_operation(input_arg, workspace_, image_loader_fun, process_info))
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
        {
            os << "Printing image content for " << input_arg << ":\n";
            img.print(",");
            os << "Done.\n";
        };

        if (!dispatch_image_operation(input_arg, workspace_, image_loader_fun, process_print))
        {
            // If dispatch_image_operation returns false, it must be a $ variable holding a scalar or unsupported type
            const std::string_view var_name = input_arg.substr(1);
            
            using complex_scalar_types = std::tuple<
                TinyDIP::RGB_DOUBLE,
                TinyDIP::HSV,
                TinyDIP::MultiChannel<double>
            >;

            // Polymorphic lambda returning true if the complex custom scalar type matched
            auto try_print_complex_scalar = [&]<typename T>() -> bool
            {
                if (workspace_->retrieve<T>(var_name))
                {
                    os << "Printing scalar value for " << input_arg << ":\n";
                    os << *workspace_->retrieve<T>(var_name) << "\nDone.\n";
                    return true;
                }
                return false;
            };

            if (match_any_type<complex_scalar_types>(try_print_complex_scalar))
            {
                // Handled successfully by try_print_complex_scalar short-circuit logic
            }
            else
            {
                using numeric_types = std::tuple<
                    bool, char, signed char, unsigned char,
                    short, unsigned short, int, unsigned int,
                    long, unsigned long, long long, unsigned long long,
                    float, double, long double, std::size_t
                >;

                // Polymorphic lambda returning true if the numeric type matched
                auto try_print_numeric = [&]<typename T>() -> bool
                {
                    if (workspace_->retrieve<T>(var_name))
                    {
                        os << "Printing scalar value for " << input_arg << ":\n";
                        if constexpr (sizeof(T) == 1) // Safely print 8-bit integer types as numbers, not unprintable chars
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

                if (!match_any_type<numeric_types>(try_print_numeric))
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
        std::invocable<const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<double>&&> ImageSaverFun = ImageSaver
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

//  SumHandler struct implementation
struct SumHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = ImageLoader
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

        if (std::ranges::empty(filtered_args))
        {
            os << "Usage: sum [execution_policy] <input_img | $var> [output_var | $var]\n";
            os << "       Execution policies: seq, par, par_unseq, unseq\n";
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        std::string_view output_arg = "";
        
        if (std::ranges::size(filtered_args) > 1)
        {
            output_arg = filtered_args[1];
        }

        if (!std::ranges::empty(policy_str))
        {
            os << "Calculating sum of " << input_arg << " (Policy: " << policy_str << ")...\n";
        }
        else
        {
            os << "Calculating sum of " << input_arg << "...\n";
        }

        // Polymorphic lambda to cleanly execute the algorithm dynamically independent of image type
        auto process_sum = [&]<typename ImageType>(ImageType&& input_img)
        {
            auto handle_result = [&](auto&& sum_result)
            {
                if (!std::ranges::empty(output_arg))
                {
                    if (output_arg.starts_with('$'))
                    {
                        workspace_->store(output_arg.substr(1), sum_result);
                        os << "Saved sum result to " << output_arg << "\n";
                    }
                    else
                    {
                        os << "Error: Output must be a memory variable starting with '$'.\n";
                    }
                }
                else
                {
                    if constexpr (requires { os << sum_result; })
                    {
                        os << "Sum result: " << sum_result << "\n";
                    }
                    else
                    {
                        os << "Sum result evaluated successfully (Non-printable complex type).\n";
                    }
                }
            };

            auto execute_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy)
                requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
            {
                if constexpr (requires { TinyDIP::sum(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(input_img)); })
                {
                    handle_result(TinyDIP::sum(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(input_img)));
                }
                else
                {
                    os << "Warning: Execution policy requested but not supported for this image type. Falling back to default.\n";
                    handle_result(TinyDIP::sum(std::forward<ImageType>(input_img)));
                }
            };

            const std::map<std::string_view, std::function<void()>> execute_policy_map = {
                {"par",       [&]() { execute_policy(std::execution::par); }},
                {"par_unseq", [&]() { execute_policy(std::execution::par_unseq); }},
                {"unseq",     [&]() { execute_policy(std::execution::unseq); }},
                {"seq",       [&]() { execute_policy(std::execution::seq); }}
            };

            if (auto it = execute_policy_map.find(policy_str); it != std::ranges::end(execute_policy_map))
            {
                it->second();
            }
            else
            {
                handle_result(TinyDIP::sum(std::forward<ImageType>(input_img)));
            }
        };

        if (input_arg.starts_with('$'))
        {
            const std::string_view var_name = input_arg.substr(1);
            if (workspace_->retrieve<TinyDIP::Image<TinyDIP::RGB>>(var_name))
            {
                process_sum(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
            else if (workspace_->retrieve<TinyDIP::Image<double>>(var_name))
            {
                process_sum(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
            }
            else
            {
                os << "Error: Memory variable not found or unsupported type.\n";
                return;
            }
        }
        else
        {
            const std::filesystem::path input_path = std::string(input_arg);
            if (input_path.extension() == ".dbmp")
            {
                process_sum(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
            }
            else
            {
                process_sum(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
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

//  Main Entry Point
int main(int argc, char* argv[])
{
    // Configure the shared state memory workspace
    auto workspace = std::make_shared<Workspace>();

    // Define human-readable pipeline schema routing constants
    constexpr auto GeneratorSchema   = IOSchema{-1, 1};
    constexpr auto TerminatorSchema  = IOSchema{0, -1};
    constexpr auto TransformerSchema = IOSchema{0, 1};
    constexpr auto IndependentSchema = IOSchema{-1, -1};

    // Register commands directly with context-injected instances using generic variadic bundles
    CommandRegistry registry = command_registration(
        CommandBundle{"bicubic_resize", "Resize an image using Bicubic interpolation.", TransformerSchema, BicubicResizeHandler{workspace}},
        CommandBundle{"dct2", "Calculate Discrete Cosine Transformation for an image.", TransformerSchema, Dct2Handler{workspace}},
        CommandBundle{"idct2", "Calculate Inverse Discrete Cosine Transformation for an image.", TransformerSchema, Idct2Handler{workspace}},
        CommandBundle{"info", "Display basic information about an image.", TerminatorSchema, InfoHandler{workspace}},
        CommandBundle{"lanczos_resample", "Resize an image using Lanczos resampling.", TransformerSchema, LanczosResampleHandler{workspace}},
        CommandBundle{"load_workspace", "Load memory variables from a directory bundle.", IndependentSchema, LoadWorkspaceHandler{workspace}},
        CommandBundle{"print", "Print the contents of a memory variable.", TerminatorSchema, PrintHandler{workspace}},
        CommandBundle{"rand", "Generate random multi-dimensional image with specified URBG.", GeneratorSchema, RandHandler{workspace}},
        CommandBundle{"read", "Read an image from disk into a memory variable.", GeneratorSchema, ReadHandler{workspace}},
        CommandBundle{"remove", "Remove memory variables from the workspace (or 'all' to clear).", IndependentSchema, RemoveHandler{workspace}},
        CommandBundle{"save_workspace", "Save all memory variables to a directory bundle.", IndependentSchema, SaveWorkspaceHandler{workspace}},
        CommandBundle{"sum", "Calculate the sum of all elements in an image.", TransformerSchema, SumHandler{workspace}},
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
