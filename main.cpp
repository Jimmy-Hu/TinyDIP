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
T parse_arg(std::string_view sv)
{
    T result{};
    if constexpr (std::is_arithmetic_v<T>)
    {
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);
        if (ec != std::errc())
        {
            throw std::runtime_error(TinyDIP::Formatter() << "Error parsing argument: " << sv << " \n");
        }
    }
    else
    {
        //  Fallback for non-arithmetic types (unlikely to be used with this function in current context)
        //  This path forces allocation, but is rarely hit for numeric parsing
        std::string temp(sv);
        std::stringstream ss(temp);
        ss >> result;
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

//  CommandRegistry class implementation
class CommandRegistry
{
private:
    std::map<std::string, std::pair<std::string, CommandHandler>> commands;

public:
    void register_command(const std::string& name, const std::string& description, CommandHandler handler)
    {
        commands.emplace(name, std::make_pair(description, std::move(handler)));
    }

    void list_commands(std::ostream& os = std::cout) const
    {
        os << "Available Commands:\n";
        for (const auto& [name, info] : commands)
        {
            os << "  " << std::left << std::setw(15) << name << " : " << info.first << "\n";
        }
        os << "\nUsage: ./tinydip <command> [args...]\n";
    }

    //  Refactored execute to pass the std::ostream& context directly down to handlers
    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    void execute(const std::string& command_name, const ArgsT& args, std::ostream& os = std::cout) const
    {
        if (auto it = commands.find(command_name); it != std::ranges::end(commands))
        {
            try
            {
                //  Zero-copy path: If the incoming generic range is already contiguous (e.g., std::vector, std::array),
                //  we can wrap it directly in a std::span without allocating any memory.
                if constexpr (std::ranges::contiguous_range<ArgsT> && std::same_as<std::ranges::range_value_t<ArgsT>, std::string_view>)
                {
                    it->second.second(std::span<const std::string_view>{std::ranges::data(args), std::ranges::size(args)}, os);
                }
                //  Fallback path: Convert non-contiguous generic random-access range to a contiguous block.
                else
                {
                    std::vector<std::string_view> contiguous_args;
                    contiguous_args.reserve(std::ranges::size(args));
                    for (const auto& arg : args)
                    {
                        contiguous_args.emplace_back(arg);
                    }
                    it->second.second(std::span<const std::string_view>{std::ranges::data(contiguous_args), std::ranges::size(contiguous_args)}, os);
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

//  BicubicResizeHandler struct implementation
//  Wrapper for the 'bicubic_resize' functionality
//  Args: input_path output_path width height
struct BicubicResizeHandler
{
    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        if (std::ranges::size(args) < 4)
        {
            std::cerr << "Usage: bicubic_resize <input_bmp> <output_bmp> <width> <height>\n";
            return;
        }

        std::string input_path(std::string_view{args[0]});
        std::filesystem::path output_filepath = std::string(std::string_view{args[1]});
        std::size_t width = parse_arg<std::size_t>(std::string_view{args[2]});
        std::size_t height = parse_arg<std::size_t>(std::string_view{args[3]});

        os << "Resizing " << input_path << " to " << width << "x" << height << "...\n";

        //  Reading image
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), true); // Assume true for convert to RGB/standard

        //  Perform operation
        //  Using execution policy if TinyDIP supports it internally, otherwise standard call
        auto output_img = TinyDIP::copyResizeBicubic(input_img, width, height);

        //  Writing image
        std::filesystem::path path_without_extension = output_filepath.parent_path() / output_filepath.stem();
        TinyDIP::bmp_write(path_without_extension.string().c_str(), output_img);
        os << "Saved to " << output_filepath.string() << "\n";
    }
};

//  InfoHandler struct implementation
//  Wrapper for 'info' functionality
//  Args: input_path
struct InfoHandler
{
    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        if (std::ranges::empty(args))
        {
            std::cerr << "Usage: info <input_bmp>\n";
            return;
        }

        std::filesystem::path input_path = std::string(args[0]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << "\n";
            return;
        }
        std::filesystem::path path_without_extension = input_path.parent_path() / input_path.stem();
        auto img = TinyDIP::bmp_read(path_without_extension.string().c_str(), false);
        os << "Image Info:\n";
        os << "  Path:   " << input_path << "\n";
        os << "  Width:  " << img.getWidth() << "\n";
        os << "  Height: " << img.getHeight() << "\n";
        //  Add more info if available (channels, etc.)
    }
};

//  RandHandler struct implementation
//  Wrapper for 'rand' functionality
//  Args: urbg_type output_path dim1 [dim2] [dim3] ...
struct RandHandler
{
    //  Define a struct with a call operator to be used as the generator lambda, 
    //  allowing for generic type deduction and avoiding raw lambdas.
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

    template <std::ranges::random_access_range ArgsT>
    requires std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view>
    void operator()(const ArgsT& args, std::ostream& os = std::cout) const
    {
        //  Generic lambda to dispatch the generation logic using the built-in concept constraint
        //  Takes initialized variables as explicitly passed arguments to eliminate state dependency
        auto dispatch_generation = [&]
        <std::ranges::random_access_range SzArgsT>
        requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
        (std::uniform_random_bit_generator auto&& urbg, const std::filesystem::path& out_path, const SzArgsT& sz)
        {
            std::uniform_real_distribution<double> dist{};
            using UrbgType = std::remove_cvref_t<decltype(urbg)>;
            using DistType = decltype(dist);

            RandomGenerator<UrbgType, DistType> gen{urbg, dist};

            //  Calling the dynamic range-based generate overload directly from TinyDIP.
            auto output_img = TinyDIP::generate(gen, sz);

            //  Writing image
            std::filesystem::path path_without_extension = out_path.parent_path() / out_path.stem();
            TinyDIP::double_image::write(path_without_extension.string().c_str(), output_img);
            os << "Saved to " << out_path.string() << "\n";
        };

        //  Runtime string dispatch mapping to compile-time URBG types via std::map
        //  Uses std::span to avoid hardcoding std::vector, establishing a fully generic type-erased boundary
        std::map<std::string_view, std::function<void(const std::filesystem::path&, std::span<const std::size_t>)>> urbg_mapping = {
            {"knuth_b",       [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::knuth_b{std::random_device{}()}, out_path, sz); 
                }
            },
            {"minstd_rand",   [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::minstd_rand{std::random_device{}()}, out_path, sz); 
                }
            },
            {"minstd_rand0",  [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::minstd_rand0{std::random_device{}()}, out_path, sz); 
                }
            },
            {"mt19937",       [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::mt19937{std::random_device{}()}, out_path, sz); 
                }
            },
            {"mt19937_64",    [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::mt19937_64{std::random_device{}()}, out_path, sz); 
                }
            },
            {"ranlux24",      [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::ranlux24{std::random_device{}()}, out_path, sz); 
                }
            },
            {"ranlux24_base", [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::ranlux24_base{std::random_device{}()}, out_path, sz); 
                }
            },
            {"ranlux48",      [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::ranlux48{std::random_device{}()}, out_path, sz); 
                }
            },
            {"ranlux48_base", [&]
                <std::ranges::random_access_range SzArgsT>
                requires std::convertible_to<std::ranges::range_value_t<SzArgsT>, std::size_t>
                (const std::filesystem::path& out_path, const SzArgsT& sz)
                { 
                    dispatch_generation(std::ranlux48_base{std::random_device{}()}, out_path, sz); 
                }
            }
        };

        //  Helper lambda to dynamically print available URBGs safely formatted
        auto print_available_urbgs = [&]()
        {
            os << "Available URBGs: ";
            for (auto it = urbg_mapping.begin(); it != urbg_mapping.end(); ++it)
            {
                os << it->first;
                if (std::next(it) != urbg_mapping.end())
                {
                    os << ", ";
                }
            }
            os << '\n';
        };

        if (std::ranges::size(args) < 3)
        {
            os << "Usage: rand <urbg_type> <output_bmp> <dim1> [dim2] [dim3] ...\n";
            print_available_urbgs();
            return;
        }

        std::string_view urbg_type = args[0];
        std::filesystem::path output_filepath = std::string(std::string_view{args[1]});
        
        //  Constructing sizes vector sequentially from arbitrary dimensions given in the CLI
        std::vector<std::size_t> sizes;
        sizes.reserve(std::ranges::size(args) - 2);
        
        for (std::size_t i = 2; i < std::ranges::size(args); ++i)
        {
            sizes.emplace_back(parse_arg<std::size_t>(std::string_view{args[i]}));
        }

        os << "Generating random image with dimensions: ";
        for (const auto& size : sizes)
        {
            os << size << " ";
        }
        os << "using URBG '" << urbg_type << "'...\n";

        if (auto it = urbg_mapping.find(urbg_type); it != urbg_mapping.end())
        {
            it->second(output_filepath, sizes);
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

    bmp1 = TinyDIP::concat(TinyDIP::recursive_transform<2>(
        //std::execution::par,
        [](auto&& element)
        {
            auto hsv_block = TinyDIP::rgb2hsv(TinyDIP::im2double(element));
            auto v_block = TinyDIP::getVplane(hsv_block);
            auto v_block_dct = TinyDIP::dct2(v_block);
            return TinyDIP::hsv2rgb(TinyDIP::constructHSV(
                TinyDIP::getHplane(hsv_block),
                TinyDIP::getSplane(hsv_block),
                TinyDIP::idct2(v_block_dct)
            ));
        },
        TinyDIP::split(bmp1, block_count_x, block_count_y)));
    bmp1 = copyResizeBicubic(bmp1, bmp1.getWidth() * 2, bmp1.getHeight() * 2);
    //bmp1 = gaussian_fisheye(bmp1, 800.0);
    auto v_plane = TinyDIP::getVplane(TinyDIP::rgb2hsv(bmp1));
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

//  command_registration function implementation
//  Function to initialize and register all available commands
template <
    std::invocable<std::span<const std::string_view>, std::ostream&> BicubicResizeFun = BicubicResizeHandler,
    std::invocable<std::span<const std::string_view>, std::ostream&> InfoHandlerFun = InfoHandler,
    std::invocable<std::span<const std::string_view>, std::ostream&> RandHandlerFun = RandHandler>
CommandRegistry command_registration(
    BicubicResizeFun&& bicubic_resize_fun = {},
    InfoHandlerFun&& info_handler_fun = {},
    RandHandlerFun&& rand_handler_fun = {})
{
    CommandRegistry registry;

    //  Registering commands
    registry.register_command("bicubic_resize", "Resize an image using Bicubic interpolation.", std::forward<BicubicResizeFun>(bicubic_resize_fun));

    registry.register_command("info", "Display basic information about an image.", std::forward<InfoHandlerFun>(info_handler_fun));

    registry.register_command("rand", "Generate random multi-dimensional image.", std::forward<RandHandlerFun>(rand_handler_fun));

    //  Note: Wrapped in a lambda to conform to the type-erased span boundary
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
                std::cerr << "Usage: batch_add_zeros <input_dir> <output_dir>\n";
                return;
            }
            //  Note: args[0] and args[1] need to be converted to std::string if your underlying function expects strings
            //  Example: addLeadingZeros(std::string(args[0]), std::string(args[1]));
            os << "Batch processing from " << args[0] << " to " << args[1] << "\n";
        }
    );

    return registry;
}

//  run_interactive_mode function implementation
//  Interactive REPL loop implementation
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

        if (line.empty())
        {
            continue;
        }

        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;
        
        while (iss >> token)
        {
            tokens.emplace_back(std::move(token));
        }

        if (std::ranges::empty(tokens))
        {
            continue;
        }

        std::string command = tokens[0];
        
        if (command == "exit" || command == "quit")
        {
            break;
        }

        std::vector<std::string_view> args;
        args.reserve(std::ranges::size(tokens) - 1);
        
        for (std::size_t i = 1; i < std::ranges::size(tokens); ++i)
        {
            args.emplace_back(tokens[i]);
        }

        registry.execute(command, args);
    }
}

//  Main Entry Point
int main(int argc, char* argv[])
{
    CommandRegistry registry = command_registration();

    //  Dynamically register the generic help command handler
    registry.register_command("help", "List all available commands.", HelpHandler{registry});

    //  Argument Parsing (No argument launches interactive REPL mode)
    if (argc < 2)
    {
        run_interactive_mode(registry);
        return EXIT_SUCCESS;
    }

    std::string command = argv[1];
    
    //  Using std::string_view for arguments to avoid heap allocation
    std::vector<std::string_view> args;

    //  Collect arguments for the command
    if (argc > 2)
    {
        args.reserve(argc - 2);
        for (int i = 2; i < argc; ++i)
        {
            //  Construct string_view directly from argv char pointers
            //  No string copying happens here!
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
