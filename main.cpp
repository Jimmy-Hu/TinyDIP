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
        if (auto it = memory_store.find(std::string(name)); it != memory_store.end())
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

    //  list_variables function implementation
    void list_variables(std::ostream& os) const
    {
        if (std::ranges::empty(memory_store))
        {
            os << "  (Workspace is empty)\n";
            return;
        }
        for (const auto& [name, value] : memory_store)
        {
            // Note: value.type().name() will print the mangled compiler name, 
            // but is helpful enough for debugging type information dynamically.
            os << "  $" << std::left << std::setw(15) << name << " : [Type Hash: " << value.type().hash_code() << "]";
            
            if (value.type() == typeid(TinyDIP::Image<TinyDIP::RGB>))
            {
                os << ", size = ";
                const auto* image_ptr = std::any_cast<TinyDIP::Image<TinyDIP::RGB>>(&value);
                const auto& image_size = image_ptr->getSize();
                
                auto it = std::ranges::begin(image_size);
                const auto end = std::ranges::end(image_size);
                if (it != end)
                {
                    os << +(*it);
                    ++it;
                    for (; it != end; ++it)
                    {
                        os << " x " << +(*it);
                    }
                }
            }
            else if (value.type() == typeid(TinyDIP::Image<double>))
            {
                os << ", size = ";
                const auto* image_ptr = std::any_cast<TinyDIP::Image<double>>(&value);
                const auto& image_size = image_ptr->getSize();
                
                auto it = std::ranges::begin(image_size);
                const auto end = std::ranges::end(image_size);
                if (it != end)
                {
                    os << +(*it);
                    ++it;
                    for (; it != end; ++it)
                    {
                        os << " x " << +(*it);
                    }
                }
            }
            else
            {
                os << " (Unsupported serialization type)";
            }
            os << '\n';
        }
    }
};

//  ImageLoader struct implementation
//  Generic functor to abstract loading an image from either a Workspace Memory Variable or Disk
struct ImageLoader
{
    template <typename ImageType = TinyDIP::Image<TinyDIP::RGB>>
    ImageType operator()(const std::string_view arg, const std::shared_ptr<Workspace>& ws) const
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
            return TinyDIP::bmp_read(std::string(arg).c_str(), true);
        }
        else if constexpr (std::is_same_v<ImageType, TinyDIP::Image<double>>)
        {
            return TinyDIP::double_image::read(std::string(arg).c_str(), true);
        }
        else
        {
            throw std::invalid_argument("Direct file reading is not explicitly implemented for this abstract image type.");
        }
    }
};

//  ImageSaver struct implementation
//  Generic functor to abstract saving an image to either a Workspace Memory Variable or Disk
struct ImageSaver
{
    template <typename ImageType>
    constexpr void operator()(const std::string_view arg, const std::shared_ptr<Workspace>& ws, ImageType&& img) const
    {
        if (arg.starts_with('$'))
        {
            std::string_view var_name = arg.substr(1);
            ws->store(var_name, std::forward<ImageType>(img));
        }
        else
        {
            std::filesystem::path output_filepath = std::string(arg);
            std::filesystem::path path_without_extension = output_filepath.parent_path() / output_filepath.stem();
            
            if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<double>>)
            {
                TinyDIP::double_image::write(path_without_extension.string().c_str(), std::forward<ImageType>(img));
            }
            else if constexpr (std::is_same_v<std::decay_t<ImageType>, TinyDIP::Image<TinyDIP::RGB>>)
            {
                TinyDIP::bmp_write(path_without_extension.string().c_str(), std::forward<ImageType>(img));
            }
        }
    }
};

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

//  ReadHandler struct implementation
struct ReadHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT, 
        typename ImageLoaderFun = ImageLoader,
        typename ImageSaverFun = ImageSaver
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<double>&&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: read <input_file> <$var>\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];

        if (!output_arg.starts_with('$'))
        {
            os << "Error: Output must be a memory variable starting with '$'.\n";
            return;
        }

        os << "Reading " << input_arg << " into memory as " << output_arg << "...\n";
        
        const std::filesystem::path input_path = std::string(input_arg);
        if (input_path.extension() == ".dbmp")
        {
            auto img = image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_);
            image_saver_fun(output_arg, workspace_, std::move(img));
        }
        else
        {
            auto img = image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_);
            image_saver_fun(output_arg, workspace_, std::move(img));
        }
        os << "Done.\n";
    }
};

//  WriteHandler struct implementation
struct WriteHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = ImageLoader,
        typename ImageSaverFun = ImageSaver
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

        // Dynamic type-erasure boundary resolution to safely write correct type
        if (workspace_->retrieve<TinyDIP::Image<TinyDIP::RGB>>(input_arg.substr(1)))
        {
            auto img = image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_);
            image_saver_fun(output_arg, workspace_, std::move(img));
        }
        else if (workspace_->retrieve<TinyDIP::Image<double>>(input_arg.substr(1)))
        {
            auto img = image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_);
            image_saver_fun(output_arg, workspace_, std::move(img));
        }
        else
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
            if (value.type() == typeid(TinyDIP::Image<TinyDIP::RGB>))
            {
                auto img = std::any_cast<TinyDIP::Image<TinyDIP::RGB>>(value);
                const std::filesystem::path file_path = dir_path / (name);
                TinyDIP::bmp_write(file_path.string().c_str(), img);
                os << "  Saved $" << name << " -> " << file_path.string() << ".bmp\n";
            }
            else if (value.type() == typeid(TinyDIP::Image<double>))
            {
                auto img = std::any_cast<TinyDIP::Image<double>>(value);
                const std::filesystem::path file_path = dir_path / (name);
                TinyDIP::double_image::write(file_path.string().c_str(), img);
                os << "  Saved $" << name << " -> " << file_path.string() << ".dbmp\n";
            }
            else
            {
                os << "  Skipped $" << name << " (Unsupported serialization type)\n";
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

//  BicubicResizeHandler struct implementation
//  Wrapper for the 'bicubic_resize' functionality
//  Args: input_path output_path width height
struct BicubicResizeHandler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = ImageLoader,
        typename ImageSaverFun = ImageSaver
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<double>&&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        if (std::ranges::size(args) < 4)
        {
            os << "Usage: bicubic_resize <input_bmp | $var> <output_bmp | $var> <width> <height>\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];
        const std::size_t width = parse_arg<std::size_t>(args[2]);
        const std::size_t height = parse_arg<std::size_t>(args[3]);

        os << "Resizing " << input_arg << " to " << width << "x" << height << "...\n";

        // Polymorphic lambda to cleanly execute the algorithm dynamically independent of image type
        auto process_resize = [&]<typename ImageType>(ImageType&& input_img)
        {
            auto output_img = TinyDIP::copyResizeBicubic(std::forward<ImageType>(input_img), width, height);
            image_saver_fun(output_arg, workspace_, std::move(output_img));
            os << "Saved to " << output_arg << "\n";
        };

        if (input_arg.starts_with('$'))
        {
            const std::string_view var_name = input_arg.substr(1);
            if (workspace_->retrieve<TinyDIP::Image<TinyDIP::RGB>>(var_name))
            {
                process_resize(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
            else if (workspace_->retrieve<TinyDIP::Image<double>>(var_name))
            {
                process_resize(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
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
                process_resize(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
            }
            else
            {
                process_resize(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
        }
    }
};

//  Dct2Handler struct implementation
struct Dct2Handler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = ImageLoader,
        typename ImageSaverFun = ImageSaver
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<double>&&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: dct2 <input_img | $var> <output_img | $var>\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];

        os << "Calculating DCT-2 for " << input_arg << "...\n";

        // Polymorphic lambda to cleanly execute the algorithm dynamically independent of image type
        auto process_dct2 = [&]<typename ImageType>(ImageType&& input_img)
        {
            auto output_img = TinyDIP::dct2(std::forward<ImageType>(input_img));
            image_saver_fun(output_arg, workspace_, std::move(output_img));
            os << "Saved to " << output_arg << "\n";
        };

        if (input_arg.starts_with('$'))
        {
            const std::string_view var_name = input_arg.substr(1);
            if (workspace_->retrieve<TinyDIP::Image<TinyDIP::RGB>>(var_name))
            {
                process_dct2(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
            else if (workspace_->retrieve<TinyDIP::Image<double>>(var_name))
            {
                process_dct2(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
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
                process_dct2(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
            }
            else
            {
                process_dct2(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
        }
    }
};

//  Idct2Handler struct implementation
struct Idct2Handler
{
    std::shared_ptr<Workspace> workspace_;

    template <
        std::ranges::random_access_range ArgsT,
        typename ImageLoaderFun = ImageLoader,
        typename ImageSaverFun = ImageSaver
    >
    requires (std::convertible_to<std::ranges::range_value_t<ArgsT>, std::string_view> &&
              std::invocable<ImageLoaderFun, const std::string_view, const std::shared_ptr<Workspace>&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, const std::shared_ptr<Workspace>&, TinyDIP::Image<double>&&>)
    constexpr void operator()(const ArgsT& args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}, ImageSaverFun&& image_saver_fun = ImageSaverFun{}) const
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: idct2 <input_img | $var> <output_img | $var>\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];

        os << "Calculating Inverse DCT-2 for " << input_arg << "...\n";

        // Polymorphic lambda to cleanly execute the algorithm dynamically independent of image type
        auto process_idct2 = [&]<typename ImageType>(ImageType&& input_img)
        {
            auto output_img = TinyDIP::idct2(std::forward<ImageType>(input_img));
            image_saver_fun(output_arg, workspace_, std::move(output_img));
            os << "Saved to " << output_arg << "\n";
        };

        if (input_arg.starts_with('$'))
        {
            const std::string_view var_name = input_arg.substr(1);
            if (workspace_->retrieve<TinyDIP::Image<TinyDIP::RGB>>(var_name))
            {
                process_idct2(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
            else if (workspace_->retrieve<TinyDIP::Image<double>>(var_name))
            {
                process_idct2(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
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
                process_idct2(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
            }
            else
            {
                process_idct2(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
        }
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
        typename ImageLoaderFun = ImageLoader
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

        if (input_arg.starts_with('$'))
        {
            const std::string_view var_name = input_arg.substr(1);
            if (workspace_->retrieve<TinyDIP::Image<TinyDIP::RGB>>(var_name))
            {
                process_info(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
            else if (workspace_->retrieve<TinyDIP::Image<double>>(var_name))
            {
                process_info(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
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
                process_info(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
            }
            else
            {
                process_info(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
        }
    }
};

//  PrintHandler struct implementation
struct PrintHandler
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

        if (input_arg.starts_with('$'))
        {
            const std::string_view var_name = input_arg.substr(1);
            if (workspace_->retrieve<TinyDIP::Image<TinyDIP::RGB>>(var_name))
            {
                process_print(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
            }
            else if (workspace_->retrieve<TinyDIP::Image<double>>(var_name))
            {
                process_print(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
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
                process_print(image_loader_fun.template operator()<TinyDIP::Image<double>>(input_arg, workspace_));
            }
            else
            {
                process_print(image_loader_fun.template operator()<TinyDIP::Image<TinyDIP::RGB>>(input_arg, workspace_));
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
        CommandBundle{"load_workspace", "Load memory variables from a directory bundle.", IndependentSchema, LoadWorkspaceHandler{workspace}},
        CommandBundle{"print", "Print the contents of a memory variable.", TerminatorSchema, PrintHandler{workspace}},
        CommandBundle{"rand", "Generate random multi-dimensional image with specified URBG.", GeneratorSchema, RandHandler{workspace}},
        CommandBundle{"read", "Read an image from disk into a memory variable.", GeneratorSchema, ReadHandler{workspace}},
        CommandBundle{"save_workspace", "Save all memory variables to a directory bundle.", IndependentSchema, SaveWorkspaceHandler{workspace}},
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
