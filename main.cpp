/* Developed by Jimmy Hu */
/* Refactored for CLI Application capability */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  main.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o main -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./main && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall main.cpp -o main 


//#define USE_BOOST_ITERATOR
//#define USE_BOOST_SERIALIZATION

#include "main.h"


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
//  Generic Meta Handler strictly refactoring transform commands like abs, bicubic_resize, dct2, idct2, and lanczos_resample
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

//  MetaScalarHandler template struct implementation
//  Generic Meta Handler strictly refactoring scalar reduction commands like max, min, and sum
template <
    std::size_t MinArgs,
    typename SetupFun,
    typename ArgsContainer = std::vector<std::string_view>,
    typename CheckingTypes = master_data_types
>
requires(std::invocable<SetupFun, const ArgsContainer&, const std::string_view, std::ostream&>)
struct MetaScalarHandler
{
    std::string_view usage_string_;
    std::string_view op_name_;
    std::string_view capitalized_op_name_;
    SetupFun setup_fun_;

    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void operator()(Workspace& workspace, std::span<const std::string_view> args, std::ostream& os = std::cout, ImageLoaderFun&& image_loader_fun = ImageLoaderFun{}) const
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
        std::string_view output_arg = "";
        
        if (std::ranges::size(filtered_args) > 1)
        {
            output_arg = filtered_args[1];
        }

        auto core_processor = setup_fun_(filtered_args, policy_str, os);

        std::optional<std::any> final_result_opt;

        // Polymorphic lambda to cleanly execute the algorithm dynamically independent of data type
        auto process_scalar = [&]<typename DataT>(DataT&& input_data)
        {
            final_result_opt = core_processor(std::forward<DataT>(input_data));
        };

        if (!dispatch_data_operation<CheckingTypes>(input_arg, workspace, image_loader_fun, process_scalar))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
            return;
        }

        if (final_result_opt.has_value())
        {
            std::any scalar_result_any = std::move(*final_result_opt);

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
                            workspace.store(output_arg.substr(1), scalar_result);
                            os << "Saved " << op_name_ << " result to " << output_arg << "\n";
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
                        else if constexpr (requires { os << scalar_result; })
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
        }
    }
};

//  make_meta_scalar_handler template function implementation
template <std::size_t MinArgs, typename CheckingTypes = master_data_types, typename SetupFun, typename ArgsContainer = std::vector<std::string_view>>
constexpr auto make_meta_scalar_handler(std::string_view usage, std::string_view op_name, std::string_view capitalized_op_name, SetupFun&& setup)
{
    return MetaScalarHandler<MinArgs, std::remove_cvref_t<SetupFun>, ArgsContainer, CheckingTypes>{
        usage, op_name, capitalized_op_name, std::forward<SetupFun>(setup)
    };
}

namespace handlers
{
    //  abs function implementation
    constexpr auto abs(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2, master_data_types>(
            "abs [execution_policy] <input_data | $var> <output_var | $var>",
            [](std::span<const std::string_view> filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                os << "Calculating abs of " << filtered_args[0];
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [policy_str, &os]<typename DataT>(DataT&& data) -> std::any
                {
                    using DecayedDataT = std::remove_cvref_t<DataT>;

                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            if constexpr (requires { TinyDIP::abs(std::forward<DataT>(data)); })
                            {
                                return TinyDIP::abs(std::forward<DataT>(data));
                            }
                            else
                            {
                                throw std::invalid_argument("Input image type does not support abs operation.");
                                return std::any{};
                            }
                        }
                        else if constexpr (std::ranges::input_range<DecayedDataT>)
                        {
                            return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                [](auto&& element) 
                                { 
                                    return TinyDIP::generic_abs(std::forward<decltype(element)>(element));
                                },
                                std::forward<DataT>(data)
                            );
                        }
                        else
                        {
                            return TinyDIP::generic_abs(std::forward<DataT>(data));
                        }
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            if constexpr (requires { TinyDIP::abs(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                            {
                                return TinyDIP::abs(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
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
                                    return TinyDIP::generic_abs(std::forward<decltype(element)>(element));
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

    //  append_element template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void append_element(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        if (std::ranges::size(args) < 3)
        {
            os << "Usage: append_element <input_container | $var> <element | $var> <output_var | $var>\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view elem_arg = args[1];
        const std::string_view output_arg = args[2];

        if (!output_arg.starts_with('$'))
        {
            os << "Error: Output must be a memory variable starting with '$'.\n";
            return;
        }

        os << "Appending element " << elem_arg << " to " << input_arg << "...\n";

        auto process_append = [&]<typename CandidateType>(CandidateType&& candidate)
        {
            using DecayedT = std::remove_cvref_t<CandidateType>;

            // Mathematically verify that the type is a registered container that supports back insertion
            if constexpr (is_vector_v<DecayedT> || is_deque_v<DecayedT> || is_list_v<DecayedT>)
            {
                using ElementT = typename DecayedT::value_type;
                
                if (!elem_arg.starts_with('$'))
                {
                    os << "Error: Element must be a memory variable starting with '$'.\n";
                    return;
                }

                const std::string_view elem_name = elem_arg.substr(1);
                
                // Directly retrieve the perfectly matched complex object from the workspace
                if (const ElementT* element_ptr = workspace.template retrieve<ElementT>(elem_name))
                {
                    DecayedT new_container = candidate; // Copy original container completely independently
                    
                    // Utilizing emplace_back to directly construct the complex object in place
                    // optimizing memory allocation and avoiding unnecessary temporary copies natively.
                    new_container.emplace_back(*element_ptr);
                    
                    workspace.store(output_arg.substr(1), std::move(new_container));
                    os << "Saved updated container to " << output_arg << ".\n";
                }
                else
                {
                    os << "Error: Element variable $" << elem_name << " not found or type mismatch. Expected exact type: [" << get_type_name<ElementT>() << "].\n";
                }
            }
            else
            {
                os << "Error: Input type [" << get_type_name<DecayedT>() << "] is not a supported container type that supports appending.\n";
            }
        };

        // Leverage tuple_cat_t to flawlessly support both scalar containers and newly registered image containers!
        using AllContainerTypes = tuple_cat_t<master_data_types, master_image_container_types>;

        if (!dispatch_data_operation<AllContainerTypes>(input_arg, workspace, image_loader_fun, process_append))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }
    
    //  bicubic_resize function implementation
    constexpr auto bicubic_resize(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<4>(
            "bicubic_resize [execution_policy] <input_img | $var> <output_img | $var> <width> <height>",
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                const std::size_t width = parse_arg<std::size_t>(filtered_args[2]);
                const std::size_t height = parse_arg<std::size_t>(filtered_args[3]);

                os << "Resizing " << filtered_args[0] << " to " << width << "x" << height;
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [width, height, policy_str, &os]<typename ImageType>(ImageType && img) -> std::any
                {
                    auto exec_default = [&]() -> std::any
                        {
                            return TinyDIP::copyResizeBicubic(std::forward<ImageType>(img), width, height);
                        };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (requires { TinyDIP::copyResizeBicubic(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height); })
                        {
                            return TinyDIP::copyResizeBicubic(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height);
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

    //  construct_rgb template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader,
        typename ImageSaverFun = MetaImageIO::Saver
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&> &&
              std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::RGB>&&>)
    constexpr void construct_rgb(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{},
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
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

        if (std::ranges::size(filtered_args) < 4)
        {
            os << "Usage: constructRGB [execution_policy] <R_img | $var> <G_img | $var> <B_img | $var> <output_img | $var>\n";
            os << "       Optional Execution policies: seq, par, par_unseq, unseq\n";
            return;
        }

        const std::string_view r_arg = filtered_args[0];
        const std::string_view g_arg = filtered_args[1];
        const std::string_view b_arg = filtered_args[2];
        const std::string_view output_arg = filtered_args[3];

        os << "Constructing RGB image from " << r_arg << ", " << g_arg << ", " << b_arg;
        if (!std::ranges::empty(policy_str))
        {
            os << " (Policy: " << policy_str << ")";
        }
        os << "...\n";

        // Restrict allowed types to strictly 8-bit unsigned integers to mathematically prevent template combinatorial explosions
        using AllowedTypes = std::tuple<TinyDIP::Image<std::uint8_t>, TinyDIP::Image<unsigned char>>;

        auto process_r = [&]<typename ImgR>(ImgR&& img_r)
        {
            auto process_g = [&]<typename ImgG>(ImgG&& img_g)
            {
                auto process_b = [&]<typename ImgB>(ImgB&& img_b)
                {
                    if constexpr (
                        (!std::same_as<std::remove_cvref_t<ImgR>, TinyDIP::Image<std::uint8_t>> &&
                         !std::same_as<std::remove_cvref_t<ImgR>, TinyDIP::Image<unsigned char>>) ||
						(!std::same_as<std::remove_cvref_t<ImgG>, TinyDIP::Image<std::uint8_t>> &&
						 !std::same_as<std::remove_cvref_t<ImgG>, TinyDIP::Image<unsigned char>>) ||
						(!std::same_as<std::remove_cvref_t<ImgB>, TinyDIP::Image<std::uint8_t>>&&
                         !std::same_as<std::remove_cvref_t<ImgB>, TinyDIP::Image<unsigned char>>)
                        )
                    {
                        throw std::invalid_argument("R / G / B plane image must be 8-bit unsigned integer type.");
                    }
                    else
                    {
                        const std::size_t width = img_r.getWidth();
                        const std::size_t height = img_r.getHeight();

                        if (width != img_g.getWidth() || height != img_g.getHeight() ||
                            width != img_b.getWidth() || height != img_b.getHeight())
                        {
                            throw std::invalid_argument("Dimension mismatch among R, G, B plane images.");
                        }

                        TinyDIP::Image<TinyDIP::RGB> output_image(width, height);

                        auto exec_default = [&]() -> std::any
                            {
                                for (std::size_t y = 0; y < height; ++y)
                                {
                                    for (std::size_t x = 0; x < width; ++x)
                                    {
                                        TinyDIP::RGB pixel{};
                                        pixel.channels[0] = static_cast<std::uint8_t>(img_r.at(x, y));
                                        pixel.channels[1] = static_cast<std::uint8_t>(img_g.at(x, y));
                                        pixel.channels[2] = static_cast<std::uint8_t>(img_b.at(x, y));
                                        output_image.at(x, y) = pixel;
                                    }
                                }
                                return output_image;
                            };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            auto indices = std::views::iota(std::size_t{ 0 }, width * height);
                            std::for_each(
                                std::forward<ExecPolicy>(exec_policy),
                                std::ranges::begin(indices),
                                std::ranges::end(indices),
                                [&](const std::size_t idx)
                                {
                                    const std::size_t y = idx / width;
                                    const std::size_t x = idx % width;

                                    TinyDIP::RGB pixel{};
                                    pixel.channels[0] = static_cast<std::uint8_t>(img_r.at(x, y));
                                    pixel.channels[1] = static_cast<std::uint8_t>(img_g.at(x, y));
                                    pixel.channels[2] = static_cast<std::uint8_t>(img_b.at(x, y));
                                    output_image.at(x, y) = pixel;
                                }
                            );
                            return output_image;
                        };

                        std::any final_result = dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                        image_saver_fun(output_arg, workspace, std::move(std::any_cast<TinyDIP::Image<TinyDIP::RGB>&>(final_result)));
                        os << "Saved to " << output_arg << "\n";
                    }
                };

                if (!dispatch_data_operation<AllowedTypes>(b_arg, workspace, image_loader_fun, process_b))
                {
                    os << "Error: Memory variable for B plane not found or not an 8-bit unsigned integer type. Use 'im2uint8' first.\n";
                }
            };

            if (!dispatch_data_operation<AllowedTypes>(g_arg, workspace, image_loader_fun, process_g))
            {
                os << "Error: Memory variable for G plane not found or not an 8-bit unsigned integer type. Use 'im2uint8' first.\n";
            }
        };

        if (!dispatch_data_operation<AllowedTypes>(r_arg, workspace, image_loader_fun, process_r))
        {
            os << "Error: Memory variable for R plane not found or not an 8-bit unsigned integer type. Use 'im2uint8' first.\n";
        }
    }

	//  create_container template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void create_container(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: create_container <prototype_element | $var> <output_container | $var>\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];

        if (!output_arg.starts_with('$'))
        {
            os << "Error: Output must be a memory variable starting with '$'.\n";
            return;
        }

        os << "Creating container based on type of " << input_arg << "...\n";

        auto process_create = [&]<typename CandidateType>(CandidateType&& candidate)
        {
            using DecayedT = std::remove_cvref_t<CandidateType>;
            
            std::vector<DecayedT> new_vec;
            // Utilizing emplace_back to insert the initial candidate prototype safely and efficiently
            new_vec.emplace_back(std::forward<CandidateType>(candidate));

            workspace.store(output_arg.substr(1), std::move(new_vec));
            os << "Created container and added initial element. Saved [std::vector<" << get_type_name<DecayedT>() << ">] to " << output_arg << ".\n";
        };

        using AllElementTypes = tuple_cat_t<master_data_types>;

        if (!dispatch_data_operation<AllElementTypes>(input_arg, workspace, image_loader_fun, process_create))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }

    //  create_image_with_initial_value template function implementation
    template <
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<double>&&> ImageSaverFun = MetaImageIO::Saver
    >
    constexpr void create_image_with_initial_value(
        Workspace& workspace,
        std::span<const std::string_view> args,
        const double initial_value,
        const std::string_view command_name,
        std::ostream& os = std::cout,
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: " << command_name << " <output_img | $var> <dim1> [dim2] [dim3] ...\n";
            return;
        }

        const std::string_view output_arg = args[0];
        
        std::vector<std::size_t> sizes;
        sizes.reserve(std::ranges::size(args) - 1);
        std::size_t total_elements = 1;
        
        for (std::size_t i = 1; i < std::ranges::size(args); ++i)
        {
            const std::size_t dim = parse_arg<std::size_t>(args[i]);
            sizes.emplace_back(dim);
            total_elements *= dim;
        }

        os << "Generating " << command_name << " image with dimensions: ";
        for (std::size_t i = 0; i < std::ranges::size(sizes); ++i)
        {
            os << sizes[i];
            if (i + 1 < std::ranges::size(sizes))
            {
                os << " x ";
            }
        }
        os << "...\n";

        std::vector<double> data(total_elements, initial_value);
        TinyDIP::Image<double> output_img(data, sizes);

        if constexpr (requires { output_img.setAllValue(initial_value); })
        {
            output_img.setAllValue(initial_value);
        }

        image_saver_fun(output_arg, workspace, std::move(output_img));
        os << "Saved to " << output_arg << "\n";
    }

    //  dct2 function implementation
    constexpr void dct2(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2>(
            "dct2 [execution_policy] <input_img | $var> <output_img | $var>",
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                os << "Calculating DCT-2 for " << filtered_args[0];
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [policy_str, &os]<typename ImageType>(ImageType && img) -> std::any
                {
                    auto exec_default = [&]() -> std::any
                        {
                            return TinyDIP::dct2(std::forward<ImageType>(img));
                        };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
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
        );

        transform_handler(workspace, args, os);
    }

    //  estimate_gaussian_params_2d function implementation
    constexpr void estimate_gaussian_params_2d(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        auto transform_handler = make_meta_scalar_handler<2>(
            "estimate_gaussian_params_2d [execution_policy] <input_img | $var> <output_var | $var> [max_iterations=1000] [tolerance=1e-7]", 
            "estimate_gaussian_params_2d", "Estimate Gaussian Parameters 2D", 
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                std::size_t max_iterations = 1000;
                double tolerance = 1e-7;
                if (std::ranges::size(filtered_args) > 2)
                {
                    max_iterations = parse_arg<std::size_t>(filtered_args[2]);
                }
                if (std::ranges::size(filtered_args) > 3)
                {
                    tolerance = parse_arg<double>(filtered_args[3]);
                }

                os << "Estimating 2D Gaussian parameters for " << filtered_args[0] << "...\n";

                return [max_iterations, tolerance, policy_str, &os]<typename DataT>(DataT&& data) -> std::any
                {
                    using DecayedDataT = std::remove_cvref_t<DataT>;

                    if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                    {
                        using ElementT = TinyDIP::get_deep_scalar_t<DecayedDataT>;
                        
                        if constexpr (std::is_arithmetic_v<ElementT> && !TinyDIP::is_bool_data_v<ElementT>)
                        {
                            auto exec_default = [&]() -> std::any
                            {
                                if constexpr (requires { TinyDIP::estimate_gaussian_parameters_2d(std::execution::seq, std::forward<DataT>(data), max_iterations, tolerance); })
                                {
                                    return TinyDIP::estimate_gaussian_parameters_2d(std::execution::seq, std::forward<DataT>(data), max_iterations, tolerance);
                                }
                                else if constexpr (requires { TinyDIP::estimate_gaussian_parameters_2d(std::forward<DataT>(data), max_iterations, tolerance); })
                                {
                                    // Just in case the backend doesn't support execution policies natively yet
                                    return TinyDIP::estimate_gaussian_parameters_2d(std::forward<DataT>(data), max_iterations, tolerance);
                                }
                                else
                                {
                                    throw std::invalid_argument(std::string("Input image type [") + std::string(get_type_name<DecayedDataT>()) + "] does not support estimate_gaussian_parameters_2d.");
                                    return std::any{};
                                }
                            };

                            auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                                requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                            {
                                if constexpr (requires { TinyDIP::estimate_gaussian_parameters_2d(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), max_iterations, tolerance); })
                                {
                                    return TinyDIP::estimate_gaussian_parameters_2d(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), max_iterations, tolerance);
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
                        }
                        else
                        {
                            throw std::invalid_argument("Input image must have arithmetic elements to estimate Gaussian parameters.");
                            return std::any{};
                        }
                    }
                    else
                    {
                        throw std::invalid_argument("Input data type must be an Image.");
                        return std::any{};
                    }
                };
            }
        );

        transform_handler(workspace, args, os);
    }

    //  gaussian_figure_2d template function implementation
    template <
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<double>&&> ImageSaverFun = MetaImageIO::Saver
    >
    constexpr void gaussian_figure_2d(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        if (std::ranges::size(args) < 6)
        {
            os << "Usage: gaussian_figure_2d <output_img | $var> <width> <height> <center_x> <center_y> <sigma>\n";
            return;
        }

        const std::string_view output_arg = args[0];
        const std::size_t width = parse_arg<std::size_t>(args[1]);
        const std::size_t height = parse_arg<std::size_t>(args[2]);
        const std::size_t center_x = parse_arg<std::size_t>(args[3]);
        const std::size_t center_y = parse_arg<std::size_t>(args[4]);
        const double sigma = parse_arg<double>(args[5]);

        os << "Generating Gaussian Figure 2D with dimensions: " << width << " x " << height 
           << ", center: (" << center_x << ", " << center_y << "), sigma: " << sigma << "...\n";

        // Generate the 2D Gaussian Figure utilizing the native double precision wrapper directly from the TinyDIP engine
        auto output_img = TinyDIP::gaussianFigure2D(width, height, center_x, center_y, sigma);

        image_saver_fun(output_arg, workspace, std::move(output_img));
        os << "Saved to " << output_arg << "\n";
    }

    //  get_element template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void get_element(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        if (std::ranges::size(args) < 3)
        {
            os << "Usage: get_element <input_container | $var> <output_var | $var> <index>\n";
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

        os << "Extracting element at index " << index << " from " << input_arg << "...\n";

        auto process_element = [&]<typename CandidateType>(CandidateType&& candidate)
        {
            using DecayedT = std::remove_cvref_t<CandidateType>;

            // Verify the type is a mathematically registered container
            if constexpr (is_vector_v<DecayedT> || is_deque_v<DecayedT> || is_list_v<DecayedT> || is_std_array_v<DecayedT>)
            {
                if (index >= std::ranges::size(candidate))
                {
                    os << "Error: Index " << index << " is out of bounds for container of size " << std::ranges::size(candidate) << ".\n";
                    return;
                }

                // Cleanly traverse using ranges to natively support both vectors and lists
                auto it = std::ranges::begin(candidate);
                std::ranges::advance(it, index);
                
                // Store the extracted element safely into the workspace (invokes copy constructor to preserve independence)
                workspace.store(output_arg.substr(1), *it);
                os << "Saved element to " << output_arg << ".\n";
            }
            else
            {
                os << "Error: Input type [" << get_type_name<DecayedT>() << "] is not a supported container type.\n";
            }
        };

        // Leverage tuple_cat_t to flawlessly support both scalar containers and newly registered image containers!
        using AllContainerTypes = tuple_cat_t<master_data_types, master_image_container_types>;

        if (!dispatch_data_operation<AllContainerTypes>(input_arg, workspace, image_loader_fun, process_element))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }

    //  getPlane_channel_description function implementation
    constexpr auto getPlane_channel_description(const std::size_t channel_index)
    {
        if (channel_index == 0)
        {
            "getRplane [execution_policy] <input_img | $var> <output_img | $var>";
        }
        else if (channel_index == 1)
        {
            return "getGplane [execution_policy] <input_img | $var> <output_img | $var>";
        }
        else if (channel_index == 2)
        {
            return "getBplane [execution_policy] <input_img | $var> <output_img | $var>";
        }
        return "";
    }

    //  getPlane function implementation
    constexpr void getPlane(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        const std::size_t channel_index = 0
    )
    {
        auto transform_handler = make_meta_transform_handler<2>(
                getPlane_channel_description(channel_index), 
                [&](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    os << "Extracting channel " << std::to_string(channel_index) << " of " << filtered_args[0];
                    if (!std::ranges::empty(policy_str))
                    {
                        os << " (Policy: " << policy_str << ")";
                    }
                    os << "...\n";

                    return [policy_str, &os, channel_index]<typename ImageType>(ImageType&& img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (requires { TinyDIP::getPlane(std::forward<ImageType>(img), channel_index); })
                            {
                                return TinyDIP::getPlane(std::forward<ImageType>(img), channel_index);
                            }
                            else
                            {
                                throw std::invalid_argument("Input image does not support multi-channel plane extraction.");
                                return std::any{};
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::getPlane(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), channel_index); })
                            {
                                return TinyDIP::getPlane(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), channel_index);
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

	//  get_sift_potential_keypoint template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void get_sift_potential_keypoint(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        std::string_view policy_str = "";
        std::vector<std::string_view> filtered_args;
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
            os << "Usage: get_sift_potential_keypoint [execution_policy] <input_img | $var> <output_var | $var> "
               << "[octaves=4] [levels=5] [sigma=1.6] [k=1.414] [contrast_thresh=8.0] [edge_thresh=12.1] [resampling=bicubic]\n";
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view output_arg = filtered_args[1];

        if (!output_arg.starts_with('$'))
        {
            os << "Error: Output must be a memory variable starting with '$'.\n";
            return;
        }

        const std::size_t octaves = (std::ranges::size(filtered_args) > 2) ? parse_arg<std::size_t>(filtered_args[2]) : 4;
        const std::size_t levels = (std::ranges::size(filtered_args) > 3) ? parse_arg<std::size_t>(filtered_args[3]) : 5;
        const double sigma = (std::ranges::size(filtered_args) > 4) ? parse_arg<double>(filtered_args[4]) : 1.6;
        const double k = (std::ranges::size(filtered_args) > 5) ? parse_arg<double>(filtered_args[5]) : std::numbers::sqrt2_v<double>;
        const double contrast = (std::ranges::size(filtered_args) > 6) ? parse_arg<double>(filtered_args[6]) : 8.0;
        const double edge = (std::ranges::size(filtered_args) > 7) ? parse_arg<double>(filtered_args[7]) : 12.1;
        const std::string_view resample_str = (std::ranges::size(filtered_args) > 8) ? filtered_args[8] : "bicubic";

        os << "Extracting SIFT potential keypoints from " << input_arg;
        if (!std::ranges::empty(policy_str))
        {
            os << " (Policy: " << policy_str << ")";
        }
        os << "...\n";

        auto process_sift = [&]<typename ImageType>(ImageType&& input_img)
        {
            using DecayedImageType = std::remove_cvref_t<ImageType>;

            if constexpr (TinyDIP::is_bool_data_v<DecayedImageType> || TinyDIP::is_complex_data_v<DecayedImageType>)
            {
                os << "Error: Input image type [" << get_type_name<DecayedImageType>() << "] does not support SIFT keypoint extraction.\n";
                return;
            }
            else
            {
                // Helper to cleanly execute SIFT after mathematical elevation to double precision
                auto process_sift_impl = [&]<typename T>(T&& double_img)
                {
                    using ImplDecayedT = std::remove_cvref_t<T>;
                    // Fetch the exact elevated element type (e.g., double or RGB_DOUBLE)
                    using ImplElementT = std::remove_cvref_t<decltype(double_img.at(0, 0))>;

                    // Create a default polymorphic lambda acting as the ResamplingFunc bridge
                    auto resample_fn = [resample_str](const auto& img, const std::size_t w, const std::size_t h) 
                    {
                        if (resample_str == "lanczos") 
                        {
                            return TinyDIP::lanczos_resample(img, w, h);
                        } 
                        else if (resample_str == "bicubic") 
                        {
                            return TinyDIP::copyResizeBicubic(img, w, h);
                        }
                        else
                        {
                            throw std::invalid_argument("Unknown resampling method. Use 'lanczos' or 'bicubic'.");
                        }
                    };

                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (requires { TinyDIP::SIFT_impl::get_potential_keypoint(std::forward<T>(double_img), octaves, levels, sigma, k, static_cast<ImplElementT>(contrast), static_cast<ImplElementT>(edge), resample_fn); })
                        {
                            return TinyDIP::SIFT_impl::get_potential_keypoint(std::forward<T>(double_img), octaves, levels, sigma, k, static_cast<ImplElementT>(contrast), static_cast<ImplElementT>(edge), resample_fn);
                        }
                        else
                        {
                            throw std::invalid_argument(std::string("Input image type [") + std::string(get_type_name<ImplDecayedT>()) + "] does not support SIFT keypoint extraction.");
                            return std::any{};
                        }
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (requires { TinyDIP::SIFT_impl::get_potential_keypoint(std::forward<ExecPolicy>(exec_policy), std::forward<T>(double_img), octaves, levels, sigma, k, static_cast<ImplElementT>(contrast), static_cast<ImplElementT>(edge), resample_fn); })
                        {
                            return TinyDIP::SIFT_impl::get_potential_keypoint(std::forward<ExecPolicy>(exec_policy), std::forward<T>(double_img), octaves, levels, sigma, k, static_cast<ImplElementT>(contrast), static_cast<ImplElementT>(edge), resample_fn);
                        }
                        else
                        {
                            if (!std::ranges::empty(policy_str))
                            {
                                os << "Warning: Execution policy requested but not supported. Falling back to default.\n";
                            }
                            return exec_default();
                        }
                    };

                    return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                };

                using RawScalarT = TinyDIP::get_deep_scalar_t<DecayedImageType>;

                // Elevate the input image strictly to double precision before executing the mathematical pipeline
                if constexpr (std::same_as<RawScalarT, double>)
                {
                    std::any result = process_sift_impl(std::forward<ImageType>(input_img));
                    workspace.store(output_arg.substr(1), std::move(result));
                    os << "Saved keypoints to " << output_arg << ".\n";
                }
                else if constexpr (requires { TinyDIP::im2double(std::forward<ImageType>(input_img)); })
                {
                    std::any result = process_sift_impl(TinyDIP::im2double(std::forward<ImageType>(input_img)));
                    workspace.store(output_arg.substr(1), std::move(result));
                    os << "Saved keypoints to " << output_arg << ".\n";
                }
                else
                {
                    os << "Error: Input image type [" << get_type_name<DecayedImageType>() << "] cannot be converted to double precision for SIFT.\n";
                }
            }
        };

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_sift))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }

    //  grid_generator template function implementation
    template <
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::RGB>&&> ImageSaverFun = MetaImageIO::Saver
    >
    constexpr void grid_generator(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        std::string_view policy_str = "";
        std::vector<std::string_view> filtered_args;
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
            os << "Usage: grid [execution_policy] <output_bmp | $var> <width> <height> [grid_size=10]\n";
            os << "       Optional Execution policies: seq, par, par_unseq, unseq\n";
            return;
        }

        const std::string_view output_arg = filtered_args[0];
        const std::size_t width = parse_arg<std::size_t>(filtered_args[1]);
        const std::size_t height = parse_arg<std::size_t>(filtered_args[2]);
        std::size_t grid_size = 10;
        
        if (std::ranges::size(filtered_args) > 3)
        {
            grid_size = parse_arg<std::size_t>(filtered_args[3]);
        }

        if (!std::ranges::empty(policy_str))
        {
            os << "Generating grid image (" << width << "x" << height << ") with cell size " << grid_size << " (Policy: " << policy_str << ")...\n";
        }
        else
        {
            os << "Generating grid image (" << width << "x" << height << ") with cell size " << grid_size << "...\n";
        }

        auto exec_default = [&]() -> std::any
        {
            TinyDIP::Image<TinyDIP::RGB> output_img(width, height);
            for (std::size_t y = 0; y < height; ++y)
            {
                for (std::size_t x = 0; x < width; ++x)
                {
                    TinyDIP::RGB pixel{};
                    if (x % grid_size == 0 || y % grid_size == 0)
                    {
                        // Grid line (Black)
                        pixel.channels[0] = 0;
                        pixel.channels[1] = 0;
                        pixel.channels[2] = 0;
                    }
                    else
                    {
                        // Background (White)
                        pixel.channels[0] = 255;
                        pixel.channels[1] = 255;
                        pixel.channels[2] = 255;
                    }
                    output_img.at(x, y) = pixel;
                }
            }
            return output_img;
        };

        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
        {
            TinyDIP::Image<TinyDIP::RGB> output_img(width, height);
            auto indices = std::views::iota(std::size_t{0}, width * height);
            std::for_each(
                std::forward<ExecPolicy>(exec_policy),
                std::ranges::begin(indices),
                std::ranges::end(indices),
                [&](const std::size_t idx)
                {
                    const std::size_t y = idx / width;
                    const std::size_t x = idx % width;

                    TinyDIP::RGB pixel{};
                    if (x % grid_size == 0 || y % grid_size == 0)
                    {
                        // Grid line (Black)
                        pixel.channels[0] = 0;
                        pixel.channels[1] = 0;
                        pixel.channels[2] = 0;
                    }
                    else
                    {
                        // Background (White)
                        pixel.channels[0] = 255;
                        pixel.channels[1] = 255;
                        pixel.channels[2] = 255;
                    }
                    output_img.at(x, y) = pixel;
                }
            );
            return output_img;
        };

        std::any final_result = dispatch_policy_string(policy_str, exec_policy, exec_default, os);
        image_saver_fun(output_arg, workspace, std::move(std::any_cast<TinyDIP::Image<TinyDIP::RGB>&>(final_result)));
        os << "Saved to " << output_arg << "\n";
    }

    //  help function implementation
    constexpr void help(
        const CommandRegistry& registry,
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        (void)workspace;
        (void)args;
        registry.list_commands(os);
    }

    //  hsv2rgb function implementation
    constexpr void hsv2rgb(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2, master_data_types>(
                "hsv2rgb [execution_policy] <input_data | $var> <output_var | $var>", 
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    os << "Converting " << filtered_args[0] << " to RGB";
                    if (!std::ranges::empty(policy_str))
                    {
                        os << " (Policy: " << policy_str << ")";
                    }
                    os << "...\n";

                    return [policy_str, &os]<typename DataT>(DataT&& data) -> std::any
                    {
                        using DecayedDataT = std::remove_cvref_t<DataT>;

                        if constexpr (TinyDIP::is_bool_data_v<DecayedDataT>)
                        {
                            throw std::invalid_argument("Input data type (bool) does not support hsv2rgb conversion.");
                            return std::any{};
                        }
                        else
                        {
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
                                        return std::any{};
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
                                        return std::any{};
                                    }
                                }
                                else
                                {
                                    throw std::invalid_argument("Input data type does not support hsv2rgb conversion.");
                                    return std::any{};
                                }
                            };

                            auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
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

                            return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                        }
                    };
                }
            );

        transform_handler(workspace, args, os);
    }

    //  idct2 function implementation
    constexpr void idct2(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2>(
                "idct2 [execution_policy] <input_img | $var> <output_img | $var>", 
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    os << "Calculating Inverse DCT-2 for " << filtered_args[0];
                    if (!std::ranges::empty(policy_str))
                    {
                        os << " (Policy: " << policy_str << ")";
                    }
                    os << "...\n";

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

                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            );
            
        transform_handler(workspace, args, os);
    }

    //  im2double function implementation
    constexpr void im2double(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2, master_data_types>(
                "im2double [execution_policy] <input_data | $var> <output_var | $var>", 
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    os << "Converting " << filtered_args[0] << " to double";
                    if (!std::ranges::empty(policy_str))
                    {
                        os << " (Policy: " << policy_str << ")";
                    }
                    os << "...\n";

                    return [policy_str, &os]<typename DataT>(DataT&& data) -> std::any
                    {
                        using DecayedDataT = std::remove_cvref_t<DataT>;

                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                            {
                                if constexpr (requires { TinyDIP::im2double(std::forward<DataT>(data)); })
                                {
                                    return TinyDIP::im2double(std::forward<DataT>(data));
                                }
                                else
                                {
                                    throw std::invalid_argument("Input image type does not support im2double conversion.");
                                    return std::any{};
                                }
                            }
                            else if constexpr (std::ranges::input_range<DecayedDataT>)
                            {
                                if constexpr (requires { TinyDIP::im2double(*std::ranges::begin(data)); })
                                {
                                    return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                        [](auto&& element) 
                                        { 
                                            return TinyDIP::im2double(std::forward<decltype(element)>(element));
                                        },
                                        std::forward<DataT>(data)
                                    );
                                }
                                else
                                {
                                    throw std::invalid_argument("Input container type does not support im2double conversion.");
                                    return std::any{};
                                }
                            }
                            else
                            {
                                throw std::invalid_argument("Input data type does not support im2double conversion.");
                                return std::any{};
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                            {
                                if constexpr (requires { TinyDIP::im2double(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                                {
                                    return TinyDIP::im2double(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
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
                                if constexpr (requires { TinyDIP::im2double(*std::ranges::begin(data)); })
                                {
                                    return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                        std::forward<ExecPolicy>(exec_policy),
                                        [](auto&& element) 
                                        { 
                                            return TinyDIP::im2double(std::forward<decltype(element)>(element));
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

                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            );

        transform_handler(workspace, args, os);
    }

    //  im2uint8 function implementation
    constexpr void im2uint8(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2, master_data_types>(
                "im2uint8 [execution_policy] <input_data | $var> <output_var | $var>",
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    os << "Converting " << filtered_args[0] << " to uint8";
                    if (!std::ranges::empty(policy_str))
                    {
                        os << " (Policy: " << policy_str << ")";
                    }
                    os << "...\n";

                    return[policy_str, &os]<typename DataT>(DataT && data) -> std::any
                    {
                        using DecayedDataT = std::remove_cvref_t<DataT>;

                        auto exec_default = [&]() -> std::any
                            {
                                if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                                {
                                    if constexpr (requires { TinyDIP::im2uint8(std::forward<DataT>(data)); })
                                    {
                                        return TinyDIP::im2uint8(std::forward<DataT>(data));
                                    }
                                    else
                                    {
                                        throw std::invalid_argument("Input image type does not support im2uint8 conversion.");
                                        return std::any{};
                                    }
                                }
                                else if constexpr (std::ranges::input_range<DecayedDataT>)
                                {
                                    if constexpr (requires { TinyDIP::im2uint8(*std::ranges::begin(data)); })
                                    {
                                        return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                            [](auto&& element)
                                            {
                                                return TinyDIP::im2uint8(std::forward<decltype(element)>(element));
                                            },
                                            std::forward<DataT>(data)
                                        );
                                    }
                                    else
                                    {
                                        throw std::invalid_argument("Input container type does not support im2uint8 conversion.");
                                        return std::any{};
                                    }
                                }
                                else
                                {
                                    throw std::invalid_argument("Input data type does not support im2uint8 conversion.");
                                    return std::any{};
                                }
                            };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                            {
                                if constexpr (requires { TinyDIP::im2uint8(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                                {
                                    return TinyDIP::im2uint8(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
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
                                if constexpr (requires { TinyDIP::im2uint8(*std::ranges::begin(data)); })
                                {
                                    return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                        std::forward<ExecPolicy>(exec_policy),
                                        [](auto&& element)
                                        {
                                            return TinyDIP::im2uint8(std::forward<decltype(element)>(element));
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

                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            );
        transform_handler(workspace, args, os);
    }

    //  info template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void info(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
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

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_info))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }

    //  lanczos_resample function implementation
    constexpr void lanczos_resample(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<4>(
                "lanczos_resample [execution_policy] <input_img | $var> <output_img | $var> <width> <height> [a=3]", 
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    const std::size_t width = parse_arg<std::size_t>(filtered_args[2]);
                    const std::size_t height = parse_arg<std::size_t>(filtered_args[3]);
                    std::size_t a = 3;
                    
                    if (std::ranges::size(filtered_args) >= 5)
                    {
                        a = parse_arg<std::size_t>(filtered_args[4]);
                    }

                    os << "Resizing " << filtered_args[0] << " to " << width << "x" << height << " with Lanczos radius " << a;
                    if (!std::ranges::empty(policy_str))
                    {
                        os << " (Policy: " << policy_str << ")";
                    }
                    os << "...\n";

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
                            {
                                return TinyDIP::lanczos_resample(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height, static_cast<int>(a));
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

    //  max function implementation
    constexpr void max(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_scalar_handler<1>(
                "max <input_data | $var> [output_var | $var]", 
                "max", "Max", 
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Warning: Execution policy '" << policy_str << "' is ignored for 'max'.\n";
                    }
                    os << "Calculating max of " << filtered_args[0] << "...\n";

                    return []<typename DataT>(DataT&& data) -> std::any
                    {
                        using DecayedDataT = std::remove_cvref_t<DataT>;
                        
                        if constexpr (TinyDIP::is_complex_data_v<DecayedDataT>)
                        {
                            throw std::invalid_argument("Input data type (complex) does not support max (elements are not comparable).");
                            return std::any{};
                        }
                        else if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            if constexpr (requires { TinyDIP::max(std::forward<DataT>(data)); })
                            {
                                return TinyDIP::max(std::forward<DataT>(data));
                            }
                            else
                            {
                                throw std::invalid_argument("Input image type does not support max.");
                                return std::any{};
                            }
                        }
                        else
                        {
                            if constexpr (requires { std::ranges::max(std::forward<DataT>(data)); })
                            {
                                return std::ranges::max(std::forward<DataT>(data));
                            }
                            else
                            {
                                throw std::invalid_argument("Input container type does not support max (elements are not comparable).");
                                return std::any{};
                            }
                        }
                    };
                }
            );

        transform_handler(workspace, args, os);
    }

    //  min function implementation
    constexpr void min(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_scalar_handler<1>(
            "min <input_data | $var> [output_var | $var]", 
            "min", "Min", 
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                if (!std::ranges::empty(policy_str))
                {
                    os << "Warning: Execution policy '" << policy_str << "' is ignored for 'min'.\n";
                }
                os << "Calculating min of " << filtered_args[0] << "...\n";

                return []<typename DataT>(DataT&& data) -> std::any
                {
                    using DecayedDataT = std::remove_cvref_t<DataT>;
                    
                    if constexpr (TinyDIP::is_complex_data_v<DecayedDataT>)
                    {
                        throw std::invalid_argument("Input data type (complex) does not support min (elements are not comparable).");
                        return std::any{};
                    }
                    else if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                    {
                        if constexpr (requires { TinyDIP::min(std::forward<DataT>(data)); })
                        {
                            return TinyDIP::min(std::forward<DataT>(data));
                        }
                        else
                        {
                            throw std::invalid_argument("Input image type does not support min.");
                            return std::any{};
                        }
                    }
                    else
                    {
                        if constexpr (requires { std::ranges::min(std::forward<DataT>(data)); })
                        {
                            return std::ranges::min(std::forward<DataT>(data));
                        }
                        else
                        {
                            throw std::invalid_argument("Input container type does not support min (elements are not comparable).");
                            return std::any{};
                        }
                    }
                };
            }
        );

        transform_handler(workspace, args, os);
    }

	//  multiply function implementation
    constexpr void multiply(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        auto transform_handler = make_meta_transform_handler<3, master_data_types>(
            "multiply [execution_policy] <input_data | $var> <output_var | $var> <scalar>", 
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                const double scalar_val = parse_arg<double>(filtered_args[2]);

                os << "Multiplying " << filtered_args[0] << " by " << scalar_val;
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [scalar_val, policy_str, &os]<typename DataT>(DataT&& data) -> std::any
                {
                    using DecayedDataT = std::remove_cvref_t<DataT>;

                    auto mult_op = [scalar_val](auto&& element)
                    {
                        using ValT = std::remove_cvref_t<decltype(element)>;
                        if constexpr (std::is_integral_v<ValT>)
                        {
                            return static_cast<ValT>(std::clamp(
                                static_cast<double>(element) * scalar_val,
                                static_cast<double>(std::numeric_limits<ValT>::lowest()),
                                static_cast<double>(std::numeric_limits<ValT>::max())
                            ));
                        }
                        else if constexpr (std::is_floating_point_v<ValT>)
                        {
                            return static_cast<ValT>(element * static_cast<ValT>(scalar_val));
                        }
                        else if constexpr (TinyDIP::is_complex_data_v<ValT>)
                        {
                            using S = TinyDIP::get_deep_scalar_t<ValT>;
                            if constexpr (requires { element * static_cast<S>(scalar_val); })
                            {
                                return static_cast<ValT>(element * static_cast<S>(scalar_val));
                            }
                            else
                            {
                                throw std::invalid_argument("Input element type does not support scalar multiplication.");
                                return ValT{};
                            }
                        }
                        else if constexpr (requires { std::forward<decltype(element)>(element) * scalar_val; })
                        {
                            // Provide support for natively overloaded structures like RGB
                            return std::forward<decltype(element)>(element) * scalar_val;
                        }
                        else
                        {
                            throw std::invalid_argument("Input element type does not support scalar multiplication.");
                            return ValT{};
                        }
                    };

                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            // Short-circuit evaluation: Place !is_bool_data_v and !is_complex_data_v on the left side of && to prevent
                            // SFINAE hard errors when instantiating unsupported native functions with boolean or complex channels!
                            if constexpr (!TinyDIP::is_bool_data_v<DecayedDataT> && !TinyDIP::is_complex_data_v<DecayedDataT> && requires { TinyDIP::multiplies(std::forward<DataT>(data), scalar_val); })
                            {
                                return TinyDIP::multiplies(std::forward<DataT>(data), scalar_val);
                            }
                            else if constexpr (!TinyDIP::is_bool_data_v<DecayedDataT> && !TinyDIP::is_complex_data_v<DecayedDataT> && requires { std::forward<DataT>(data) * scalar_val; })
                            {
                                return std::forward<DataT>(data) * scalar_val;
                            }
                            else if constexpr (requires { TinyDIP::pixelwise_transform(mult_op, std::forward<DataT>(data)); })
                            {
                                return TinyDIP::pixelwise_transform(mult_op, std::forward<DataT>(data));
                            }
                            else
                            {
                                throw std::invalid_argument("Input image type does not support scalar multiplication.");
                                return std::any{};
                            }
                        }
                        else if constexpr (std::ranges::input_range<DecayedDataT> &&
                            requires { TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(mult_op, std::forward<DataT>(data)); })
                        {
                            return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                mult_op,
                                std::forward<DataT>(data)
                            );
                        }
                        else if constexpr (std::is_invocable_v<decltype(mult_op), DataT>)
                        {
                            return mult_op(std::forward<DataT>(data));
                        }
                        else
                        {
                            throw std::invalid_argument("Input data type does not support scalar multiplication.");
                            return std::any{};
                        }
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            if constexpr (!TinyDIP::is_bool_data_v<DecayedDataT> && !TinyDIP::is_complex_data_v<DecayedDataT> && requires { TinyDIP::multiplies(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), scalar_val); })
                            {
                                return TinyDIP::multiplies(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), scalar_val);
                            }
                            else if constexpr (requires { TinyDIP::pixelwise_transform(std::forward<ExecPolicy>(exec_policy), mult_op, std::forward<DataT>(data)); })
                            {
                                return TinyDIP::pixelwise_transform(std::forward<ExecPolicy>(exec_policy), mult_op, std::forward<DataT>(data));
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
                        else if constexpr (std::ranges::input_range<DecayedDataT> &&
                            requires { TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(std::forward<ExecPolicy>(exec_policy), mult_op, std::forward<DataT>(data)); })
                        {
                            return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                std::forward<ExecPolicy>(exec_policy),
                                mult_op,
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

    //  ones template function implementation
    template <
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<double>&&> ImageSaverFun = MetaImageIO::Saver
    >
    constexpr void ones(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        create_image_with_initial_value(
            workspace, args, 1.0, "ones", os, std::forward<ImageSaverFun>(image_saver_fun)
        );
    }

    //  print template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void print(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
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

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_print))
        {
            // If dispatch_data_operation returns false, it must be a $ variable holding a scalar or unsupported type
            const std::string_view var_name = input_arg.substr(1);
            
            // Polymorphic lambda returning true if the complex custom scalar type matched
            auto try_print_complex_scalar = [&]<typename T>() -> bool
            {
                if (workspace.retrieve<T>(var_name))
                {
                    os << "Printing scalar value for " << input_arg << ":\n";
                    if constexpr (is_vector_v<T> || is_deque_v<T> || is_list_v<T> || is_std_array_v<T>)
                    {
                        os << "container value = {";
                        bool first = true;
                        const auto* container_ptr = workspace.retrieve<T>(var_name);
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
                        os << *workspace.retrieve<T>(var_name) << "\nDone.\n";
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
                    if (workspace.retrieve<T>(var_name))
                    {
                        os << "Printing scalar value for " << input_arg << ":\n";
                        if constexpr (sizeof(T) == 1 && std::is_integral_v<T>) // Safely print 8-bit integer types as numbers, not unprintable chars
                        {
                            os << +(*workspace.retrieve<T>(var_name)) << "\nDone.\n";
                        }
                        else
                        {
                            os << *workspace.retrieve<T>(var_name) << "\nDone.\n";
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

    //  RandomGenerator template struct implementation
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

    //  rand_generator template function implementation
    template <
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<double>&&> ImageSaverFun = MetaImageIO::Saver
    >
    constexpr void rand_generator(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        auto dispatch_generation = [&]
        (std::uniform_random_bit_generator auto&& urbg, const std::string_view& out_path, std::span<const std::size_t> sz)
        {
            std::uniform_real_distribution<double> dist{};
            using UrbgType = std::remove_cvref_t<decltype(urbg)>;
            using DistType = decltype(dist);

            RandomGenerator<UrbgType, DistType> gen{urbg, dist};

            //  Calling the dynamic range-based generate overload directly from TinyDIP.
            auto output_img = TinyDIP::generate(gen, sz);

            // Dynamically save image via the injected saver abstraction
            image_saver_fun(out_path, workspace, std::move(output_img));
            os << "Saved to " << out_path << "\n";
        };

        std::map<std::string_view, std::function<void(const std::string_view&, std::span<const std::size_t>)>> urbg_mapping = {
            {"knuth_b",       [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::knuth_b{std::random_device{}()}, out_path, sz); }
            },
            {"minstd_rand",   [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::minstd_rand{std::random_device{}()}, out_path, sz); }
            },
            {"minstd_rand0",  [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::minstd_rand0{std::random_device{}()}, out_path, sz); }
            },
            {"mt19937",       [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::mt19937{std::random_device{}()}, out_path, sz); }
            },
            {"mt19937_64",    [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::mt19937_64{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux24",      [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::ranlux24{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux24_base", [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::ranlux24_base{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux48",      [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::ranlux48{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux48_base", [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
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

    template <
        std::invocable<const std::string_view, Workspace&> ImageLoaderFun = MetaImageIO::Loader,
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::RGB>&&> ImageSaverFun = MetaImageIO::Saver
    >
    void read(
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
            image_saver_fun(output_arg, workspace, std::forward<ImageType>(input_img));
        };

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_read))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
            return;
        }

        os << "Done.\n";
	}

    void remove(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        if (std::ranges::empty(args))
        {
            os << "Usage: remove <$var1> [$var2] ... OR remove all\n";
            return;
        }

        if (std::ranges::size(args) == 1 && std::string_view(args[0]) == "all")
        {
            workspace.clear();
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
            if (workspace.remove(var_name))
            {
                os << "Removed memory variable $" << var_name << ".\n";
            }
            else
            {
                os << "Warning: Memory variable $" << var_name << " not found.\n";
            }
        }
    }

	//  rename function implementation
    void rename(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
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

        if (workspace.rename(old_name, new_name))
        {
            os << "Renamed variable $" << old_name << " to $" << new_name << ".\n";
        }
        else
        {
            os << "Error: Memory variable $" << old_name << " not found.\n";
        }
    }

    //  rotate function implementation
    constexpr void rotate(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        auto transform_handler = make_meta_transform_handler<3>(
            "rotate [execution_policy] <input_img | $var> <output_img | $var> <angle>", 
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                const double angle = parse_arg<double>(filtered_args[2]);

                os << "Rotating " << filtered_args[0] << " by " << angle;
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [angle, policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                {
                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (requires { TinyDIP::rotate_detail_shear_transformation(std::forward<ImageType>(img), angle); })
                        {
                            return TinyDIP::rotate_detail_shear_transformation(std::forward<ImageType>(img), angle);
                        }
                        else
                        {
                            throw std::invalid_argument("Input image type does not support rotate_detail_shear_transformation.");
                            return std::any{};
                        }
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (requires { TinyDIP::rotate_detail_shear_transformation(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), angle); })
                        {
                            return TinyDIP::rotate_detail_shear_transformation(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), angle);
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

        const std::filesystem::path dir_path = std::string(args[0]);
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

    //  sift_generate_octave template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void sift_generate_octave(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: sift_generate_octave <input_img | $var> <output_var | $var> [levels=5] [initial_sigma=1.6] [k=1.414]\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];

        if (!output_arg.starts_with('$'))
        {
            os << "Error: Output must be a memory variable starting with '$'.\n";
            return;
        }

        const std::size_t levels = (std::ranges::size(args) > 2) ? parse_arg<std::size_t>(args[2]) : 5;
        const double initial_sigma = (std::ranges::size(args) > 3) ? parse_arg<double>(args[3]) : 1.6;
        const double k = (std::ranges::size(args) > 4) ? parse_arg<double>(args[4]) : std::numbers::sqrt2_v<double>;

        os << "Generating SIFT octave from " << input_arg << " with " << levels << " levels...\n";

        auto process_octave = [&]<typename ImageType>(ImageType&& input_img)
        {
            using DecayedImageType = std::remove_cvref_t<ImageType>;

            if constexpr (TinyDIP::is_bool_data_v<DecayedImageType> || TinyDIP::is_complex_data_v<DecayedImageType>)
            {
                os << "Error: Input image type [" << get_type_name<DecayedImageType>() << "] does not support SIFT octave generation.\n";
                return;
            }
            else
            {
                auto process_octave_impl = [&]<typename T>(T&& double_img)
                {
                    using ImplDecayedT = std::remove_cvref_t<T>;

                    if constexpr (requires { TinyDIP::SIFT_impl::generate_octave(std::forward<T>(double_img), levels, initial_sigma, k); })
                    {
                        auto result = TinyDIP::SIFT_impl::generate_octave(std::forward<T>(double_img), levels, initial_sigma, k);
                        workspace.store(output_arg.substr(1), std::move(result));
                        os << "Saved SIFT octave to " << output_arg << ".\n";
                    }
                    else
                    {
                        os << "Error: Elevated input image type [" << get_type_name<ImplDecayedT>() << "] does not support generate_octave.\n";
                    }
                };

                using RawScalarT = TinyDIP::get_deep_scalar_t<DecayedImageType>;

                if constexpr (std::same_as<RawScalarT, double>)
                {
                    process_octave_impl(std::forward<ImageType>(input_img));
                }
                else if constexpr (requires { TinyDIP::im2double(std::forward<ImageType>(input_img)); })
                {
                    process_octave_impl(TinyDIP::im2double(std::forward<ImageType>(input_img)));
                }
                else
                {
                    os << "Error: Input image type [" << get_type_name<DecayedImageType>() << "] cannot be converted to double precision for SIFT octave generation.\n";
                }
            }
        };

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_octave))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }

    //  subimage function implementation
    constexpr void subimage(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        auto transform_handler = make_meta_transform_handler<6>(
            "subimage [execution_policy] <input_img | $var> <output_img | $var> <x_offset> <y_offset> <width> <height>", 
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                const std::size_t x_offset = parse_arg<std::size_t>(filtered_args[2]);
                const std::size_t y_offset = parse_arg<std::size_t>(filtered_args[3]);
                const std::size_t sub_width = parse_arg<std::size_t>(filtered_args[4]);
                const std::size_t sub_height = parse_arg<std::size_t>(filtered_args[5]);

                os << "Extracting subimage from " << filtered_args[0] << " at (" << x_offset << ", " << y_offset 
                   << ") with size " << sub_width << "x" << sub_height;
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [x_offset, y_offset, sub_width, sub_height, policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                {
                    using DecayedT = std::remove_cvref_t<ImageType>;

                    // Mathematically guarantee the type is a 2D matrix/image
                    if constexpr (requires { img.getWidth(); img.getHeight(); img.at(0, 0); })
                    {
                        if (x_offset + sub_width > img.getWidth() || y_offset + sub_height > img.getHeight())
                        {
                            throw std::out_of_range("Subimage bounds exceed original image dimensions.");
                            return std::any{};
                        }

                        auto exec_default = [&]() -> std::any
                        {
                            DecayedT out_img(sub_width, sub_height);
                            for (std::size_t y = 0; y < sub_height; ++y)
                            {
                                for (std::size_t x = 0; x < sub_width; ++x)
                                {
                                    out_img.at(x, y) = img.at(x + x_offset, y + y_offset);
                                }
                            }
                            return out_img;
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            DecayedT out_img(sub_width, sub_height);
                            auto indices = std::views::iota(std::size_t{0}, sub_width * sub_height);
                            
                            std::for_each(
                                std::forward<ExecPolicy>(exec_policy),
                                std::ranges::begin(indices),
                                std::ranges::end(indices),
                                [&](const std::size_t idx)
                                {
                                    const std::size_t y = idx / sub_width;
                                    const std::size_t x = idx % sub_width;
                                    out_img.at(x, y) = img.at(x + x_offset, y + y_offset);
                                }
                            );
                            return out_img;
                        };

                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    }
                    else
                    {
                        throw std::invalid_argument("Input type does not support subimage extraction.");
                        return std::any{};
                    }
                };
            }
        );

        transform_handler(workspace, args, os);
    }

    //  sum function implementation
    constexpr void sum(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        auto transform_handler = make_meta_scalar_handler<1>(
            "sum [execution_policy] <input_data | $var> [output_var | $var]", 
            "sum", "Sum", 
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                os << "Calculating sum of " << filtered_args[0];
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [policy_str, &os]<typename DataT>(DataT&& img) -> std::any
                {
                    // Helper to safely execute sum on the potentially casted image or generic container
                    auto process_sum_impl = [&]<typename T>(T&& actual_data) -> std::any
                    {
                        using DecayedT = std::remove_cvref_t<T>;
                        
                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (TinyDIP::is_Image<DecayedT>::value)
                            {
                                if constexpr (requires { TinyDIP::sum(std::forward<T>(actual_data)); })
                                {
                                    return TinyDIP::sum(std::forward<T>(actual_data));
                                }
                                else
                                {
                                    throw std::invalid_argument("Input image type does not support sum.");
                                    return std::any{};
                                }
                            }
                            else
                            {
                                using ValT = std::ranges::range_value_t<DecayedT>;
                                return std::accumulate(std::ranges::begin(actual_data), std::ranges::end(actual_data), ValT{});
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (TinyDIP::is_Image<DecayedT>::value)
                            {
                                if constexpr (requires { TinyDIP::sum(std::forward<ExecPolicy>(exec_policy), std::forward<T>(actual_data)); })
                                {
                                    return TinyDIP::sum(std::forward<ExecPolicy>(exec_policy), std::forward<T>(actual_data));
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

                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
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
        );

        transform_handler(workspace, args, os);
    }

    //  to_complex function implementation
    constexpr void to_complex(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        auto transform_handler = make_meta_transform_handler<2, master_data_types>(
                "to_complex [execution_policy] <input_data | $var> <output_var | $var>", 
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    if (!std::ranges::empty(policy_str))
                    {
                        os << "Converting " << filtered_args[0] << " to complex (Policy: " << policy_str << ")...\n";
                    }
                    else
                    {
                        os << "Converting " << filtered_args[0] << " to complex...\n";
                    }

                    return [policy_str, &os]<typename DataT>(DataT&& data) -> std::any
                    {
                        using DecayedDataT = std::remove_cvref_t<DataT>;

                        if constexpr (TinyDIP::is_bool_data_v<DecayedDataT>)
                        {
                            throw std::invalid_argument("Input data type (bool) does not support to_complex conversion.");
                            return std::any{};
                        }
                        else if constexpr (TinyDIP::is_complex_data_v<DecayedDataT>)
                        {
                            // The complex value of an already complex type is simply the exact same value!
                            // Returning the forwarded data natively bypasses the C++ standard library's 
                            // ambiguous template instantiations for complex types.
                            return DecayedDataT(std::forward<DataT>(data));
                        }
                        else
                        {
                            auto exec_default = [&]() -> std::any
                            {
                                if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                                {
                                    if constexpr (requires { TinyDIP::to_complex(std::forward<DataT>(data)); })
                                    {
                                        return TinyDIP::to_complex(std::forward<DataT>(data));
                                    }
                                    else
                                    {
                                        throw std::invalid_argument("Input image type does not support to_complex conversion.");
                                        return std::any{};
                                    }
                                }
                                else if constexpr (std::ranges::input_range<DecayedDataT>)
                                {
                                    if constexpr (requires { TinyDIP::to_complex(*std::ranges::begin(data)); })
                                    {
                                        return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                            [](auto&& element) 
                                            { 
                                                return TinyDIP::to_complex(std::forward<decltype(element)>(element));
                                            },
                                            std::forward<DataT>(data)
                                        );
                                    }
                                    else
                                    {
                                        throw std::invalid_argument("Input container type does not support to_complex conversion.");
                                        return std::any{};
                                    }
                                }
                                else
                                {
                                    throw std::invalid_argument("Input data type does not support to_complex conversion.");
                                    return std::any{};
                                }
                            };

                            auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                                requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                            {
                                if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                                {
                                    if constexpr (requires { TinyDIP::to_complex(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                                    {
                                        return TinyDIP::to_complex(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
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
                                    if constexpr (requires { TinyDIP::to_complex(*std::ranges::begin(data)); })
                                    {
                                        return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                            std::forward<ExecPolicy>(exec_policy),
                                            [](auto&& element) 
                                            { 
                                                return TinyDIP::to_complex(std::forward<decltype(element)>(element));
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

                            return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                        }
                    };
                }
            );

        transform_handler(workspace, args, os);
    }

	//  vars function implementation
    void vars(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        (void)args;
        os << "Current Workspace Variables:\n";
        workspace.list_variables(os);
    }

	//  write template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader,
        typename ImageSaverFun = MetaImageIO::Saver
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&> &&
              std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<double>&&>)
    constexpr void write(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{},
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
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
            image_saver_fun(output_arg, workspace, std::forward<ImageType>(input_img));
        };

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_write))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
            return;
        }

        os << "Done.\n";
    }

    //  zeros template function implementation
    template <
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<double>&&> ImageSaverFun = MetaImageIO::Saver
    >
    constexpr void zeros(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        create_image_with_initial_value(
            workspace, args, 0.0, "zeros", os, std::forward<ImageSaverFun>(image_saver_fun)
        );
    }
}

//  run_legacy_tests function implementation
//  Legacy test function wrapper
void run_legacy_tests(
    Workspace& workspace,
    std::span<const std::string_view> args,
    std::ostream& os = std::cout
)
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
template <std::invocable<Workspace&, std::span<const std::string_view>, std::ostream&>... Funs>
constexpr CommandRegistry command_registration(CommandBundle<Funs>&&... bundles)
{
    CommandRegistry registry;

    //  Unpack and register all provided command bundles automatically using a C++17 fold expression
    (registry.register_command(bundles.name, bundles.description, bundles.schema, std::forward<Funs>(bundles.handler)), ...);

    //  Internal / Anonymous Handlers can still be registered statically here
    registry.register_command("test", "Run internal integration tests.", 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            run_legacy_tests(workspace, args, os);
        }
    );

    registry.register_command("batch_add_zeros", "Add leading zeros to filenames in a directory.",
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            (void)workspace;
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
void run_interactive_mode(Workspace& workspace, const CommandRegistry& registry, std::ostream& os = std::cout)
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

            registry.execute(workspace, command_name, args_sv, os);

            prev_pipe_var = next_pipe_var;
        }
    }
}


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

//  Main Entry Point
int main(int argc, char* argv[])
{
    // Configure the shared state memory workspace
    auto workspace = std::make_shared<Workspace>();
    
    // Register commands directly with context-injected instances using generic variadic bundles
    CommandRegistry registry = command_registration(
        CommandBundle{"abs", "Calculate the absolute value of an image or container.", TransformerSchema, 
            handlers::abs
        },
        CommandBundle{"append_element", "Append an element to the back of a container using emplace_back.", CombinerSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::append_element(workspace, args, os);
            }
        },
        CommandBundle{"bicubic_resize", "Resize an image using Bicubic interpolation.", TransformerSchema, 
            handlers::bicubic_resize
        },
        CommandBundle{"constructRGB", "Merge separate R, G, and B planes into an RGB image.", IndependentSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::construct_rgb(workspace, args, os);
            }
        },        
        CommandBundle{"create_container", "Create a std::vector container initialized with the provided prototype element.", TransformerSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::create_container(workspace, args, os);
            }
        },
        CommandBundle{"dct2", "Calculate Discrete Cosine Transformation for an image.", TransformerSchema, 
            handlers::dct2
        },
        CommandBundle{"gaussian_figure_2d", "Generate a 2D Gaussian figure image.", GeneratorSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::gaussian_figure_2d(workspace, args, os);
            }
        },
        CommandBundle{"get_element", "Extract an element from a container (e.g., an octave vector) by index.", TransformerSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::get_element(workspace, args, os);
            }
        },
        CommandBundle{"getBplane", "Extract the Blue plane (channel 2) from a multi-channel image.", TransformerSchema,
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::getPlane(workspace, args, os, 2);
            }
        },
        CommandBundle{"getGplane", "Extract the Green plane (channel 1) from a multi-channel image.", TransformerSchema,
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::getPlane(workspace, args, os, 1);
            }
        },
        CommandBundle{"getRplane", "Extract the Red plane (channel 0) from a multi-channel image.", TransformerSchema,
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::getPlane(workspace, args, os, 0);
            }
        },
        CommandBundle{"get_sift_potential_keypoint", "Extract SIFT potential keypoints from an image.", TransformerSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::get_sift_potential_keypoint(workspace, args, os);
            }
        },
        CommandBundle{"grid", "Generate a grid image.", GeneratorSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::grid_generator(workspace, args, os);
            }
        },
        CommandBundle{"hsv2rgb", "Convert an HSV image or container to RGB color space.", TransformerSchema,
            handlers::hsv2rgb
        },
        CommandBundle{"idct2", "Calculate Inverse Discrete Cosine Transformation for an image.", TransformerSchema,
            handlers::idct2 
        },
        CommandBundle{"im2double", "Convert an image or container to double precision floating-point.", TransformerSchema,
            handlers::im2double
        },
        CommandBundle{"im2uint8", "Convert an image or container to 8-bit unsigned integers.", TransformerSchema,
            handlers::im2uint8
        },
        CommandBundle{"info", "Display basic information about an image.", TerminatorSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::info(workspace, args, os);
            }
        },
        CommandBundle{"lanczos_resample", "Resize an image using Lanczos resampling.", TransformerSchema,
            handlers::lanczos_resample
        },
        CommandBundle{"load_workspace", "Load memory variables from a directory bundle.", IndependentSchema, 
            handlers::load_workspace
        },
        CommandBundle{"max", "Calculate the maximum value of an image or container.", TransformerSchema,
            handlers::max
        },
        CommandBundle{"min", "Calculate the minimum value of an image or container.", TransformerSchema,
            handlers::min
        },
        CommandBundle{ "multiply", "Multiply an image or container by a scalar.", TransformerSchema,
            handlers::multiply
        },
        CommandBundle{"ones", "Generate an image filled with ones.", GeneratorSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::ones(workspace, args, os);
            }
        },
        CommandBundle{"print", "Print the contents of a memory variable.", TerminatorSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::print(workspace, args, os);
            }
        },
        CommandBundle{"rand", "Generate random multi-dimensional image with specified URBG.", GeneratorSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::rand_generator(workspace, args, os);
            }
        },
        CommandBundle{"read", "Read an image from disk into a memory variable.", GeneratorSchema,
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::read(workspace, args, os);
            }
        },
        CommandBundle{"remove", "Remove memory variables from the workspace (or 'all' to clear).", IndependentSchema, handlers::remove },
        CommandBundle{"rename", "Rename a memory variable in the workspace.", IndependentSchema, handlers::rename },
        CommandBundle{"rotate", "Rotate an image using shear transformations.", TransformerSchema,
            handlers::rotate
        },
        CommandBundle{"save_workspace", "Save all memory variables to a directory bundle.", IndependentSchema,
            handlers::save_workspace
        },
        CommandBundle{"sift_generate_octave", "Generate a SIFT octave (Difference of Gaussian images).", TransformerSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::sift_generate_octave(workspace, args, os);
            }
        },
        CommandBundle{"subimage", "Extract a sub-region from an image.", TransformerSchema,
            handlers::subimage
        },
        CommandBundle{"sum", "Calculate the sum of all elements in an image or container.", TransformerSchema,
            handlers::sum
        },
        CommandBundle{"to_complex", "Convert an image or container to a complex number format.", TransformerSchema,
            handlers::to_complex
        },
        CommandBundle{"vars", "List all currently allocated memory variables.", IndependentSchema, handlers::vars },
        CommandBundle{"write", "Write a memory variable out to a disk file.", TerminatorSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::write(workspace, args, os);
            }
        },
        CommandBundle{"zeros", "Generate an image filled with zeros.", GeneratorSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::zeros(workspace, args, os);
            }
        }
    );

    // Register the help command dynamically to ensure it has access to the final mapped registry
    registry.register_command("help", "List all available commands.", IndependentSchema,
        [&registry](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::help(registry, workspace, args, os);
        }
    );

    if (argc < 2)
    {
        run_interactive_mode(*workspace, registry);
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

    registry.execute(*workspace, command, args);

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
