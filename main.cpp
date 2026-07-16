/* Developed by Jimmy Hu */
/* Refactored for CLI Application capability */

//  compile command:
//  clang++ -std=c++20 -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp  main.cpp -L /usr/local/Cellar/llvm/10.0.0_3/lib/ -lm -O3 -o main -v
//  https://stackoverflow.com/a/61821729/6667035
//  clear && rm -rf ./main && g++-11 -std=c++20 -O4 -ffast-math -funsafe-math-optimizations -std=c++20 -fpermissive -H --verbose -Wall main.cpp -o main 


//#define USE_BOOST_ITERATOR
//#define USE_BOOST_SERIALIZATION

#include "main.h"
#include "dynamic_loader.h"


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

namespace handlers
{
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
//  Interactive REPL loop implementation with dynamic Pipeline '|' AST Generation
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

        std::vector<QueuedCommand> execution_pipeline;
        std::string prev_pipe_var = "";
        bool parsing_failed = false;

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
                parsing_failed = true;
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

            execution_pipeline.push_back({command_name, std::move(args_str)});
            prev_pipe_var = next_pipe_var;
        }

        if (parsing_failed)
        {
            continue;
        }

        PeepholeOptimizer<decltype(execution_pipeline)>::optimize(execution_pipeline, registry, os);

        for (const auto& cmd : execution_pipeline)
        {
            std::vector<std::string_view> args_sv;
            args_sv.reserve(std::ranges::size(cmd.args));
            for (const auto& arg : cmd.args)
            {
                args_sv.emplace_back(arg);
            }

            registry.execute(workspace, cmd.name, args_sv, os);
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

//  load_plugins template function implementation
//  Helper function to seamlessly load and register dynamic modules natively
//  Upgraded to natively accept any container type (std::vector, std::deque, std::list)
template <typename PluginContainerT>
requires(std::same_as<std::ranges::range_value_t<PluginContainerT>, DynamicLibrary>)
inline void load_plugins(
    CommandRegistry& registry,
    PluginContainerT& active_plugins,
    const std::filesystem::path& plugin_dir = "./plugins",
    std::ostream& os = std::cout)
{
    if (std::filesystem::exists(plugin_dir) && std::filesystem::is_directory(plugin_dir))
    {
        os << "Scanning for dynamic modules in " << plugin_dir.string() << "...\n";
        
        for (const auto& entry : std::filesystem::directory_iterator(plugin_dir))
        {
            const std::string ext = entry.path().extension().string();
            if (ext == ".dll" || ext == ".so" || ext == ".dylib")
            {
                try
                {
                    DynamicLibrary lib(entry.path());
                    
                    // Locate the exact C-ABI export symbol
                    using RegisterFunc = void(*)(CommandRegistry&);
                    auto register_fn = lib.get_function<RegisterFunc>("register_plugin_commands");
                    
                    // Inject the central registry into the DLL
                    register_fn(registry);
                    
                    // Securely hold the library handle in RAM so it doesn't destruct while executing
                    // Utilizing emplace_back to dynamically support vector, deque, and list natively
                    if constexpr (requires { active_plugins.emplace_back(std::move(lib)); })
                    {
                        active_plugins.emplace_back(std::move(lib));
                    }
                    else if constexpr (requires { active_plugins.push_back(std::move(lib)); })
                    {
                        active_plugins.push_back(std::move(lib));
                    }
                    else
                    {
                        active_plugins.insert(std::ranges::end(active_plugins), std::move(lib));
                    }
                    
                    os << "  Loaded module: " << entry.path().filename().string() << "\n";
                }
                catch (const std::exception& e)
                {
                    os << "  Warning: Module skipped (" << e.what() << ")\n";
                }
            }
        }
    }
    else
    {
        os << "No " << plugin_dir.string() << " directory found. Running with core commands only.\n";
    }
}

//  Main Entry Point
int main(int argc, char* argv[])
{
    // Configure the shared state memory workspace
    auto workspace = std::make_shared<Workspace>();

    // Define the persistent active plugins container at the highest scope to govern RAII destructors safely
    std::vector<DynamicLibrary> active_plugins;
    
    // Register commands directly with context-injected instances using generic variadic bundles
    CommandRegistry registry = command_registration(
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
        CommandBundle{"copy", "Copy a memory variable in the workspace to a new variable.", IndependentSchema,
            handlers::copy
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
        CommandBundle{"dct3", "Calculate three-dimensional Discrete Cosine Transformation for an image.", TransformerSchema, 
            handlers::dct3
        },
        CommandBundle{"erase_element", "Erase an element from a container by index.", TransformerSchema, 
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::erase_element(workspace, args, os);
            }
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
        CommandBundle{ "multiply", "Multiply an image or container by a scalar.", TransformerSchema,
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::multiply(workspace, args, os);
            }
        },
        CommandBundle{"normalize", "Normalize an image or container.", TransformerSchema,
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::normalize(workspace, args, os);
            }
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
            [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
            {
                handlers::save_workspace(workspace, args, os);
            }
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

    // Register transform_container strictly after registry initialization to prevent circular reference cycles natively!
    registry.register_command("transform_container", "Apply a CLI command dynamically to each element of a container sequentially or parallelly.", IndependentSchema, 
        [&registry](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::transform_container(workspace, args, registry, os);
        }
    );

    // Register the dynamic load_plugins command natively to allow on-the-fly module injections!
    registry.register_command("load_plugins", "Dynamically load plugin modules from a specified directory.", IndependentSchema, 
        [&registry, &active_plugins](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            (void)workspace; // Securely suppress unused warning gracefully
            std::filesystem::path dir = "./plugins";
            
            if (!std::ranges::empty(args))
            {
                dir = std::string(sanitize_string_view(args[0]));
            }
            
            load_plugins(registry, active_plugins, dir, os);
        }
    );

    //  Dynamically Load Plugins on boot automatically
    load_plugins(registry, active_plugins);

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
