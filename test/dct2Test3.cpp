/* Developed by Jimmy Hu */

#include <chrono>
#include <execution>
#include <map>
#include <omp.h>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

//  get_offset template function implementation
template<class ExPo, class ElementT, class DistanceFunction>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr static auto get_offset( ExPo execution_policy, 
                                  const TinyDIP::Image<ElementT>& input,
                                  const std::vector<TinyDIP::Image<ElementT>>& dictionary_x,
                                  const std::vector<TinyDIP::Image<ElementT>>& dictionary_y,
                                  const ElementT sigma, const ElementT threshold,
                                  const DistanceFunction distance_function,
                                  bool display_sum_of_weights = false,
                                  std::ostream& os = std::cout
                                ) noexcept
{
    auto output = TinyDIP::zeros<ElementT>(input.getWidth(), input.getHeight());
    auto weights = TinyDIP::recursive_transform<1>(
        execution_policy,
        [&](auto&& element)
        { 
            return TinyDIP::normalDistribution1D(std::invoke(distance_function, input, element), sigma);
        }, dictionary_x);
    auto sum_of_weights = TinyDIP::recursive_reduce(weights, ElementT{});
    if (display_sum_of_weights)
    {
        std::cout << "sum_of_weights: " << std::format("{}", sum_of_weights) << '\n';
    }
    if (sum_of_weights < threshold)
    {
        return output;
    }
    //std::cout << "#weights: " << std::to_string(weights.size()) << "\t#dictionary_y: " << std::to_string(dictionary_y.size()) << '\n';
    if constexpr(true)    //    Use OpenMP
    {
        std::vector<TinyDIP::Image<ElementT>> outputs;
        outputs.resize(dictionary_y.size());
        #pragma omp parallel for
        //  C3016 'i': index variable in OpenMP 'for' statement must have signed integral type
        for (std::size_t i = 0; i < dictionary_y.size(); ++i)
        {
            outputs[i] = dictionary_y[i] * weights[i];
        }
        output =  TinyDIP::recursive_reduce(outputs, output);
    }
    else
    {
        auto outputs = TinyDIP::recursive_transform<1>(
        [&](auto&& input1, auto&& input2)
        {
            return input1 * input2;
        }, dictionary_y, weights);
        output =  TinyDIP::recursive_reduce(outputs, output);
    }
    auto image_for_divides = TinyDIP::Image<ElementT>(output.getWidth(), output.getHeight());
    image_for_divides.setAllValue(sum_of_weights);
    output = TinyDIP::divides(output, image_for_divides);
    return output;
}

//    each_image Template Function Implementation
template<class ExPo, class ElementT1, class ElementT2>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr auto each_image( ExPo execution_policy, 
                 const TinyDIP::Image<ElementT2>& input_img,
                 std::vector<TinyDIP::Image<ElementT1>>& dictionary_x,
                 std::vector<TinyDIP::Image<ElementT1>>& dictionary_y,
                 const std::size_t N1 = 8, const std::size_t N2 = 8, const ElementT1 sigma = 0.1) noexcept
{
    auto mod_x = std::fmod(static_cast<double>(input_img.getWidth()), static_cast<double>(N1));
    auto mod_y = std::fmod(static_cast<double>(input_img.getHeight()), static_cast<double>(N2));

    auto input_hsv = TinyDIP::subimage(
                        TinyDIP::rgb2hsv(input_img),
                        input_img.getWidth() - mod_x, input_img.getHeight() - mod_y,
                        static_cast<double>(input_img.getWidth()) / 2.0, static_cast<double>(input_img.getHeight()) / 2.0
                        );
    auto h_plane = TinyDIP::getHplane(input_hsv);
    auto s_plane = TinyDIP::getSplane(input_hsv);
    auto image_255 = TinyDIP::Image<double>(input_img.getWidth(), input_img.getHeight());
    image_255.setAllValue(255);
    auto v_plane = TinyDIP::divides(TinyDIP::getVplane(input_hsv), image_255);

    std::cout << "Call dct2 function..." << '\n';
    auto input_dct_blocks = TinyDIP::recursive_transform<2>(
        execution_policy,
        [](auto&& element) { return TinyDIP::dct2(element); },
        TinyDIP::split(v_plane, v_plane.getWidth() / N1, v_plane.getHeight() / N2)
        );

    auto output_dct_blocks = input_dct_blocks;
    auto y_size = input_dct_blocks.size();
    auto x_size = input_dct_blocks[0].size();
    #pragma omp parallel for collapse(2)
    for (std::size_t y = 0; y < y_size; ++y)
    {
        for (std::size_t x = 0; x < x_size; ++x)
        {
            auto function = [&](auto&& element) {
                return TinyDIP::plus(element,
                    get_offset(
                        execution_policy,
                        element,
                        dictionary_x,
                        dictionary_y,
                        sigma,
                        std::pow(10, -30),
                        [&](auto&& input1, auto&& input2) { return TinyDIP::euclidean_distance(input1, input2); }
                    )); };
            output_dct_blocks[y][x] = std::invoke(function, input_dct_blocks[y][x]);
        }
        std::cout << "y = " << y << " / " << input_dct_blocks.size() << " block done.\n";
    }
    
    auto output_img = TinyDIP::hsv2rgb(TinyDIP::constructHSV(
        h_plane,
        s_plane,
        TinyDIP::pixelwise_multiplies(
            TinyDIP::concat(
                TinyDIP::recursive_transform<2>(
                    execution_policy,
                    [](auto&& element) { return TinyDIP::idct2(element); },
                    output_dct_blocks)),
            image_255)
    ));
    return output_img;
}

//    load_dictionary Template Function Implementation
template<TinyDIP::arithmetic ElementT = double>
constexpr auto load_dictionary( const std::string_view dictionary_path = "Dictionary",
                                const std::size_t dic_start_index = 80,
                                const std::size_t dic_end_index = 99,
                                const std::size_t N1 = 8,
                                const std::size_t N2 = 8)
{
    auto dictionary_path_temp = dictionary_path;
    //  https://stackoverflow.com/a/63622202/6667035
    if (!std::filesystem::is_directory(dictionary_path_temp))
    {
        dictionary_path_temp = "../Dictionary";
    }
    if (!std::filesystem::is_directory(dictionary_path_temp))
    {
        dictionary_path_temp = "../../Dictionary";
    }
    if (!std::filesystem::is_directory(dictionary_path_temp))
    {
        dictionary_path_temp = "../../../Dictionary";
    }
    //***Load dictionary***
    std::vector<TinyDIP::Image<ElementT>> x, y;
    for (std::size_t i = dic_start_index; i <= dic_end_index; ++i)
    {
        std::string fullpath = std::string(dictionary_path_temp) + "/" + std::to_string(i);
        std::cout << "Dictionary path: " << fullpath << '\n';
        auto input_dbmp = TinyDIP::double_image::read(fullpath.c_str(), false);
        auto dct_block_x = TinyDIP::split(input_dbmp, input_dbmp.getWidth() / N1, input_dbmp.getHeight() / N2);
        x.reserve(dct_block_x.size() * dct_block_x.at(0).size());
        TinyDIP::recursive_for_each<2>(
            std::execution::seq,
            [&](auto&& element) 
            {
                x.emplace_back(element);
            },
            dct_block_x);

        std::string fullpath_gt = std::string(dictionary_path_temp) + "/GT";
        auto input_dbmp_gt = TinyDIP::double_image::read(fullpath_gt.c_str(), false);
        auto dct_block_y = TinyDIP::split(input_dbmp_gt, input_dbmp_gt.getWidth() / N1, input_dbmp_gt.getHeight() / N2);
        y.reserve(dct_block_y.size() * dct_block_y.at(0).size());
        TinyDIP::recursive_for_each<2>(
            std::execution::seq,
            [&](auto&& element)
            {
                y.emplace_back(element);
            },
            dct_block_y);
    }
    auto xy_diff = TinyDIP::recursive_transform([&](auto&& element1, auto&& element2) { return TinyDIP::subtract(element2, element1); }, x, y);
    std::cout << "x count: " << x.size() << "\txy_diff count: " << xy_diff.size() << '\n';
    std::tuple<std::vector<TinyDIP::Image<ElementT>>, std::vector<TinyDIP::Image<ElementT>>> output{x, xy_diff};
    return output;
}

//    dct2Test3 Template Function Implementation
void dct2Test3( const std::string& input_folder, const std::string& output_folder,
                const std::string& dictionary_path,
                const std::size_t start_index = 1, const std::size_t end_index = 1,
                const std::size_t dic_start_index = 80, const std::size_t dic_end_index = 99,
                const std::size_t N1 = 8, const std::size_t N2 = 8, const double sigma = 0.1) noexcept
{
    std::cout << "dct2Test3 program..." << '\n';
    std::cout << "sigma = " << std::to_string(sigma) << '\n';
    
    auto dictionary = load_dictionary(dictionary_path, dic_start_index, dic_end_index, N1, N2);
    
    for (std::size_t i = start_index; i <= end_index; ++i)
    {
        std::string input_path = input_folder + "/" + std::to_string(i);
        std::cout << "input_path: " << input_path << '\n';
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);
        auto output_img = each_image(std::execution::seq, input_img, std::get<0>(dictionary), std::get<1>(dictionary), N1, N2, sigma);
        std::string output_path = output_folder + "/" + std::to_string(i);
        std::cout << "Save output to " << output_path << '\n';
        TinyDIP::bmp_write(output_path.c_str(), output_img);
    }
    return;
}

//  remove_extension Function Implementation
//  Copy from: https://stackoverflow.com/a/6417908/6667035
std::string remove_extension(const std::string& filename)
{
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

int main(int argc, char* argv[])
{
    auto start = std::chrono::system_clock::now();
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';
    if(argc == 2)
    {
        auto input_path = std::string(argv[1]);
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);
        auto dictionary = load_dictionary();
        auto output_img = each_image(std::execution::seq, input_img, std::get<0>(dictionary), std::get<1>(dictionary));
        auto output_path = std::string(argv[2]);
        std::cout << "Save output to " << output_path << '\n';
        TinyDIP::bmp_write(output_path.c_str(), output_img);
    }
    if (argc == 6)
    {
        //    example: ./build/dct2Test3 
        auto arg1 = std::string(argv[1]);
        auto arg2 = std::string(argv[2]);
        auto arg3 = std::string(argv[3]);
        auto arg4 = std::string(argv[4]);
        std::stringstream start_index_ss(arg4);
        std::size_t start_index;
        start_index_ss >> start_index;
        auto arg5 = std::string(argv[5]);
        std::stringstream end_index_ss(arg5);
        std::size_t end_index;
        end_index_ss >> end_index;
        const double sigma = 0.1;
        dct2Test3(arg1, arg2, arg3, start_index, end_index,
         static_cast<std::size_t>(80), static_cast<std::size_t>(100), static_cast<std::size_t>(8), static_cast<std::size_t>(8), sigma);
    }
    else if (argc == 7)
    {
        auto arg1 = std::string(argv[1]);
        auto arg2 = std::string(argv[2]);
        auto arg3 = std::string(argv[3]);
        auto arg4 = std::string(argv[4]);
        std::stringstream start_index_ss(arg4);
        std::size_t start_index;
        start_index_ss >> start_index;
        auto arg5 = std::string(argv[5]);
        std::stringstream end_index_ss(arg5);
        std::size_t end_index;
        end_index_ss >> end_index;
        double sigma = std::stod(std::string(argv[6]));
        dct2Test3(arg1, arg2, arg3, start_index, end_index, 80, 100, 8, 8, sigma);
    }
    else
    {
        //dct2Test3("InputImages/RainImages/2", ".", "Dictionary", 1, 1);
        for(std::size_t sigma = 1; sigma < 10; ++sigma)
        {
            std::string input_path = "InputImages/RainImages/S__55246868.bmp";
            if (!std::filesystem::is_regular_file(input_path))
            {
                input_path = "../InputImages/RainImages/S__55246868.bmp";
            }
            if (!std::filesystem::is_regular_file(input_path))
            {
                input_path = "../../InputImages/RainImages/S__55246868.bmp";
            }
            if (!std::filesystem::is_regular_file(input_path))
            {
                input_path = "../../../InputImages/RainImages/S__55246868.bmp";
            }
            auto input_img = TinyDIP::bmp_read(input_path.c_str(), true);
            auto dictionary = load_dictionary();
            auto output_img = each_image(std::execution::seq, input_img, std::get<0>(dictionary), std::get<1>(dictionary), 8, 8, static_cast<double>(sigma) / 10.0);
            std::error_code ec;
            if (!std::filesystem::is_directory("OutputImages"))
            {
                std::filesystem::create_directories("OutputImages", ec);
            }
            if (ec)
            {
                std::cerr << ec.message();
            }
            if (!std::filesystem::is_directory("OutputImages/RainImages"))
            {
                std::filesystem::create_directories("OutputImages/RainImages", ec);
            }
            if (ec)
            {
                std::cerr << ec.message();
            }
            auto output_path = std::string("OutputImages/RainImages/S__55246868_") + std::to_string(static_cast<double>(sigma) / 10.0);
            std::cout << "Save output to " << output_path << '\n';
            TinyDIP::bmp_write(output_path.c_str(), output_img);
            TinyDIP::bmp_write((output_path + std::string("_difference")).c_str(), TinyDIP::difference(input_img, output_img));
        }
        

    }
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    if (elapsed_seconds.count() != 1)
    {
        std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " seconds.\n";
    }
    else
    {
        std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << " second.\n";
    }
    return EXIT_SUCCESS;
}