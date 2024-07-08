/* Developed by Jimmy Hu */

#include <chrono>
#include <execution>
#include <omp.h>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

template<class ExPo, class ElementT>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr static auto get_offset( ExPo execution_policy, 
                                  const TinyDIP::Image<ElementT>& input,
                                  const std::vector<TinyDIP::Image<ElementT>>& dictionary_x,
                                  const std::vector<TinyDIP::Image<ElementT>>& dictionary_y,
                                  const ElementT sigma, const ElementT threshold) noexcept
{
    auto output = TinyDIP::Image<ElementT>(input.getWidth(), input.getHeight());
    auto weights = TinyDIP::recursive_transform<1>(
        execution_policy,
        [&](auto&& element)
        { 
            return TinyDIP::normalDistribution1D(TinyDIP::manhattan_distance(input, element), sigma);
        }, dictionary_x);
    auto sum_of_weights = TinyDIP::recursive_reduce(weights, ElementT{});
    std::cout << "sum_of_weights: " << std::to_string(sum_of_weights) << '\n';
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
        for (size_t i = 0; i < dictionary_y.size(); ++i)
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
    auto input_hsv = TinyDIP::rgb2hsv(input_img);
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

    auto output_dct_blocks = TinyDIP::recursive_transform<2>(
        execution_policy,
        [&](auto&& element) { return TinyDIP::plus(element, get_offset(std::execution::seq, element, dictionary_x, dictionary_y, sigma, std::pow(10, -30))); },
        input_dct_blocks
        );
    
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
                                const std::size_t dic_end_index = 99)
{
    //***Load dictionary***
    std::vector<TinyDIP::Image<ElementT>> x, y;
    for (std::size_t i = dic_start_index; i <= dic_end_index; ++i)
    {
        std::string fullpath = dictionary_path + "/" + std::to_string(i);
        std::cout << "Dictionary path: " << fullpath << '\n';
        auto input_dbmp = TinyDIP::double_image::read(fullpath.c_str(), false);
        auto dct_block_x = TinyDIP::split(input_dbmp, input_dbmp.getWidth() / N1, input_dbmp.getHeight() / N2);
        TinyDIP::recursive_for_each<2>(
            std::execution::seq,
            [&](auto&& element) 
            {
                x.push_back(element);
            },
            dct_block_x);

        std::string fullpath_gt = dictionary_path + "/GT";
        auto input_dbmp_gt = TinyDIP::double_image::read(fullpath_gt.c_str(), false);
        auto dct_block_y = TinyDIP::split(input_dbmp_gt, input_dbmp_gt.getWidth() / N1, input_dbmp_gt.getHeight() / N2);
        TinyDIP::recursive_for_each<2>(
            std::execution::seq,
            [&](auto&& element)
            {
                y.push_back(element);
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
    
    auto dictionary = load_dictionary(dictionary_path, dic_start_index, dic_end_index);
    
    for (std::size_t i = start_index; i <= end_index; ++i)
    {
        std::string fullpath = input_folder + "/" + std::to_string(i);
        std::cout << "fullpath: " << fullpath << '\n';
        std::string output_path = output_folder + "/" + std::to_string(i);
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), false);
        auto output_img = each_image(std::execution::seq, input_img, std::get<0>(dictionary), std::get<1>(dictionary), N1, N2, sigma);
        std::cout << "Save output to " << output_path << '\n';
        TinyDIP::bmp_write(output_path.c_str(), output_img);
    }
    return;
}

int main(int argc, char* argv[])
{
    auto start = std::chrono::system_clock::now();
    std::cout << std::to_string(argc) << '\n';
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
        dct2Test3("InputImages/RainVideos/2", ".", "Dictionary", 1, 1);
    }
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return 0;
}