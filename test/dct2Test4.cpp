/* Developed by Jimmy Hu
    dct2Test4 performs image super-resolution experiments
 */

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

//  get_block_output template function implementation
template<class ExPo, class ElementT>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr static auto get_block_output(
    ExPo execution_policy, 
    const TinyDIP::Image<ElementT>& input,
    const std::vector<TinyDIP::Image<ElementT>>& dictionary_x,
    const std::vector<TinyDIP::Image<ElementT>>& dictionary_y,
    const ElementT sigma,
    const ElementT threshold
) noexcept
{
    auto output = TinyDIP::zeros<ElementT>(dictionary_y.at(0).getWidth(), dictionary_y.at(0).getHeight());
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
        for (std::size_t i = 0; i < dictionary_y.size(); ++i)
        {
            outputs[i] = dictionary_y[i] * weights[i];
        }
        output = TinyDIP::recursive_reduce(outputs, output);
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

//  each_plane Template Function Implementation
template<class ExPo, class ElementT1, class ElementT2>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr auto each_plane(
    ExPo execution_policy,
    const TinyDIP::Image<ElementT2>& input_img,
    std::tuple<std::vector<TinyDIP::Image<ElementT1>>, std::vector<TinyDIP::Image<ElementT1>>>& dictionary,
    const std::size_t low_res_N1 = 8, const std::size_t low_res_N2 = 8, const ElementT1 sigma = 0.1
) noexcept
{
    auto image_255 = TinyDIP::Image<double>(input_img.getSize());
    image_255.setAllValue(255);
    auto input_dct_blocks = TinyDIP::recursive_transform<2>(
        std::execution::seq,
        [](auto&& element) { return TinyDIP::dct2(element); },
        TinyDIP::split(
            TinyDIP::divides(
                input_img,
                image_255),
            input_img.getWidth() / low_res_N1,
            input_img.getHeight() / low_res_N2)
    );
    auto output_dct_blocks = TinyDIP::recursive_transform<2>(
        execution_policy,
        [&](auto&& element)
        {
            return get_block_output(
                execution_policy,
                element,
                std::get<0>(dictionary),
                std::get<1>(dictionary),
                sigma,
                std::pow(10, -30)
            );
        },
        input_dct_blocks
    );
    auto output_img = TinyDIP::concat(
        TinyDIP::recursive_transform<2>(
            execution_policy,
            [](auto&& element) { return TinyDIP::idct2(element); },
            output_dct_blocks)
    );
    image_255 = TinyDIP::Image<double>(output_img.getSize());
    image_255.setAllValue(255);
    output_img = TinyDIP::pixelwise_multiplies(output_img, image_255);
    return output_img;
}

//  each_image Template Function Implementation
template<class ExPo, class ElementT1, class ElementT2>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr auto each_image( ExPo execution_policy, 
                 const TinyDIP::Image<ElementT2>& input_img,
                 std::tuple<std::vector<TinyDIP::Image<ElementT1>>, std::vector<TinyDIP::Image<ElementT1>>>& dictionary,
                 const std::size_t low_res_N1 = 8, const std::size_t low_res_N2 = 8, const ElementT1 sigma = 0.1) noexcept
{
    std::cout << "Call each_image function..." << '\n';
    auto mod_x = std::fmod(static_cast<double>(input_img.getWidth()), static_cast<double>(low_res_N1));
    auto mod_y = std::fmod(static_cast<double>(input_img.getHeight()), static_cast<double>(low_res_N2));

    auto input_subimage = TinyDIP::subimage(
                        input_img,
                        input_img.getWidth() - mod_x, input_img.getHeight() - mod_y,
                        static_cast<double>(input_img.getWidth()) / 2.0, static_cast<double>(input_img.getHeight()) / 2.0
                        );
    auto Rplane = TinyDIP::im2double(TinyDIP::getRplane(input_subimage));
    auto Rplane_output = each_plane(execution_policy, Rplane, dictionary, low_res_N1, low_res_N2, sigma);
    std::cout << "Rplane_output calculation done\n";
    auto Gplane = TinyDIP::im2double(TinyDIP::getGplane(input_subimage));
    auto Gplane_output = each_plane(execution_policy, Gplane, dictionary, low_res_N1, low_res_N2, sigma);
    std::cout << "Gplane_output calculation done\n";
    auto Bplane = TinyDIP::im2double(TinyDIP::getBplane(input_subimage));
    auto Bplane_output = each_plane(execution_policy, Bplane, dictionary, low_res_N1, low_res_N2, sigma);
    std::cout << "Bplane_output calculation done\n";
    return TinyDIP::constructRGB(
        TinyDIP::im2uint8(Rplane_output),
        TinyDIP::im2uint8(Gplane_output),
        TinyDIP::im2uint8(Bplane_output)
    );
}

namespace impl{
    //  load_dictionary_RGB_each_channel Template Function Implementation 
    template<TinyDIP::arithmetic ElementT = double>
    constexpr static auto load_dictionary_RGB_each_channel(
        std::string low_res_fullpath,
        std::string high_res_fullpath,
        const std::size_t dic_start_index = 1,
        const std::size_t dic_end_index = 10,
        std::size_t high_res_N1 = 80,
        std::size_t high_res_N2 = 80,
        std::size_t low_res_N1 = 8,
        std::size_t low_res_N2 = 8

    )
    {
        std::vector<TinyDIP::Image<ElementT>> x, y;
        std::cout << "LowRes image path: " << low_res_fullpath << '\n';
        auto input_dbmp = TinyDIP::double_image::read(low_res_fullpath.c_str(), false);
        auto dct_block_x = TinyDIP::split(input_dbmp, input_dbmp.getWidth() / low_res_N1, input_dbmp.getHeight() / low_res_N2);
        std::cout << "HighRes image path: " << high_res_fullpath << '\n';
        input_dbmp = TinyDIP::double_image::read(high_res_fullpath.c_str(), false);
        auto dct_block_y = TinyDIP::split(input_dbmp, input_dbmp.getWidth() / high_res_N1, input_dbmp.getHeight() / high_res_N2);
        if (dct_block_x.size() == dct_block_y.size() && dct_block_x.at(0).size() == dct_block_y.at(0).size())
        {
            x.reserve(dct_block_x.size() * dct_block_x.at(0).size());
            TinyDIP::recursive_for_each<2>(
                std::execution::seq,
                [&](auto&& element)
                {
                    x.emplace_back(element);
                },
                dct_block_x);


            y.reserve(dct_block_y.size() * dct_block_y.at(0).size());
            TinyDIP::recursive_for_each<2>(
                std::execution::seq,
                [&](auto&& element)
                {
                    y.emplace_back(element);
                },
                dct_block_y);
        }
        return std::make_tuple(x, y);
    }

    //    load_dictionary_RGB Template Function Implementation
    template<TinyDIP::arithmetic ElementT = double>
    constexpr auto load_dictionary_RGB( 
        const std::string_view dictionary_path = "Dictionary",
        const std::size_t dic_start_index = 1,
        const std::size_t dic_end_index = 10,
        std::size_t high_res_N1 = 80,
        std::size_t high_res_N2 = 80,
        std::size_t low_res_N1 = 8,
        std::size_t low_res_N2 = 8)
    {
        //***Load dictionary***
        std::vector<TinyDIP::Image<ElementT>> x, y;
        for (std::size_t i = dic_start_index; i <= dic_end_index; ++i)
        {
            std::string low_res_fullpath = std::string(dictionary_path) + "/LowRes/Bucubic0.1/" + std::to_string(i) + std::string("_R");
            std::string high_res_fullpath = std::string(dictionary_path) + "/HighRes/" + std::to_string(i) + std::string("_R");
            auto dictionary_R = load_dictionary_RGB_each_channel(
                low_res_fullpath,
                high_res_fullpath,
                dic_start_index,
                dic_end_index,
                high_res_N1,
                high_res_N2,
                low_res_N1,
                low_res_N2);
            for(auto&& each_x : std::get<0>(dictionary_R))
                x.emplace_back(each_x);
            for(auto&& each_y : std::get<1>(dictionary_R))
                y.emplace_back(each_y);
            low_res_fullpath = std::string(dictionary_path) + "/LowRes/Bucubic0.1/" + std::to_string(i) + std::string("_G");
            high_res_fullpath = std::string(dictionary_path) + "/HighRes/" + std::to_string(i) + std::string("_G");
            auto dictionary_G = load_dictionary_RGB_each_channel(
                low_res_fullpath,
                high_res_fullpath,
                dic_start_index,
                dic_end_index,
                high_res_N1,
                high_res_N2,
                low_res_N1,
                low_res_N2);
            for(auto&& each_x : std::get<0>(dictionary_G))
                x.emplace_back(each_x);
            for(auto&& each_y : std::get<1>(dictionary_G))
                y.emplace_back(each_y);
            low_res_fullpath = std::string(dictionary_path) + "/LowRes/Bucubic0.1/" + std::to_string(i) + std::string("_B");
            high_res_fullpath = std::string(dictionary_path) + "/HighRes/" + std::to_string(i) + std::string("_B");
            auto dictionary_B = load_dictionary_RGB_each_channel(
                low_res_fullpath,
                high_res_fullpath,
                dic_start_index,
                dic_end_index,
                high_res_N1,
                high_res_N2,
                low_res_N1,
                low_res_N2);
            for(auto&& each_x : std::get<0>(dictionary_B))
                x.emplace_back(each_x);
            for(auto&& each_y : std::get<1>(dictionary_B))
                y.emplace_back(each_y);
        }
        std::cout << "x count: " << x.size() << "\ty count: " << y.size() << '\n';
        std::tuple<std::vector<TinyDIP::Image<ElementT>>, std::vector<TinyDIP::Image<ElementT>>> output{x, y};
        return output;
    }
}

//    load_dictionary Template Function Implementation
template<TinyDIP::arithmetic ElementT = double>
constexpr auto load_dictionary( 
    const std::string_view dictionary_path = "Dictionary",
    const std::size_t dic_start_index = 1,
    const std::size_t dic_end_index = 10,
    std::size_t high_res_N1 = 80,
    std::size_t high_res_N2 = 80,
    std::size_t low_res_N1 = 8,
    std::size_t low_res_N2 = 8)
{
    //***Load dictionary***
    std::vector<TinyDIP::Image<ElementT>> x, y;
    for (std::size_t i = dic_start_index; i <= dic_end_index; ++i)
    {
        std::string low_res_fullpath = std::string(dictionary_path) + "/LowRes/Bucubic0.1/" + std::to_string(i);
        std::cout << "LowRes image path: " << low_res_fullpath << '\n';
        auto input_dbmp = TinyDIP::double_image::read(low_res_fullpath.c_str(), false);
        auto dct_block_x = TinyDIP::split(input_dbmp, input_dbmp.getWidth() / low_res_N1, input_dbmp.getHeight() / low_res_N2);
        std::string high_res_fullpath = std::string(dictionary_path) + "/HighRes/" + std::to_string(i);
        input_dbmp = TinyDIP::double_image::read(high_res_fullpath.c_str(), false);
        auto dct_block_y = TinyDIP::split(input_dbmp, input_dbmp.getWidth() / high_res_N1, input_dbmp.getHeight() / high_res_N2);
        if (dct_block_x.size() == dct_block_y.size() && dct_block_x.at(0).size() == dct_block_y.at(0).size())
        {
            x.reserve(dct_block_x.size() * dct_block_x.at(0).size());
            TinyDIP::recursive_for_each<2>(
                std::execution::seq,
                [&](auto&& element)
                {
                    x.emplace_back(element);
                },
                dct_block_x);


            y.reserve(dct_block_y.size() * dct_block_y.at(0).size());
            TinyDIP::recursive_for_each<2>(
                std::execution::seq,
                [&](auto&& element)
                {
                    y.emplace_back(element);
                },
                dct_block_y);
        }
    }
    std::cout << "x count: " << x.size() << "\ty count: " << y.size() << '\n';
    std::tuple<std::vector<TinyDIP::Image<ElementT>>, std::vector<TinyDIP::Image<ElementT>>> output{x, y};
    return output;
}


int main(int argc, char* argv[])
{
    auto start = std::chrono::system_clock::now();
    std::cout << "argc = " << std::to_string(argc) << '\n';
    if(argc == 2)
    {
        
    }
    else
    {
        auto dictionary = load_dictionary();
        std::string input_path = "InputImages/LowRes/Bucubic0.1/0001.bmp";
        auto input_img = TinyDIP::bmp_read(input_path.c_str(), true);
        auto output_img = each_image(std::execution::seq, input_img, dictionary, 8, 8, 0.1);
        auto output_path = std::string("OutputImages/0001_superres") + std::to_string(0.1);
        std::cout << "Save output to " << output_path << '\n';
        TinyDIP::bmp_write(output_path.c_str(), output_img);
        

    }
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}