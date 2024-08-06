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
    std::cout << std::to_string(argc) << '\n';
    if(argc == 2)
    {
        
    }
    else
    {
        auto dictionary = load_dictionary();
        for(std::size_t sigma = 1; sigma < 10; ++sigma)
        {
            std::string input_path = "InputImages/RainVideos/2.bmp";
            auto input_img = TinyDIP::bmp_read(input_path.c_str(), true);
            
            //auto output_img = each_image(std::execution::seq, input_img, std::get<0>(dictionary), std::get<1>(dictionary), 8, 8, static_cast<double>(sigma) / 10.0);
            //auto output_path = std::string("OutputImages/2_") + std::to_string(static_cast<double>(sigma) / 10.0);
            //std::cout << "Save output to " << output_path << '\n';
            //TinyDIP::bmp_write(output_path.c_str(), output_img);
            //TinyDIP::bmp_write((output_path + std::string("_difference")).c_str(), output_img);
        }
        

    }
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}