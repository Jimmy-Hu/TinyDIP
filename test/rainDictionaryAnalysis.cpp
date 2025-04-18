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
#include "../timer.h"

//    load_dictionary Template Function Implementation
template<TinyDIP::arithmetic ElementT = double>
constexpr auto load_dictionary( const std::string_view dictionary_path = "Dictionary",
                                const std::size_t dic_start_index = 80,
                                const std::size_t dic_end_index = 99,
                                const std::size_t N1 = 8,
                                const std::size_t N2 = 8)
{
    auto dictionary_path_temp = std::string(dictionary_path);
    //  https://stackoverflow.com/a/63622202/6667035
    if (!std::filesystem::is_directory(dictionary_path_temp))
    {
        dictionary_path_temp = "../" + std::string(dictionary_path);
    }
    if (!std::filesystem::is_directory(dictionary_path_temp))
    {
        dictionary_path_temp = "../../" + std::string(dictionary_path);
    }
    if (!std::filesystem::is_directory(dictionary_path_temp))
    {
        dictionary_path_temp = "../../../" + std::string(dictionary_path);
    }
    if (!std::filesystem::is_directory(dictionary_path_temp))
    {
        dictionary_path_temp = "../../../../" + std::string(dictionary_path);
    }
    //***Load dictionary***
    std::vector<TinyDIP::Image<ElementT>> x, y;
    for (std::size_t i = dic_start_index; i <= dic_end_index; ++i)
    {
        std::string fullpath = std::string(dictionary_path_temp) + "/" + std::to_string(i);
        std::cout << "Dictionary path: " << fullpath << '\n';
        if (!std::filesystem::is_regular_file(fullpath + ".dbmp"))
        {
            std::cerr << "File " + fullpath + ".dbmp" + " not found!\n";
            continue;
        }
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
    std::tuple<std::vector<TinyDIP::Image<ElementT>>, std::vector<TinyDIP::Image<ElementT>>> output{x, y};
    return output;
}

template<class ExecutionPolicy, std::floating_point ElementT = double>
requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
constexpr static auto averageIntraEuclideanDistances(
    ExecutionPolicy&& execution_policy,
    const std::vector<TinyDIP::Image<ElementT>>& input,
    std::size_t index
)
{
    ElementT result{};
    std::size_t count{};
    for (std::size_t i = index + 1; i < input.size(); ++i)
    {
        result += TinyDIP::euclidean_distance(std::forward<ExecutionPolicy>(execution_policy), input[index], input[i]);
        ++count;
    }
    return result / static_cast<ElementT>(count);
}

//  fullAverageIntraEuclideanDistances Template Function Implementation
template<class ExecutionPolicy, std::floating_point ElementT = double>
requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
constexpr static auto fullAverageIntraEuclideanDistances(
    ExecutionPolicy&& execution_policy,
    const std::vector<TinyDIP::Image<ElementT>>& input
)
{
    std::vector<double> results(input.size());
    auto index_upper_bound = input.size() - 1;
    #pragma omp parallel for
    for (std::size_t i = 0; i < index_upper_bound; ++i)
    {
        results[i] = averageIntraEuclideanDistances(std::forward<ExecutionPolicy>(execution_policy), input, i);
    }
    return TinyDIP::arithmetic_mean(results);
}

//    rainDictionaryAnalysis Template Function Implementation
void rainDictionaryAnalysis(
                const std::string& output_folder = ".",
                const std::string& dictionary_path = "Dictionary",
                const std::size_t dic_start_index = 80, const std::size_t dic_end_index = 99,
                const std::size_t N1 = 8, const std::size_t N2 = 8) noexcept
{
    std::cout << "rainDictionaryAnalysis program..." << '\n';
    
    auto dictionary = load_dictionary(dictionary_path, dic_start_index, dic_end_index, N1, N2);
    auto dictionary_x = std::get<0>(dictionary);
    auto dictionary_y = std::get<1>(dictionary);
    #ifdef _HAS_CXX23
    std::cout << std::format("Average intra-Euclidean distances for x set: {}\n", 
        fullAverageIntraEuclideanDistances(std::execution::par, dictionary_x)
        );
    std::cout << std::format("Average intra-Euclidean distances for y set: {}\n",
        fullAverageIntraEuclideanDistances(std::execution::par, dictionary_y)
    );
    #else
    std::cout << "Average intra-Euclidean distances for x set: " <<
        fullAverageIntraEuclideanDistances(std::execution::par, dictionary_x) << '\n';
    std::cout << "Average intra-Euclidean distances for y set: " <<
        fullAverageIntraEuclideanDistances(std::execution::par, dictionary_y) << '\n';
    #endif
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
    TinyDIP::Timer timer1;
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';
    rainDictionaryAnalysis();
    return EXIT_SUCCESS;
}