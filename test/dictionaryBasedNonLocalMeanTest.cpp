#include <chrono>
#include <execution>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"

/*  Matlab version:
function Dictionary = CreateDictionary(ND, xsize, ysize, zsize)
    Dictionary.X = zeros(xsize, ysize, zsize, ND);
    Dictionary.Y = zeros(xsize, ysize, zsize, ND);
    for i = 1:ND
        Dictionary.X(:, :, :, i) = ones(xsize, ysize, zsize) .* (i / ND);
        Dictionary.Y(:, :, :, i) = ones(xsize, ysize, zsize) .* (1 + i / ND);
    end
end
*/

template<class ElementT = double>
constexpr static auto create_dictionary(const std::size_t ND, const std::size_t xsize, const std::size_t ysize, const std::size_t zsize)
{
    auto code_words_x = TinyDIP::n_dim_vector_generator<1>(std::vector<TinyDIP::Image<ElementT>>(), ND);
    auto code_words_y = TinyDIP::n_dim_vector_generator<1>(std::vector<TinyDIP::Image<ElementT>>(), ND);
    for (std::size_t i = 0; i < ND; ++i)
    {
        auto code_words_x_image = TinyDIP::Image<ElementT>(xsize, ysize);
        code_words_x_image.setAllValue(static_cast<ElementT>(i) / static_cast<ElementT>(ND));
        code_words_x[i] = TinyDIP::n_dim_vector_generator<1>(
            code_words_x_image, zsize);
        auto code_words_y_image = TinyDIP::Image<ElementT>(xsize, ysize);
        code_words_y_image.setAllValue(1.0 + static_cast<ElementT>(i) / static_cast<ElementT>(ND));
        code_words_y[i] = TinyDIP::n_dim_vector_generator<1>(code_words_y_image, zsize);
    }
    return std::make_tuple(code_words_x, code_words_y);
}

/*  Matlab version:
function [output] = dictionaryBasedNonlocalMean(Dictionary, input)
    gaussian_sigma = 0.1;
    gaussian_mean = 0;
    if size(Dictionary.X) ~= size(Dictionary.Y)
        disp("Size of data in dictionary incorrect.");
        output = [];
        return
    end
    [X, Y, Z, DataCount] = size(Dictionary.X);
    weightOfEach = zeros(1, DataCount);
    for i = 1:DataCount
        %   Gaussian of distance between X and input
        weightOfEach(i) = gaussmf(ManhattanDistance(input, Dictionary.X(:, :, :, i)), [gaussian_sigma gaussian_mean]);  
    end
    sumOfDist = sum(weightOfEach, 'all');
    output = zeros(X, Y, Z);
    %%% if sumOfDist too small
    if (sumOfDist < 1e-160)
        fprintf("sumOfDist = %d\n", sumOfDist);
        return;
    end
    for i = 1:DataCount
        output = output + Dictionary.Y(:, :, :, i) .* weightOfEach(i);
    end
    output = output ./ sumOfDist;
end
*/

template<class ExPo, class ElementT = double>
requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
constexpr static auto dictionaryBasedNonlocalMean(  ExPo execution_policy,
                                                    const std::tuple<
                                                        std::vector<std::vector<TinyDIP::Image<ElementT>>>,
                                                        std::vector<std::vector<TinyDIP::Image<ElementT>>>
                                                        >& dictionary,
                                                    const std::vector<TinyDIP::Image<ElementT>>& input,
                                                    const double gaussian_sigma = 3.0,
                                                    const double gaussian_mean = 0,
                                                    const double threshold = 1e-160) noexcept
{
    std::vector<TinyDIP::Image<ElementT>> output = 
        TinyDIP::n_dim_vector_generator<1>(
            TinyDIP::Image<ElementT>(input[0].getWidth(), input[0].getHeight()), input.size());
    auto code_words_x = std::get<0>(dictionary);
    auto code_words_y = std::get<1>(dictionary);
    if (code_words_x.size() != code_words_y.size())
    {
        throw std::runtime_error("Size of data in dictionary incorrect.");
    }
    auto weights = TinyDIP::recursive_transform<1>(
        execution_policy,
        [&](auto&& element)
        {
            return TinyDIP::normalDistribution1D(
                TinyDIP::recursive_reduce(
                    TinyDIP::recursive_transform<1>(
                        [&](auto&& each_plane_x, auto&& each_plane_input) { return TinyDIP::manhattan_distance(each_plane_x, each_plane_input); },
                        element, input),
                    ElementT{}) + gaussian_mean,
                gaussian_sigma);
        }, code_words_x);
    auto sum_of_weights = TinyDIP::recursive_reduce(weights, ElementT{});
    std::cout << "sum_of_weights: " << std::to_string(sum_of_weights) << '\n';
    if (sum_of_weights < threshold)
    {
        return output;
    }
    auto outputs = TinyDIP::recursive_transform<1>(
        [&](auto&& input1, auto&& input2)
        {
            auto image = TinyDIP::Image<ElementT>(input1[0].getWidth(), input1[0].getHeight());
            image.setAllValue(input2);
            return TinyDIP::pixelwise_multiplies(
                input1,
                TinyDIP::n_dim_vector_generator<1>(
                    image,
                    input1.size())
                );
        }, code_words_y, weights);
    
    for (std::size_t i = 0; i < outputs.size(); ++i)
    {
        output = TinyDIP::plus(output, outputs[i]);
    }
    auto image = TinyDIP::Image<ElementT>(output[0].getWidth(), output[0].getHeight());
    image.setAllValue(sum_of_weights);
    output = TinyDIP::divides(  output,
                                TinyDIP::n_dim_vector_generator<1>(
                                    image,
                                    output.size()
                                )
                             );
    return output;
}

void dictionaryBasedNonLocalMeanTest()
{
    std::size_t ND = 10;
    std::size_t xsize = 8;
    std::size_t ysize = 8;
    std::size_t zsize = 1;
    std::vector<TinyDIP::Image<double>> input;
    input.reserve(zsize);
    for (std::size_t z = 0; z < zsize; ++z)
    {
        auto image = TinyDIP::Image<double>(xsize, ysize);
        image.setAllValue(0.66);
        input.emplace_back(image);
    }
    dictionaryBasedNonlocalMean(
        std::execution::par,
        create_dictionary(ND, xsize, ysize, zsize),
        input
    ).at(0).print();
}

int main()
{
    auto start = std::chrono::system_clock::now();
    dictionaryBasedNonLocalMeanTest();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Computation finished at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << '\n';
    return EXIT_SUCCESS;
}