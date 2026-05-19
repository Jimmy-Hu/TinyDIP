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

void lanczosResamplingTest(
    std::string_view input_image_path = "../InputImages/1",
    std::string_view output_image_path = "../OutputImages/lanczosResamplingTest")
{
    auto input_img = TinyDIP::bmp_read(std::string(input_image_path).c_str(), false);
    auto output_img =
        TinyDIP::lanczos_resample(
            input_img,
            input_img.getWidth() * 2,
            input_img.getHeight() * 2
        );
    TinyDIP::bmp_write(
        std::string(output_image_path).c_str(),
        output_img);
    
}

//  is_execution_policy concept implementation
//  Concept ensuring the parameter is a valid execution policy
template<typename ExecutionPolicy>
concept is_execution_policy = std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>;

//  process_single_image template function implementation
template<class ExecutionPolicy>
requires is_execution_policy<ExecutionPolicy>
void process_single_image(ExecutionPolicy&& execution_policy, const std::filesystem::path& source_filename)
{
    if (!(source_filename.extension() == ".ppm" || source_filename.extension() == ".bmp"))
    {
        std::cout << "Skipping non-ppm / non-bmp file: " << source_filename.string() << '\n';
        return;
    }

    try
    {
        std::cout << "Processing image: " << source_filename.string() << '\n';
        TinyDIP::Image<TinyDIP::RGB> source_image(0, 0);
        if (source_filename.extension() == ".ppm")
        {
            source_image = TinyDIP::pnm::read(std::forward<ExecutionPolicy>(execution_policy), source_filename);
        }
        else
        {
            source_image = TinyDIP::bmp_read(source_filename);
        }
        
        auto output_image = TinyDIP::lanczos_resample(source_image, 720, 1380);
        
        // Place the output file in the same directory as the source
        const std::filesystem::path output_filename_bmp = source_filename.parent_path() / (source_filename.stem().string());
        
        if (!std::filesystem::exists(output_filename_bmp.replace_extension(.bmp)))
        {
            TinyDIP::pnm::write(
                output_image,
                output_filename_bmp.string().c_str()
            );
            std::cout << "Successfully saved: " << output_filename_bmp.string() << '\n';
        }
        else
        {
            std::cout << "Output file already exists, skipping write: " << output_filename_bmp.string() << '\n';
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception encountered during processing of " << source_filename.string() << ": " << e.what() << '\n';
    }
}

//  ProcessImageLambda struct definition
struct ProcessImageLambda
{
    void operator()(const std::filesystem::path& file_path) const
    {
        process_single_image(std::execution::seq, file_path);
    }
};

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer;
    lanczosResamplingTest();
    return EXIT_SUCCESS;
}