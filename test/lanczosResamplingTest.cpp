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
            source_image = TinyDIP::bmp_read(source_filename.string().c_str(), true);
        }
        TinyDIP::Image<TinyDIP::RGB> output_image(720, 1920);
        output_image = TinyDIP::paste(output_image, TinyDIP::lanczos_resample(source_image, 720, 1380), 0, 0);
        
        // Place the output file in the same directory as the source
        std::filesystem::path output_filename_bmp = source_filename.parent_path() / (source_filename.stem().string() + std::string("_lanczos_resample"));
        
        if (!std::filesystem::exists(output_filename_bmp.replace_extension(".bmp")))
        {
            TinyDIP::bmp_write(
                output_filename_bmp.replace_extension("").string().c_str(),
                output_image
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
    if (argc < 2)
    {
        lanczosResamplingTest();
    }
    else if (argc == 2)
    {
        const std::filesystem::path input_path = std::string(argv[1]);
        std::vector<std::filesystem::path> files_to_process;

        // Detect if input is a directory to execute batch processing
        if (std::filesystem::is_directory(input_path))
        {
            std::cout << "Directory detected. Scanning for .ppm / .bmp files...\n";
            
            for (const auto& entry : std::filesystem::directory_iterator(input_path))
            {
                if (entry.is_regular_file() && ((entry.path().extension() == ".ppm") || (entry.path().extension() == ".bmp")))
                {
                    files_to_process.emplace_back(entry.path());
                }
            }
        }
        // Detect if input is a single file
        else if (std::filesystem::is_regular_file(input_path))
        {
            std::cout << "Single file detected.\n";
            files_to_process.emplace_back(input_path);
        }
        else
        {
            std::cerr << "Error: Invalid input path provided: " << input_path.string() << '\n';
            return EXIT_FAILURE;
        }

        if (files_to_process.empty())
        {
            std::cout << "No .ppm / .bmp files found to process. Exiting.\n";
            return EXIT_SUCCESS;
        }

        std::cout << "Total images queued for processing: " << files_to_process.size() << '\n';

        // Batch process utilizing C++17/20 parallel algorithms with the struct-based lambda
        std::for_each(
            std::execution::par, 
            std::ranges::begin(files_to_process), 
            std::ranges::end(files_to_process), 
            ProcessImageLambda{}
        );
    }
    return EXIT_SUCCESS;
}