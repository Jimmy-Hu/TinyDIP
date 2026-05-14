/* Developed by Jimmy Hu */

#include <algorithm>
#include <cstdlib>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../cube.h"
#include "../cube_operations.h"
#include "../timer.h" 

//  is_execution_policy concept implementation
//  Concept ensuring the parameter is a valid execution policy
template<typename ExecutionPolicy>
concept is_execution_policy = std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>;


int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <input_image_ppm_or_directory_path>\n";
        return EXIT_SUCCESS;
    }

    const std::filesystem::path input_path = std::string(argv[1]);
    std::vector<std::filesystem::path> files_to_process;

    // Detect if input is a directory to execute batch processing
    if (std::filesystem::is_directory(input_path))
    {
        std::cout << "Directory detected. Scanning for .ppm files...\n";
        
        for (const auto& entry : std::filesystem::directory_iterator(input_path))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".ppm")
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
        std::cout << "No .ppm files found to process. Exiting.\n";
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

    return EXIT_SUCCESS;
}