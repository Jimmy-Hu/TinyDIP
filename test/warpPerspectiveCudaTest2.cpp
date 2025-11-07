/* Developed by Jimmy Hu */

#include <chrono>
#include <cuda_runtime.h> // Add for cudaDeviceReset()
#include <iostream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../image_operations_cuda.h"
#include "../timer.h"

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';
    
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_image_path>\n";
        return EXIT_FAILURE;
    }

    try
    {
        std::cout << "Loading source image: " << argv[1] << "\n";
        auto src_image = TinyDIP::bmp_read(argv[1], true);

        if ((src_image.getWidth() == 0) || (src_image.getHeight() == 0))
        {
            std::cerr << "Error: Failed to read image from " << argv[1] << "\n";
            return EXIT_FAILURE;
        }

        TinyDIP::linalg::Matrix<double> H(3, 3);
        for (int i = 0; i < 1000; ++i)
        {
            H.at(0,0) = 1.0; H.at(0,1) = 0.0; H.at(0,2) = 0.0;
            H.at(1,0) = 0.0; H.at(1,1) = 1.0; H.at(1,2) = 0.0;
            H.at(2,0) = 0.0; H.at(2,1) = static_cast<double>(i) / 1000.0; H.at(2,2) = 1.0;

            std::cout << "Homography matrix = \n" << H;

            std::size_t out_width = src_image.getWidth();
            std::size_t out_height = src_image.getHeight();

            std::cout << "Running CUDA version...\n";
            auto start_cuda = std::chrono::high_resolution_clock::now();
            auto warped_cuda = TinyDIP::warp_perspective_cuda(src_image, H, out_width, out_height);
            auto end_cuda = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> cuda_duration = end_cuda - start_cuda;
            std::string output_filename = "warped_cuda_" + std::to_string(i);
            TinyDIP::bmp_write(output_filename, warped_cuda);
            std::cout << "CUDA version finished in " << cuda_duration.count() << " ms. Saved to " << output_filename << ".bmp\n";
        }
        
    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << '\n';
        // Clean up CUDA resources before exiting on error
        cudaDeviceReset();
        return EXIT_FAILURE;
    }

    // Clean up CUDA resources before successful exit
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
