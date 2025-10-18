/* Developed by Jimmy Hu */

#include <chrono>
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

        // Define a sample homography matrix for a simple shear and scale transformation
        TinyDIP::linalg::Matrix<double> H(3, 3);
        H.at(0,0) = 1.0; H.at(0,1) = 0.2; H.at(0,2) = 50.0;
        H.at(1,0) = 0.0; H.at(1,1) = 1.2; H.at(1,2) = 20.0;
        H.at(2,0) = 0.0001; H.at(2,1) = 0.0; H.at(2,2) = 1.0;

        std::cout << "Homography matrix = \n" << H;

        std::size_t out_width = src_image.getWidth();
        std::size_t out_height = src_image.getHeight();

        std::cout << "Running OpenMP version...\n";
        auto start_omp = std::chrono::high_resolution_clock::now();
        auto warped_omp = TinyDIP::warp_perspective(src_image, H, out_width, out_height);
        auto end_omp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> omp_duration = end_omp - start_omp;
        TinyDIP::bmp_write("warped_omp", warped_omp);
        std::cout << "OpenMP version finished in " << omp_duration.count() << " ms. Saved to warped_omp.bmp\n";
        
        std::cout << "Running CUDA version...\n";
        auto start_cuda = std::chrono::high_resolution_clock::now();
        auto warped_cuda = TinyDIP::warp_perspective_cuda(src_image, H, out_width, out_height);
        auto end_cuda = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cuda_duration = end_cuda - start_cuda;
        TinyDIP::bmp_write("warped_cuda", warped_cuda);
        std::cout << "CUDA version finished in " << cuda_duration.count() << " ms. Saved to warped_cuda.bmp\n";

        std::cout << "\nCUDA speedup: " << omp_duration.count() / cuda_duration.count() << "x\n";

    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
