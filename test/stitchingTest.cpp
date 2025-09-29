/* Developed by Jimmy Hu */

#include <chrono>
#include <execution>
#include <map>
#include <omp.h>
#include <span>
#include <sstream>
#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "argc parameter: " << std::to_string(argc) << '\n';
    // === Argument Parsing ===
    if (argc != 4)
    {
        // Print usage information to standard error
        std::cerr << "Usage: " << argv[0] << " <input_image1.bmp> <input_image2.bmp> <output_filename_base>\n";
        std::cerr << "Example: " << argv[0] << " ../InputImages/s1.bmp ../InputImages/s2.bmp SIFT_Stitching_Result\n";
        return EXIT_FAILURE;
    }

    // Assign command-line arguments to variables
    const std::string file_path1 = argv[1];
    const std::string file_path2 = argv[2];
    const std::string output_filename_base = argv[3];


    // === Image Stitching Pipeline Example ===
    std::cout << "--- Starting Image Stitching Pipeline ---\n";

    // 1. Load images
    // The second parameter is 'true' because we are providing the full filename.
    auto bmp1 = TinyDIP::bmp_read(file_path1.c_str(), true);
    auto bmp2 = TinyDIP::bmp_read(file_path2.c_str(), true);
    if ( (bmp1.getWidth() == 0) || (bmp1.getHeight() == 0) || (bmp2.getWidth() == 0) || (bmp2.getHeight() == 0) )
    {
        std::cerr << "Fail to read image\n";
        return EXIT_FAILURE;
    }
    std::cout << "Image 1 size: " << bmp1.getWidth() << " x " << bmp1.getHeight() << "\n";
    std::cout << "Image 2 size: " << bmp2.getWidth() << " x " << bmp2.getHeight() << "\n";
    auto output_image = TinyDIP::imstitch(bmp1, bmp2);
    if (!TinyDIP::bmp_write(output_filename_base.c_str(), output_image))
    {
        std::cerr << "Fail to write image\n";
        return EXIT_FAILURE;
    }
	std::cout << "Image stitching completed. Output saved as " << output_filename_base << ".bmp\n";
    std::cout << "\n--- Pipeline Complete ---\n";

    return EXIT_SUCCESS;
}