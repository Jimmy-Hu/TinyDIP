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
    if (argc < 4 || argc > 5)
    {
        // Print usage information to standard error
        std::cerr << "Usage: " << argv[0] << " <input_image1.bmp> <input_image2.bmp> <output_filename_base> [ratio_threshold]\n";
        std::cerr << "Example: " << argv[0] << " ../InputImages/s1.bmp ../InputImages/s2.bmp SIFT_Stitching_Result 0.7\n";
        std::cerr << "  - ratio_threshold (optional): A value between 0.0 and 1.0. Lower is stricter. Default is 0.7.\n";
        return EXIT_FAILURE;
    }

    // Assign command-line arguments to variables
    const std::string file_path1 = argv[1];
    const std::string file_path2 = argv[2];
    const std::string output_filename_base = argv[3];
    double ratio_threshold = 0.7; // Default value

    if (argc == 5)
    {
        try
        {
            ratio_threshold = std::stod(argv[4]);
            if (ratio_threshold <= 0.0 || ratio_threshold >= 1.0)
            {
                std::cerr << "Error: ratio_threshold must be between 0.0 and 1.0.\n";
                return EXIT_FAILURE;
            }
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Error: Invalid number provided for ratio_threshold.\n";
            return EXIT_FAILURE;
        }
        catch (const std::out_of_range& e)
        {
            std::cerr << "Error: ratio_threshold value is out of range.\n";
            return EXIT_FAILURE;
        }
    }


    // === Image Stitching Pipeline Example ===
    std::cout << "--- Starting Image Stitching Pipeline ---\n";
    std::cout << "Using Lowe's ratio threshold: " << ratio_threshold << "\n\n";

    // 1. Load images
    // The second parameter is 'true' because we are providing the full filename.
    auto bmp1 = TinyDIP::bmp_read(file_path1.c_str(), true);
    auto bmp2 = TinyDIP::bmp_read(file_path2.c_str(), true);

    if ((bmp1.getWidth() == 0) || (bmp1.getHeight() == 0))
    {
        std::cerr << "Error: Failed to read image from " << file_path1 << "\n";
        return EXIT_FAILURE;
    }
    if ((bmp2.getWidth() == 0) || (bmp2.getHeight() == 0))
    {
        std::cerr << "Error: Failed to read image from " << file_path2 << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "Image 1 loaded: " << bmp1.getWidth() << " x " << bmp1.getHeight() << "\n";
    std::cout << "Image 2 loaded: " << bmp2.getWidth() << " x " << bmp2.getHeight() << "\n\n";

    // 2. Stitch the images, passing the ratio threshold
    auto stitched_result = TinyDIP::imstitch(bmp1, bmp2, ratio_threshold);

    // 3. Save the result
    if (stitched_result.getWidth() == 0 || stitched_result.getHeight() == 0)
    {
        std::cerr << "Error: Stitching resulted in an empty image.\n";
        return EXIT_FAILURE;
    }

    // Your bmp_write function automatically appends ".bmp" to the filename.
    if (TinyDIP::bmp_write(output_filename_base.c_str(), stitched_result) == 0)
    {
        std::cout << "Successfully saved stitched image to " << output_filename_base << ".bmp\n";
    }
    else
    {
        std::cerr << "Error: Failed to save stitched image.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\n--- Pipeline Complete ---\n";

    return EXIT_SUCCESS;
}