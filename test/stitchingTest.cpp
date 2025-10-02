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
    std::vector<std::string> args(argv + 1, argv + argc);
    bool preview_mode = false;

    // Search for and remove the --preview flag
    auto it = std::find(args.begin(), args.end(), "--preview");
    if (it != args.end())
    {
        preview_mode = true;
        args.erase(it);
    }

    if (args.size() < 3 || args.size() > 4)
    {
        // Print usage information to standard error
        std::cerr << "Usage: " << argv[0] << " <img1.bmp> <img2.bmp> <output_base> [ratio_threshold] [--preview]\n";
        std::cerr << "Example (Full): " << argv[0] << " s1.bmp s2.bmp result 0.7\n";
        std::cerr << "Example (Preview): " << argv[0] << " s1.bmp s2.bmp result 0.7 --preview\n";
        std::cerr << "  - ratio_threshold (optional): A value between 0.0 and 1.0. Lower is stricter. Default is 0.7.\n";
        std::cerr << "  - --preview (optional): Runs the pipeline on downscaled images for a fast preview.\n";
        return EXIT_FAILURE;
    }

    // Assign command-line arguments to variables
    const std::string file_path1 = args[0];
    const std::string file_path2 = args[1];
    const std::string output_filename_base = args[2];
    double ratio_threshold = 0.7; // Default value

    if (args.size() == 4)
    {
        try
        {
            ratio_threshold = std::stod(args[3]);
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


    // === Load Full-Resolution Images ===
    std::cout << "--- Loading Full-Resolution Images ---\n";
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

    // === Phase 1: Find Homography (always on full-res images) ===
    auto H = TinyDIP::find_stitch_homography(bmp1, bmp2, ratio_threshold);
    if(H.empty())
    {
        std::cerr << "Stitching pipeline failed during homography calculation.\n";
        return EXIT_FAILURE;
    }


    // === Phase 2: Create Stitched Image (either preview or full-res) ===
    TinyDIP::Image<TinyDIP::RGB> stitched_result;
    std::string final_output_name = output_filename_base;

    if (preview_mode)
    {
        std::cout << "\n--- Creating Preview Image ---\n";
        final_output_name += "_preview";

        // 1. Resize images for preview
        const std::size_t preview_width = 400;
        const auto new_height1 = static_cast<std::size_t>(static_cast<double>(bmp1.getHeight()) * preview_width / bmp1.getWidth());
        const auto new_height2 = static_cast<std::size_t>(static_cast<double>(bmp2.getHeight()) * preview_width / bmp2.getWidth());

        std::cout << "Resizing images for preview to " << preview_width << "px width...\n";
        auto bmp1_small = TinyDIP::copyResizeBicubic(bmp1, preview_width, new_height1);
        auto bmp2_small = TinyDIP::copyResizeBicubic(bmp2, preview_width, new_height2);

        // 2. Scale the high-quality H for the preview images
        std::cout << "Scaling homography for preview...\n";
        auto H_small = TinyDIP::scale_homography(
            H,
            bmp1.getWidth(), bmp1.getHeight(), bmp1_small.getWidth(), bmp1_small.getHeight(),
            bmp2.getWidth(), bmp2.getHeight(), bmp2_small.getWidth(), bmp2_small.getHeight()
        );

        // 3. Create the stitched image using the SCALED homography and small images
        stitched_result = TinyDIP::create_stitched_image(bmp1_small, bmp2_small, H_small);
    }
    else
    {
        std::cout << "\n--- Creating Full-Resolution Image ---\n";
        // Create the stitched image using the high-quality H and full-res images
        stitched_result = TinyDIP::create_stitched_image(bmp1, bmp2, H);
    }
    

    // === Save the final result ===
    if (stitched_result.getWidth() == 0 || stitched_result.getHeight() == 0)
    {
        std::cerr << "Error: Stitching resulted in an empty image.\n";
        return EXIT_FAILURE;
    }

    if (TinyDIP::bmp_write(final_output_name.c_str(), stitched_result) == 0)
    {
        std::cout << "Successfully saved stitched image to " << final_output_name << ".bmp\n";
    }
    else
    {
        std::cerr << "Error: Failed to save stitched image.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\n--- Pipeline Complete ---\n";

    return EXIT_SUCCESS;
}