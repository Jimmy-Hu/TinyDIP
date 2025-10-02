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


    if (preview_mode)
    {
        // === Preview Stitching Pipeline ===
        std::cout << "--- Starting Image Stitching PREVIEW Pipeline ---\n";
        std::cout << "Using Lowe's ratio threshold: " << ratio_threshold << "\n\n";

        // 1. Load images
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
        std::cout << "Images loaded successfully.\n";

        // 2. Resize images for preview
        const std::size_t preview_width = 400;
        const auto new_height1 = static_cast<std::size_t>(static_cast<double>(bmp1.getHeight()) * preview_width / bmp1.getWidth());
        const auto new_height2 = static_cast<std::size_t>(static_cast<double>(bmp2.getHeight()) * preview_width / bmp2.getWidth());

        std::cout << "Resizing images for preview to " << preview_width << "px width...\n";
        auto bmp1_small = TinyDIP::copyResizeBicubic(bmp1, preview_width, new_height1);
        auto bmp2_small = TinyDIP::copyResizeBicubic(bmp2, preview_width, new_height2);

        // 3. Stitch the small images
        auto stitched_preview = TinyDIP::imstitch(bmp1_small, bmp2_small, ratio_threshold);

        // 4. Save the preview result
        if (stitched_preview.getWidth() == 0 || stitched_preview.getHeight() == 0)
        {
            std::cerr << "Error: Preview stitching resulted in an empty image.\n";
            return EXIT_FAILURE;
        }

        const std::string preview_output_name = output_filename_base + "_preview";
        if (TinyDIP::bmp_write(preview_output_name.c_str(), stitched_preview) == 0)
        {
            std::cout << "Successfully saved stitched preview to " << preview_output_name << ".bmp\n";
        }
        else
        {
            std::cerr << "Error: Failed to save stitched preview.\n";
            return EXIT_FAILURE;
        }
        std::cout << "\n--- Preview Pipeline Complete ---\n";
    }
    else
    {
        // === Full Resolution Stitching Pipeline ===
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
    }

    return EXIT_SUCCESS;
}