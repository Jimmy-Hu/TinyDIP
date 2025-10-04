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
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <output_base.bmp> <img1.bmp> <img2.bmp> [img3.bmp ...] [ratio_threshold] [--save-intermediate]\n";
        std::cerr << "Example: " << argv[0] << " panorama.bmp s1.bmp s2.bmp 0.7\n";
        std::cerr << "  - ratio_threshold (optional): A value between 0.0 and 1.0. Lower is stricter. Default is 0.7.\n";
        std::cerr << "  - --save-intermediate (optional): Saves the panorama after each successful stitch.\n";
        return EXIT_FAILURE;
    }

    double ratio_threshold = 0.7; // Default value
    bool save_intermediate = false;
    std::vector<std::string> image_paths;
    const std::string output_filename = argv[1];

    // Parse arguments, separating image paths from flags/options
    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--save-intermediate")
        {
            save_intermediate = true;
        }
        else
        {
            try
            {
                // Try to parse as ratio threshold. This assumes it's the last argument.
                double potential_ratio = std::stod(arg);
                if (potential_ratio > 0.0 && potential_ratio < 1.0)
                {
                    ratio_threshold = potential_ratio;
                }
                else
                {
                    // If it's a number but not a valid ratio, treat it as a filename
                    image_paths.emplace_back(arg);
                }
            }
            catch (const std::exception&)
            {
                // If it's not a number, it's an image path
                image_paths.emplace_back(arg);
            }
        }
    }

    // Check if there are still enough images to stitch
    if (image_paths.size() < 2)
    {
        std::cerr << "Error: At least two input images are required for stitching.\n";
        return EXIT_FAILURE;
    }


    // === Sequentially Load and Stitch Images ===
    std::cout << "\n--- Starting Sequential Stitching ---\n";
    std::cout << "Using Lowe's ratio threshold: " << ratio_threshold << "\n";
    if (save_intermediate)
    {
        std::cout << "Intermediate images will be saved.\n";
    }

    // 1. Load the first image to start the panorama
    std::cout << "Loading base image: " << image_paths[0] << "\n";
    TinyDIP::Image<TinyDIP::RGB> panorama = TinyDIP::bmp_read(image_paths[0].c_str(), true);
    if (panorama.getWidth() == 0 || panorama.getHeight() == 0)
    {
        std::cerr << "Error: Failed to read the first image from " << image_paths[0] << "\n";
        return EXIT_FAILURE;
    }

    // We pass the full filename, so we strip the .bmp if it exists
    // for creating intermediate filenames.
    std::string output_base = output_filename;
    if (output_base.ends_with(".bmp"))
    {
        output_base = output_base.substr(0, output_base.length() - 4);
    }

    // 2. Loop through the rest of the images, reading and stitching one by one
    for (std::size_t i = 1; i < image_paths.size(); ++i)
    {
        std::cout << "\n--- Stitching image " << i + 1 << " (" << image_paths[i] << ") onto current panorama ---\n";

        // Load the next image in the sequence
        auto next_image = TinyDIP::bmp_read(image_paths[i].c_str(), true);
        if (next_image.getWidth() == 0 || next_image.getHeight() == 0)
        {
            std::cerr << "Error: Failed to read image from " << image_paths[i] << ". Stopping stitch process.\n";
            break; // Stop but proceed to save the intermediate result
        }

        // Use the existing two-image stitching functions
        auto H = TinyDIP::find_stitch_homography(panorama, next_image, ratio_threshold);
        if (H.empty())
        {
            std::cerr << "Stitching failed during homography calculation for image " << i + 1 << ". Returning intermediate result.\n";
            break;
        }

        auto next_panorama = TinyDIP::create_stitched_image(panorama, next_image, H);
        if (next_panorama.getWidth() == 0)
        {
            std::cerr << "Stitching failed during warping/blending of image " << i + 1 << ". Returning intermediate result.\n";
            break;
        }

        // The stitch was successful, update the panorama
        panorama = next_panorama;

        // Save the intermediate result if requested
        if (save_intermediate)
        {
            std::string intermediate_filename = output_base + "_step_" + std::to_string(i);
            if (TinyDIP::bmp_write(intermediate_filename.c_str(), panorama) == 0)
            {
                std::cout << "Successfully saved intermediate image to " << intermediate_filename << ".bmp\n";
            }
            else
            {
                std::cerr << "Warning: Failed to save intermediate image " << intermediate_filename << ".bmp\n";
            }
        }
    }


    // === Save the final result ===
    if (panorama.getWidth() == 0 || panorama.getHeight() == 0)
    {
        std::cerr << "Error: Multi-image stitching resulted in an empty image.\n";
        return EXIT_FAILURE;
    }

    if (TinyDIP::bmp_write(output_base.c_str(), panorama) == 0)
    {
        std::cout << "\nSuccessfully saved final panorama to " << output_base << ".bmp\n";
    }
    else
    {
        std::cerr << "Error: Failed to save final panorama.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\n--- Multi-Image Stitching Complete ---\n";

    return EXIT_SUCCESS;
}