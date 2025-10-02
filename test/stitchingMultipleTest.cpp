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
        std::cerr << "Usage: " << argv[0] << " <output_base.bmp> <img1.bmp> <img2.bmp> [img3.bmp ...]\n";
        std::cerr << "Example: " << argv[0] << " panorama.bmp s1.bmp s2.bmp s3.bmp\n";
        return EXIT_FAILURE;
    }

    const std::string output_filename = argv[1];
    std::vector<std::string> image_paths;
    for (int i = 2; i < argc; ++i)
    {
        image_paths.emplace_back(argv[i]);
    }

    // === Sequentially Load and Stitch Images ===
    std::cout << "\n--- Starting Sequential Stitching ---\n";

    // 1. Load the first image to start the panorama
    std::cout << "Loading base image: " << image_paths[0] << "\n";
    TinyDIP::Image<TinyDIP::RGB> panorama = TinyDIP::bmp_read(image_paths[0].c_str(), true);
    if (panorama.getWidth() == 0 || panorama.getHeight() == 0)
    {
        std::cerr << "Error: Failed to read the first image from " << image_paths[0] << "\n";
        return EXIT_FAILURE;
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
        // Using a default ratio_threshold, but this could also be a command-line arg
        auto H = TinyDIP::find_stitch_homography(panorama, next_image, 0.7);
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
    }


    // === Save the final result ===
    if (panorama.getWidth() == 0 || panorama.getHeight() == 0)
    {
        std::cerr << "Error: Multi-image stitching resulted in an empty image.\n";
        return EXIT_FAILURE;
    }

    // We pass the full filename, so we strip the .bmp if it exists
    // and then let bmp_write add it back, to be consistent.
    std::string output_base = output_filename;
    if (output_base.ends_with(".bmp"))
    {
        output_base = output_base.substr(0, output_base.length() - 4);
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