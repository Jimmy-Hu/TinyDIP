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

    // === Load All Images ===
    std::cout << "--- Loading " << image_paths.size() << " images ---\n";
    std::vector<TinyDIP::Image<TinyDIP::RGB>> images;
    for (const auto& path : image_paths)
    {
        auto img = TinyDIP::bmp_read(path.c_str(), true);
        if (img.getWidth() == 0 || img.getHeight() == 0)
        {
            std::cerr << "Error: Failed to read image from " << path << "\n";
            return EXIT_FAILURE;
        }
        std::cout << "Loaded image: " << path << " (" << img.getWidth() << "x" << img.getHeight() << ")\n";
        images.emplace_back(img);
    }

    // === Stitch Images Sequentially ===
    std::cout << "\n--- Starting Sequential Stitching ---\n";
    // Using a default ratio_threshold, but this could also be a command-line arg
    auto final_panorama = TinyDIP::stitch_sequential(images, 0.7);

    // === Save the final result ===
    if (final_panorama.getWidth() == 0 || final_panorama.getHeight() == 0)
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


    if (TinyDIP::bmp_write(output_base.c_str(), final_panorama) == 0)
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