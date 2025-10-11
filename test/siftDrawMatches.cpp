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

    if (args.size() < 3)
    {
        // Print usage information to standard error
        std::cerr << "Usage: " << argv[0] << " <img1.bmp> <img2.bmp> <output_base>\n";
        std::cerr << "Example: " << argv[0] << " s1.bmp s2.bmp result\n";
        return EXIT_FAILURE;
    }

    // Assign command-line arguments to variables
    const std::string file_path1 = args[0];
    const std::string file_path2 = args[1];
    const std::string output_filename_base = args[2];

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

    const TinyDIP::SiftParams sift_params = {};
    std::cout << "Detecting SIFT features...\n";
    auto v_plane1 = TinyDIP::getVplane(TinyDIP::rgb2hsv(bmp1));
    auto v_plane2 = TinyDIP::getVplane(TinyDIP::rgb2hsv(bmp2));

    std::cout << "SIFT parameters:\n" << sift_params << "\n";

    auto keypoints1 = TinyDIP::SIFT_impl::get_potential_keypoint(
        v_plane1,
        sift_params.octaves_count,
        sift_params.number_of_scale_levels,
        sift_params.initial_sigma,
        sift_params.k,
        sift_params.contrast_check_threshold,
        sift_params.edge_response_threshold
    );
    auto keypoints2 = TinyDIP::SIFT_impl::get_potential_keypoint(
        v_plane2,
        sift_params.octaves_count,
        sift_params.number_of_scale_levels,
        sift_params.initial_sigma,
        sift_params.k,
        sift_params.contrast_check_threshold,
        sift_params.edge_response_threshold
    );

    std::cout << "Found " << keypoints1.size() << " keypoints in image 1 and " << keypoints2.size() << " in image 2.\n";
    std::cout << "Generating descriptors...\n";

    std::vector<TinyDIP::SiftDescriptor> descriptors1;
    descriptors1.reserve(keypoints1.size());
    for (const auto& kp : keypoints1) descriptors1.emplace_back(TinyDIP::SIFT_impl::get_keypoint_descriptor<double>(v_plane1, kp));

    std::vector<TinyDIP::SiftDescriptor> descriptors2;
    descriptors2.reserve(keypoints2.size());
    for (const auto& kp : keypoints2) descriptors2.emplace_back(TinyDIP::SIFT_impl::get_keypoint_descriptor<double>(v_plane2, kp));

    std::cout << "Matching features...\n";
    auto matches = TinyDIP::find_robust_matches(descriptors1, descriptors2, 0.7);

    if (matches.size() < 4)
    {
        std::cerr << "Error: Not enough robust matches found to compute homography.\n";
        return EXIT_FAILURE;
    }
    
    auto output_matches_img = TinyDIP::draw_matches(bmp1, bmp2, keypoints1, keypoints2, matches);
    TinyDIP::bmp_write(output_filename_base + "_matches.bmp", output_matches_img);

    std::cout << "\n--- Pipeline Complete ---\n";

    return EXIT_SUCCESS;
}