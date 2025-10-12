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
    std::cout << "--- Testing RANSAC vs. MSAC Homography Estimation ---\n";

    // === Argument Parsing ===
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <img1.bmp> <img2.bmp>\n";
        std::cerr << "Example: " << argv[0] << " s1.bmp s2.bmp\n";
        return EXIT_FAILURE;
    }
    const std::string file_path1 = argv[1];
    const std::string file_path2 = argv[2];

    // === Load Images ===
    auto img1 = TinyDIP::bmp_read(file_path1.c_str(), true);
    auto img2 = TinyDIP::bmp_read(file_path2.c_str(), true);
    if (img1.getWidth() == 0 || img2.getWidth() == 0)
    {
        std::cerr << "Error: Failed to load one or both images.\n";
        return EXIT_FAILURE;
    }
    std::cout << "Images loaded successfully.\n";

    // === Phase 1: Feature Detection and Matching ===
    const TinyDIP::SiftParams<> sift_params;
    auto v_plane1 = TinyDIP::getVplane(TinyDIP::rgb2hsv(img1));
    auto v_plane2 = TinyDIP::getVplane(TinyDIP::rgb2hsv(img2));

    auto keypoints1 = TinyDIP::SIFT_impl::get_potential_keypoint(v_plane1, sift_params.octaves_count, sift_params.number_of_scale_levels, sift_params.initial_sigma, sift_params.k, sift_params.contrast_check_threshold, sift_params.edge_response_threshold);
    auto keypoints2 = TinyDIP::SIFT_impl::get_potential_keypoint(v_plane2, sift_params.octaves_count, sift_params.number_of_scale_levels, sift_params.initial_sigma, sift_params.k, sift_params.contrast_check_threshold, sift_params.edge_response_threshold);

    if (keypoints1.empty() || keypoints2.empty())
    {
        std::cerr << "Error: No keypoints found in one of the images.\n";
        return EXIT_FAILURE;
    }

    std::vector<TinyDIP::SiftDescriptor> descriptors1, descriptors2;
    for (const auto& kp : keypoints1) descriptors1.emplace_back(TinyDIP::SIFT_impl::get_keypoint_descriptor(v_plane1, kp));
    for (const auto& kp : keypoints2) descriptors2.emplace_back(TinyDIP::SIFT_impl::get_keypoint_descriptor(v_plane2, kp));

    auto matches = TinyDIP::find_robust_matches(descriptors1, descriptors2, 0.7);
    if (matches.size() < 4)
    {
        std::cerr << "Error: Not enough robust matches found.\n";
        return EXIT_FAILURE;
    }

    // === Phase 2: Homography Calculation (RANSAC vs. MSAC) ===
    const double inlier_threshold = 2.0; // Use a reasonable pixel distance (e.g., 2.0 pixels)
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    // --- RANSAC Path ---
    std::cout << "\n--- Calculating Homography with RANSAC --- \n";
    auto ransac_H = TinyDIP::find_homography_robust(keypoints1, keypoints2, matches, rng, TinyDIP::RobustEstimatorMethod::RANSAC, 2000, inlier_threshold);
    auto ransac_refined_H = TinyDIP::refine_homography(keypoints1, keypoints2, matches, ransac_H, inlier_threshold);

    // --- MSAC Path ---
    std::cout << "\n--- Calculating Homography with MSAC --- \n";
    auto msac_H = TinyDIP::find_homography_robust(keypoints1, keypoints2, matches, rng, TinyDIP::RobustEstimatorMethod::MSAC, 2000, inlier_threshold);
    auto msac_refined_H = TinyDIP::refine_homography(keypoints1, keypoints2, matches, msac_H, inlier_threshold);


    // === Phase 3: Create and Save Both Stitched Images ===
    std::cout << "\nCreating stitched image using RANSAC result...\n";
    auto stitched_ransac = TinyDIP::create_stitched_image(img1, img2, ransac_refined_H);

    std::cout << "Creating stitched image using MSAC result...\n";
    auto stitched_msac = TinyDIP::create_stitched_image(img1, img2, msac_refined_H);

    // Save outputs
    if (stitched_ransac.getWidth() > 0)
    {
        TinyDIP::bmp_write("homography_test_RANSAC", stitched_ransac);
        std::cout << "Saved RANSAC result to 'homography_test_RANSAC.bmp'\n";
    }
    if (stitched_msac.getWidth() > 0)
    {
        TinyDIP::bmp_write("homography_test_MSAC", stitched_msac);
        std::cout << "Saved MSAC result to 'homography_test_MSAC.bmp'\n";
    }

    std::cout << "\n--- Test Complete ---\n";
    std::cout << "Please inspect the two output images to compare alignment quality.\n";

    return EXIT_SUCCESS;
}