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

// calculate_total_cost template function implementation
// Helper function to manually calculate cost for a given homography and scorer
template<std::floating_point FloatingType = double, class Scorer>
FloatingType calculate_total_cost(
    const std::vector<TinyDIP::Point<2>>& keypoints1,
    const std::vector<TinyDIP::Point<2>>& keypoints2,
    const std::vector<std::pair<std::size_t, std::size_t>>& matches,
    const TinyDIP::linalg::Matrix<FloatingType>& H,
    const FloatingType inlier_threshold,
    const Scorer& scorer)
{
    const FloatingType threshold_sq = inlier_threshold * inlier_threshold;
    FloatingType total_cost = 0;

    for (const auto& match : matches)
    {
        const auto& p1 = keypoints1[match.first];
        const auto& p2 = keypoints2[match.second];

        const FloatingType p1x = static_cast<FloatingType>(p1.p[0]);
        const FloatingType p1y = static_cast<FloatingType>(p1.p[1]);
        const FloatingType p2x = static_cast<FloatingType>(p2.p[0]);
        const FloatingType p2y = static_cast<FloatingType>(p2.p[1]);

        FloatingType w = H.at(2, 0) * p1x + H.at(2, 1) * p1y + H.at(2, 2);
        
        FloatingType dist_sq;
        if (std::abs(w) < std::numeric_limits<FloatingType>::epsilon())
        {
            dist_sq = threshold_sq;
        }
        else
        {
            FloatingType x_proj = (H.at(0, 0) * p1x + H.at(0, 1) * p1y + H.at(0, 2)) / w;
            FloatingType y_proj = (H.at(1, 0) * p1x + H.at(1, 1) * p1y + H.at(1, 2)) / w;
            FloatingType dx = x_proj - p2x;
            FloatingType dy = y_proj - p2y;
            dist_sq = dx * dx + dy * dy;
        }
        total_cost += scorer(dist_sq, threshold_sq);
    }
    return total_cost;
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    std::cout << "--- Testing RANSAC vs. MSAC Homography Estimation ---\n";

    // === Argument Parsing ===
    std::string file_path1 = "", file_path2 = "";
    bool preview_mode = false;
    if (argc == 1)                          //  no specified input images, use default
    {
        file_path1 = "../InputImages/s1.bmp";
        file_path2 = "../InputImages/s2.bmp";
        preview_mode = true;
    }
    else if (argc == 3)
    {
        file_path1.append(argv[1]);
        file_path2.append(argv[2]);
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " <img1.bmp> <img2.bmp>\n";
        std::cerr << "Example: " << argv[0] << " s1.bmp s2.bmp\n";
        return EXIT_FAILURE;
    }

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
    TinyDIP::SiftParams<> sift_params;
    sift_params.octaves_count = 5;                  // Default is 4
    sift_params.number_of_scale_levels = 6;         // Default is 5
    const double ratio_threshold = 0.75;            // Default is 0.7

    std::cout << "\nUsing SIFT Parameters:\n" << sift_params << "\n";
    std::cout << "Using Lowe's Ratio Threshold: " << ratio_threshold << "\n\n";

    auto v_plane1 = TinyDIP::getVplane(TinyDIP::rgb2hsv(img1));
    auto v_plane2 = TinyDIP::getVplane(TinyDIP::rgb2hsv(img2));

    auto keypoints1 = TinyDIP::SIFT_impl::get_potential_keypoint(v_plane1, sift_params.octaves_count, sift_params.number_of_scale_levels, sift_params.initial_sigma, sift_params.k, sift_params.contrast_check_threshold, sift_params.edge_response_threshold);
    auto keypoints2 = TinyDIP::SIFT_impl::get_potential_keypoint(v_plane2, sift_params.octaves_count, sift_params.number_of_scale_levels, sift_params.initial_sigma, sift_params.k, sift_params.contrast_check_threshold, sift_params.edge_response_threshold);

    std::cout << "Detected " << keypoints1.size() << " keypoints in Image 1.\n";
    std::cout << "Detected " << keypoints2.size() << " keypoints in Image 2.\n";

    if (keypoints1.empty() || keypoints2.empty())
    {
        std::cerr << "Error: No keypoints found in one of the images.\n";
        return EXIT_FAILURE;
    }

    std::vector<TinyDIP::SiftDescriptor> descriptors1, descriptors2;
    for (const auto& kp : keypoints1) descriptors1.emplace_back(TinyDIP::SIFT_impl::get_keypoint_descriptor(v_plane1, kp));
    for (const auto& kp : keypoints2) descriptors2.emplace_back(TinyDIP::SIFT_impl::get_keypoint_descriptor(v_plane2, kp));

    auto matches = TinyDIP::find_robust_matches(descriptors1, descriptors2, ratio_threshold);
    if (matches.size() < 4)
    {
        std::cerr << "Error: Not enough robust matches found.\n";
        return EXIT_FAILURE;
    }

    // === Phase 2: Homography Calculation (RANSAC vs. MSAC) ===
    const double inlier_threshold = 2.0;
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    // --- RANSAC Path ---
    std::cout << "\n--- Calculating Homography with RANSAC --- \n";
    auto ransac_H = TinyDIP::find_homography_robust(keypoints1, keypoints2, matches, rng, TinyDIP::RobustEstimatorMethod::RANSAC, 2000, inlier_threshold);
    auto ransac_refined_H = TinyDIP::refine_homography(keypoints1, keypoints2, matches, ransac_H, inlier_threshold);

    // --- MSAC Path ---
    std::cout << "\n--- Calculating Homography with MSAC --- \n";
    auto msac_H = TinyDIP::find_homography_robust(keypoints1, keypoints2, matches, rng, TinyDIP::RobustEstimatorMethod::MSAC, 2000, inlier_threshold);
    auto msac_refined_H = TinyDIP::refine_homography(keypoints1, keypoints2, matches, msac_H, inlier_threshold);

    // === Phase 3: Print Matrices for Comparison ===
    std::cout << "\n--- Homography Matrix Comparison --- \n";
    std::cout << "Refined RANSAC Homography:\n" << ransac_refined_H << '\n';
    std::cout << "\nRefined MSAC Homography:\n" << msac_refined_H << '\n';


    // === Phase 4: Create and Save Both Stitched Images ===
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
    std::cout << "Please inspect the console output and the two output images.\n";

    return EXIT_SUCCESS;
}