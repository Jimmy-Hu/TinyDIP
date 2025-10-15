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
    std::cout << "--- RANSAC Inlier Visualization Test ---\n";

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
    const double ratio_threshold = 0.75;

    std::cout << "\nUsing SIFT Parameters:\n" << sift_params << "\n";
    std::cout << "Using Lowe's Ratio Threshold: " << ratio_threshold << "\n\n";

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
    for(const auto& kp : keypoints1) descriptors1.emplace_back(TinyDIP::SIFT_impl::get_keypoint_descriptor(v_plane1, kp));
    for(const auto& kp : keypoints2) descriptors2.emplace_back(TinyDIP::SIFT_impl::get_keypoint_descriptor(v_plane2, kp));
    
    auto matches = TinyDIP::find_robust_matches(descriptors1, descriptors2, ratio_threshold);
    if (matches.size() < 4)
    {
        std::cerr << "Error: Not enough robust matches found.\n";
        return EXIT_FAILURE;
    }

    // === Phase 2: Find RANSAC Homography and Identify Inliers ===
    const double inlier_threshold = 2.0;
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    std::cout << "\n--- Calculating Homography with RANSAC --- \n";
    auto ransac_H = TinyDIP::find_homography_robust(keypoints1, keypoints2, matches, rng, TinyDIP::RobustEstimatorMethod::RANSAC, 2000, inlier_threshold);
    
    if (ransac_H.empty())
    {
        std::cerr << "Error: RANSAC failed to compute a homography.\n";
        return EXIT_FAILURE;
    }
    
    // --- Identify the inliers for the RANSAC model ---
    std::vector<std::pair<std::size_t, std::size_t>> ransac_inliers;
    const double inlier_threshold_sq = inlier_threshold * inlier_threshold;

    for (const auto& match : matches)
    {
        const auto& p1 = keypoints1[match.first];
        const auto& p2 = keypoints2[match.second];
        
        double w = ransac_H.at(2, 0) * p1.p[0] + ransac_H.at(2, 1) * p1.p[1] + ransac_H.at(2, 2);
        if (std::abs(w) < 1e-9) continue;

        double x_proj = (ransac_H.at(0, 0) * p1.p[0] + ransac_H.at(0, 1) * p1.p[1] + ransac_H.at(0, 2)) / w;
        double y_proj = (ransac_H.at(1, 0) * p1.p[0] + ransac_H.at(1, 1) * p1.p[1] + ransac_H.at(1, 2)) / w;
        double dx = x_proj - p2.p[0];
        double dy = y_proj - p2.p[1];

        if (dx * dx + dy * dy < inlier_threshold_sq)
        {
            ransac_inliers.push_back(match);
        }
    }
    
    std::cout << "Identified " << ransac_inliers.size() << " inliers for the RANSAC model.\n";
    
    // === Phase 3: Visualize Only the Inlier Matches ===
    std::cout << "\nCreating visualization of RANSAC inliers...\n";
    auto inlier_visualization = TinyDIP::draw_matches(img1, img2, keypoints1, keypoints2, ransac_inliers);

    // Save output
    const std::string output_filename = "ransac_inlier_visualization";
    if (inlier_visualization.getWidth() > 0)
    {
        TinyDIP::bmp_write(output_filename.c_str(), inlier_visualization);
        std::cout << "Saved RANSAC inlier visualization to '" << output_filename << ".bmp'\n";
    }

    std::cout << "\n--- Test Complete ---\n";
    std::cout << "Please inspect the output image to see which matches supported the incorrect RANSAC model.\n";

    return EXIT_SUCCESS;
}