/* Developed by Jimmy Hu */

#ifndef TINYDIP_IMAGE_OPERATIONS_CUDA_H
#define TINYDIP_IMAGE_OPERATIONS_CUDA_H

#include "image.h"
#include "linear_algebra.h"

namespace TinyDIP
{
    // --- Forward Declarations for BicubicInterpolatorDevice ---
    template<arithmetic ElementT, arithmetic FloatingType>
    struct BicubicInterpolatorDevice;

    /**
     * @brief GPU-accelerated perspective transformation using CUDA.
     *
     * This function is a wrapper that manages memory transfers to and from the GPU
     * and launches the CUDA kernel to perform the image warping.
     *
     * @tparam ElementT The pixel element type of the image. Must be an arithmetic type.
     * @tparam FloatingType The floating-point type for matrix calculations (e.g., double, float).
     * @param src The source image to be warped.
     * @param homography The 3x3 homography matrix.
     * @param out_width The desired width of the output image.
     * @param out_height The desired height of the output image.
     * @return The warped image.
     */
    template<
        arithmetic ElementT,
        std::floating_point FloatingType = double,
        typename InterpolatorType = BicubicInterpolatorDevice<ElementT, FloatingType>
    >
    Image<ElementT> warp_perspective_cuda(
        const Image<ElementT>& src,
        const linalg::Matrix<FloatingType>& homography,
        const std::size_t out_width,
        const std::size_t out_height
    );

    //  warp_perspective_cuda overload for multi-channel images
    template<
        typename ElementT,
        std::floating_point FloatingType = double
    >
    requires((std::same_as<ElementT, RGB>) || (std::same_as<ElementT, RGB_DOUBLE>) || (std::same_as<ElementT, HSV>))
    auto warp_perspective_cuda(
        const Image<ElementT>& src,
        const linalg::Matrix<FloatingType>& homography,
        const std::size_t out_width,
        const std::size_t out_height
    )
    {
        return apply_each(src, [&](auto&& planes) { return warp_perspective_cuda(planes, homography, out_width, out_height); });
    }

    /**
     * create_stitched_image_cuda template function implementation
     * @brief Phase 2 of stitching: Warps and blends images using a pre-computed homography.
     */
    template<std::floating_point FloatingType = double>
    Image<RGB> create_stitched_image_cuda(const Image<RGB>& img1, const Image<RGB>& img2, const linalg::Matrix<FloatingType>& H_in)
    {
        if (H_in.empty()) {
            std::cerr << "Cannot create stitched image with an empty homography.\n";
            return Image<RGB>();
        }

        // 1. Determine output canvas size by transforming the corners of img2
        auto H = linalg::invert(H_in);
        if (H.empty()) {
            std::cerr << "Could not invert homography. Cannot stitch images.\n";
            return Image<RGB>();
        }

        const FloatingType w2 = static_cast<FloatingType>(img2.getWidth()), h2 = static_cast<FloatingType>(img2.getHeight());
        std::vector<Point<2>> corners = { {0,0}, {static_cast<std::size_t>(w2 - 1), 0}, {0, static_cast<std::size_t>(h2 - 1)}, {static_cast<std::size_t>(w2 - 1), static_cast<std::size_t>(h2 - 1)} };
        FloatingType min_x = 0, max_x = static_cast<FloatingType>(img1.getWidth()), min_y = 0, max_y = static_cast<FloatingType>(img1.getHeight());

        for(const auto& p : corners)
        {
            const FloatingType px = static_cast<FloatingType>(p.p[0]);
            const FloatingType py = static_cast<FloatingType>(p.p[1]);
            FloatingType w = H.at(2,0) * px + H.at(2,1) * py + H.at(2,2);
            FloatingType x = (H.at(0,0) * px + H.at(0,1) * py + H.at(0,2)) / w;
            FloatingType y = (H.at(1,0) * px + H.at(1,1) * py + H.at(1,2)) / w;
            if(x < min_x) min_x = x;
            if(x > max_x) max_x = x;
            if(y < min_y) min_y = y;
            if(y > max_y) max_y = y;
        }

        const FloatingType trans_x = -min_x;
        const FloatingType trans_y = -min_y;
        const std::size_t out_width = static_cast<std::size_t>(std::ceil(max_x - min_x));
        const std::size_t out_height = static_cast<std::size_t>(std::ceil(max_y - min_y));

        linalg::Matrix<FloatingType> H_trans(3,3);
        H_trans.at(0,0) = 1; H_trans.at(0,1) = 0; H_trans.at(0,2) = trans_x;
        H_trans.at(1,0) = 0; H_trans.at(1,1) = 1; H_trans.at(1,2) = trans_y;
        H_trans.at(2,0) = 0; H_trans.at(2,1) = 0; H_trans.at(2,2) = 1;

        // Combine translation with the original homography
        auto H_final = linalg::multiply(H_trans, H_in);

        // 2. Warp img2 to align with img1's coordinate frame
        std::cout << "Warping image 2...\n";
        auto warped_img2 = warp_perspective_cuda<RGB, FloatingType>(img2, H_final, out_width, out_height);
        
        std::cout << "Blending images with linear feathering...\n";
        Image<RGB> stitched_image(out_width, out_height);
        
        const auto img1_start_x = static_cast<std::size_t>(trans_x);
        const auto img1_end_x = static_cast<std::size_t>(trans_x + img1.getWidth());
        
        #pragma omp parallel for
        for(std::size_t y = 0; y < out_height; ++y)
        {
            for(std::size_t x = 0; x < out_width; ++x)
            {
                const auto& pixel_warped = warped_img2.at(x, y);
                bool warped_has_content = (pixel_warped.channels[0] > 5 || pixel_warped.channels[1] > 5 || pixel_warped.channels[2] > 5);

                Point<2> p1_coords = { static_cast<std::size_t>(x - trans_x), static_cast<std::size_t>(y-trans_y)};
                bool p1_has_content = (x >= img1_start_x && x < img1_end_x && y >= static_cast<std::size_t>(trans_y) && y < static_cast<std::size_t>(trans_y + img1.getHeight()));

                if (p1_has_content && warped_has_content)
                {
                    const auto& pixel1 = img1.at(p1_coords.p[0], p1_coords.p[1]);
                    FloatingType alpha = static_cast<FloatingType>(p1_coords.p[0]) / img1.getWidth();

                    RGB blended_pixel;
                    blended_pixel.channels[0] = static_cast<std::uint8_t>(pixel1.channels[0] * (1.0 - alpha) + pixel_warped.channels[0] * alpha);
                    blended_pixel.channels[1] = static_cast<std::uint8_t>(pixel1.channels[1] * (1.0 - alpha) + pixel_warped.channels[1] * alpha);
                    blended_pixel.channels[2] = static_cast<std::uint8_t>(pixel1.channels[2] * (1.0 - alpha) + pixel_warped.channels[2] * alpha);
                    stitched_image.at(x, y) = blended_pixel;
                }
                else if (p1_has_content)
                {
                    stitched_image.at(x, y) = img1.at(p1_coords.p[0], p1_coords.p[1]);
                }
                else if (warped_has_content)
                {
                    stitched_image.at(x, y) = pixel_warped;
                }
            }
        }
        
        return stitched_image;
    }

} // namespace TinyDIP

#endif // TINYDIP_IMAGE_OPERATIONS_CUDA_H