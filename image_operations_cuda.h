/* Developed by Jimmy Hu */

#ifndef TINYDIP_IMAGE_OPERATIONS_CUDA_H
#define TINYDIP_IMAGE_OPERATIONS_CUDA_H

#include "image.h"
#include "linear_algebra.h"

namespace TinyDIP
{
    /**
     * @brief GPU-accelerated perspective transformation using CUDA.
     *
     * This function is a wrapper that manages memory transfers to and from the GPU
     * and launches the CUDA kernel to perform the image warping.
     *
     * @tparam ElementT The pixel element type of the image. Must be an arithmetic type.
     * @tparam FloatingType The floating-point type for matrix calculations (e.g., double, float).
     * @param src The source image to be warped.
     * @param H The 3x3 homography matrix.
     * @param out_width The desired width of the output image.
     * @param out_height The desired height of the output image.
     * @return The warped image.
     */
    template<
        arithmetic ElementT,
        std::floating_point FloatingType = double
    >
    Image<ElementT> warp_perspective_cuda(
        const Image<ElementT>& src,
        const linalg::Matrix<FloatingType>& H,
        const std::size_t out_width,
        const std::size_t out_height
    );

} // namespace TinyDIP

#endif // TINYDIP_IMAGE_OPERATIONS_CUDA_H