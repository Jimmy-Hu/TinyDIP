/* Developed by Jimmy Hu */

/*
 * This file contains the CUDA C++ implementation (.cu).
 * It must be compiled by nvcc. It implements the wrapper
 * function declared in image_operations_cuda.h.
 */

 #include "image_operations_cuda.h" // Corresponding header

 #include <concepts>
 #include <cuda_runtime.h>
 #include <iostream>
 #include <stdexcept>

namespace TinyDIP
{
    // CUDA error checking macro
    #define CUDA_CHECK(err) { \
        cudaError_t err_code = err; \
        if (err_code != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
                      << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

    /**
     * @brief A __device__ function for the bicubic interpolation kernel.
     * This function can only be called from other device or global functions.
     * It is a direct translation of the CPU-side cubic_kernel.
     */
    template<arithmetic FloatingType = double>
    __device__ FloatingType cubic_kernel_device(const FloatingType x, const FloatingType a = -0.5)
    {
        const FloatingType abs_x = fabsf(x); // Use fabsf for float, fabs for double on device
        const FloatingType abs_x2 = abs_x * abs_x;
        const FloatingType abs_x3 = abs_x2 * abs_x;

        if (abs_x <= 1.0)
        {
            return (a + 2.0) * abs_x3 - (a + 3.0) * abs_x2 + 1.0;
        }
        else if (abs_x < 2.0)
        {
            return a * abs_x3 - 5.0 * a * abs_x2 + 8.0 * a * abs_x - 4.0 * a;
        }
        return 0.0;
    }

    /**
     * BicubicInterpolatorDevice struct implementation
     * @brief Functor for Bicubic Interpolation (Device-Side).
     */
    template<arithmetic ElementT, arithmetic FloatingType>
    struct BicubicInterpolatorDevice
    {
        __device__ auto operator()(
            const ElementT* src_data,
            const int src_width,
            const int src_height,
            const FloatingType x,
            const FloatingType y) const
        {
            // Fallback to a simple interpolation (nearest neighbor) for edges for simplicity in CUDA
            if (x < 1 || x >= src_width - 2 || y < 1 || y >= src_height - 2)
            {
                int ix = static_cast<int>(roundf(x));
                int iy = static_cast<int>(roundf(y));
                ix = max(0, min(src_width - 1, ix));
                iy = max(0, min(src_height - 1, iy));
                return src_data[iy * src_width + ix];
            }

            int x_floor = static_cast<int>(floorf(x));
            int y_floor = static_cast<int>(floorf(y));

            FloatingType total_value{};

            for (int j = 0; j <= 3; ++j)
            {
                int v = y_floor - 1 + j;
                FloatingType row_value{};
                for (int i = 0; i <= 3; ++i)
                {
                    int u = x_floor - 1 + i;
                    row_value += static_cast<FloatingType>(src_data[v * src_width + u]) * cubic_kernel_device(x - u);
                }
                total_value += row_value * cubic_kernel_device(y - v);
            }

            if constexpr (std::is_integral_v<ElementT>)
            {
                if (total_value > 255.0) return 255; // Simplified clamping for byte images
                if (total_value < 0.0) return 0;
            }

            return static_cast<ElementT>(total_value);
        }
    };

    /**
     * warp_perspective_kernel template function implementation
     * @brief The CUDA kernel that performs the perspective warp.
     * Each thread in the grid computes one pixel of the output image.
     */
    template<arithmetic ElementT, std::floating_point FloatingType, typename InterpolatorFunc>
    __global__ void warp_perspective_kernel(
        ElementT* warped_data,
        const ElementT* src_data,
        const int src_width,
        const int src_height,
        const int out_width,
        const int out_height,
        const FloatingType h11, const FloatingType h12, const FloatingType h13,
        const FloatingType h21, const FloatingType h22, const FloatingType h23,
        const FloatingType h31, const FloatingType h32, const FloatingType h33,
        InterpolatorFunc interpolator)
    {
        // Calculate the unique global x and y coordinates for this thread
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Ensure the thread is within the bounds of the output image
        if (x < out_width && y < out_height)
        {
            // Apply inverse homography to find the corresponding source coordinate
            const FloatingType w = h31 * x + h32 * y + h33;

            if (fabsf(w) > 1e-6) // Avoid division by zero
            {
                const FloatingType src_x = (h11 * x + h12 * y + h13) / w;
                const FloatingType src_y = (h21 * x + h22 * y + h23) / w;
                
                // If the source coordinate is within the source image bounds, interpolate and write the pixel
                if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height)
                {
                    // For simplicity, this example uses a single channel. For RGB, you'd handle 3 channels.
                    warped_data[y * out_width + x] = interpolator(
                        src_data, src_width, src_height, src_x, src_y);
                }
            }
        }
    }


    // Explicit template instantiation for float and double types with GrayScale.
    // This is required for the linker to find the implementation across translation units.
    // You would add instantiations for other types (e.g., RGB) as needed.
    template Image<unsigned char> warp_perspective_cuda<unsigned char, float, BicubicInterpolatorDevice>(
        const Image<unsigned char>& src, const linalg::Matrix<float>& H,
        const std::size_t out_width, const std::size_t out_height);

    template Image<unsigned char> warp_perspective_cuda<unsigned char, double, BicubicInterpolatorDevice>(
        const Image<unsigned char>& src, const linalg::Matrix<double>& H,
        const std::size_t out_width, const std::size_t out_height);


    /**
     * warp_perspective_cuda template function implementation
     * @brief The C++ wrapper function implementation.
     * This is the function that is exposed to the rest of your C++ code.
     */
    template<
        arithmetic ElementT,
        std::floating_point FloatingType,
        template<typename, std::floating_point> typename InterpolatorFuncHost
    >
    Image<ElementT> warp_perspective_cuda(
        const Image<ElementT>& src,
        const linalg::Matrix<FloatingType>& H_inv,
        const std::size_t out_width,
        const std::size_t out_height
    )
    {
        // 1. Allocate memory on the GPU (device)
        ElementT* d_src_data;
        ElementT* d_warped_data;
        const size_t src_bytes = src.count() * sizeof(ElementT);
        const size_t warped_bytes = out_width * out_height * sizeof(ElementT);

        CUDA_CHECK(cudaMalloc(&d_src_data, src_bytes));
        CUDA_CHECK(cudaMalloc(&d_warped_data, warped_bytes));

        // 2. Copy source image data from CPU (host) to GPU (device)
        CUDA_CHECK(cudaMemcpy(d_src_data, src.getImageData().data(), src_bytes, cudaMemcpyHostToDevice));
        // Initialize warped image memory to 0
        CUDA_CHECK(cudaMemset(d_warped_data, 0, warped_bytes));

        // 3. Configure and launch the kernel
        // Threads per block (a common choice is 16x16 or 32x32)
        dim3 threads_per_block(16, 16);
        // Number of blocks in the grid
        dim3 num_blocks(
            (out_width + threads_per_block.x - 1) / threads_per_block.x,
            (out_height + threads_per_block.y - 1) / threads_per_block.y
        );

        // Define the concrete interpolator type based on our hardcoded ElementT
        using InterpolatorType = InterpolatorFuncHost<ElementT, FloatingType>;

        warp_perspective_kernel<<<num_blocks, threads_per_block, InterpolatorType>>>(
            d_warped_data,
            d_src_data,
            src.getWidth(),
            src.getHeight(),
            out_width,
            out_height,
            H_inv.at(0,0), H_inv.at(0,1), H_inv.at(0,2),
            H_inv.at(1,0), H_inv.at(1,1), H_inv.at(1,2),
            H_inv.at(2,0), H_inv.at(2,1), H_inv.at(2,2),
            InterpolatorType{} // Pass a default-constructed functor
        );
        
        // Check for any errors during kernel launch
        CUDA_CHECK(cudaGetLastError());
        // Synchronize to ensure the kernel has finished before we copy back data
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Copy the result from GPU (device) back to CPU (host)
        Image<ElementT> warped_image(out_width, out_height);
        CUDA_CHECK(cudaMemcpy((void*)warped_image.getImageData().data(), d_warped_data, warped_bytes, cudaMemcpyDeviceToHost));

        // 5. Free GPU memory
        CUDA_CHECK(cudaFree(d_src_data));
        CUDA_CHECK(cudaFree(d_warped_data));

        return warped_image;
    }

} // namespace TinyDIP
