/* Developed by Jimmy Hu */

#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"
#include "../timer.h"

void gaussianFigure2DTest();

int main()
{
    TinyDIP::Timer timer1;
    gaussianFigure2DTest();
    return EXIT_SUCCESS;
}

void gaussianFigure2DTest()
{
    TinyDIP::GaussianParameters2D<double> gaussianParameters2D;
    gaussianParameters2D.x0 = 512.0;
    gaussianParameters2D.y0 = 512.0;
    gaussianParameters2D.sigma_x = 22.0;
    gaussianParameters2D.sigma_y = 44.0;
    gaussianParameters2D.rho = 0.7;
    auto gaussian_plane =
        TinyDIP::gaussianFigure2D(
            static_cast<std::size_t>(1024),
            static_cast<std::size_t>(1024),
            gaussianParameters2D
        );
	std::cout << "Max value in Gaussian Plane: " << TinyDIP::max(gaussian_plane) << '\n';
    auto estimated_gaussian_parameter = TinyDIP::estimate_gaussian_parameters_2d(std::execution::par_unseq, gaussian_plane);
	std::cout << "Estimated Gaussian Parameters: " << estimated_gaussian_parameter << '\n';

    gaussian_plane = TinyDIP::multiplies(TinyDIP::normalize(gaussian_plane), 255.0);
    TinyDIP::bmp_write("test_gaussian",
        TinyDIP::constructRGB(
            TinyDIP::im2uint8(gaussian_plane),
            TinyDIP::im2uint8(gaussian_plane),
            TinyDIP::im2uint8(gaussian_plane)
        )
    );
    return;
}