#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_operations.h"
#include "../image_io.h"
#include "../timer.h"

template<typename T>
void pixelwise_transformTest(const size_t size = 10)
{
    auto img1 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<T>(3));
    auto img2 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<T>(3));
    auto img3 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<T>(3));
    auto img4 = TinyDIP::gaussianFigure2D(size, size, 5, 5, static_cast<T>(3));

    auto output = TinyDIP::pixelwise_transform
    (
        [](auto&& pixel_in_img1, auto&& pixel_in_img2, auto&& pixel_in_img3, auto&& pixel_in_img4)
        {
            return 2 * pixel_in_img1 + pixel_in_img2 - pixel_in_img3 * pixel_in_img4;
        },
        TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img1),
        TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img2),
        TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img3),
        TinyDIP::pixelwise_transform([](auto&& element) { return element; }, img4)
    );
    output.print();
}

int main(int argc, char* argv[])
{
    TinyDIP::Timer timer1;
    if (argc < 2)
    {
        pixelwise_transformTest<int>();
        pixelwise_transformTest<long>();
        pixelwise_transformTest<float>();
        pixelwise_transformTest<double>();
    }
    else if (argc == 2)
    {
        std::filesystem::path input_path = std::string(argv[1]);
        if (!std::filesystem::exists(input_path))
        {
            std::cerr << "File not found: " << input_path << '\n';
            return EXIT_SUCCESS;
        }
        TinyDIP::Image<TinyDIP::RGB> input_img(0, 0);
        if (input_path.extension() == ".bmp")
        {
            input_img = TinyDIP::bmp_read(input_path.string().c_str(), true);
        }
        else
        {
            input_img = TinyDIP::pnm::read(std::execution::par, input_path.string().c_str());
        }
        auto output_image = TinyDIP::pixelwise_transform(
            [&](const auto& input_pixel)
            {
                auto r_value = input_pixel.channels[0];
                auto g_value = input_pixel.channels[1];
                auto b_value = input_pixel.channels[2];
                if ((r_value >= 28) && (r_value <= 37) &&
                    (g_value >= 40) && (g_value <= 57) &&
                    (b_value >= 40) && (b_value <= 52))
                {
                    return TinyDIP::RGB{28, 28, 28};
                }
				return input_pixel;
            },
            input_img
        );
        TinyDIP::bmp_write("pixelwiseTransformOutput", output_image);
    }

    return EXIT_SUCCESS;
}


