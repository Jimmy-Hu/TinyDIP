#include "../base_types.h"
#include "../basic_functions.h"
#include "../image.h"
#include "../image_io.h"
#include "../image_operations.h"

void increaseIntensityTest();

int main()
{
    increaseIntensityTest();
    return 0;
}

void increaseIntensityTest()
{
    auto bmp1 = TinyDIP::bmp_read("../InputImages/1", false);
    auto hsv1 = TinyDIP::rgb2hsv(bmp1);
    if constexpr (false)
    {
        #pragma omp parallel for
        for (int i = 0; i < 1; i++)
        {
            auto output = TinyDIP::hsv2rgb(TinyDIP::constructHSV(
                TinyDIP::getHplane(hsv1),
                TinyDIP::getSplane(hsv1),
                TinyDIP::pixelwiseOperation([i](auto&& element) { return element + i; }, TinyDIP::getVplane(hsv1))
            ));
            TinyDIP::bmp_write(std::to_string(i), output);
        }
    }
    else
    {
        for (int i = 0; i < 1; i++)
        {
            auto output = TinyDIP::hsv2rgb(TinyDIP::constructHSV(
                TinyDIP::getHplane(hsv1),
                TinyDIP::getSplane(hsv1),
                TinyDIP::pixelwiseOperation([i](auto&& element) { return element + i; }, TinyDIP::getVplane(hsv1))
            ));
            TinyDIP::bmp_write(std::to_string(i), output);
        }
    }
    
}