/* Develop by Jimmy Hu */

#ifndef ImageOperations_H
#define ImageOperations_H

#include "base_types.h"
#include "image.h"

namespace TinyDIP
{
    // Forward Declaration class Image
    template <typename ElementT>
    class Image;

    template<class ElementT>
    Image<ElementT> copyResizeBicubic(Image<ElementT> const& image, size_t width, size_t height)
    {
        auto output = Image<ElementT>(width, height);
        auto ratiox = (float)image.getWidth() / (float)width;
        auto ratioy = (float)image.getHeight() / (float)height;
        
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                float xMappingToOrigin = (float)x * ratiox;
                float yMappingToOrigin = (float)y * ratioy;
                float xMappingToOriginFloor = floor(xMappingToOrigin);
                float yMappingToOriginFloor = floor(yMappingToOrigin);
                float xMappingToOriginFrac = xMappingToOrigin - xMappingToOriginFloor;
                float yMappingToOriginFrac = yMappingToOrigin - yMappingToOriginFloor;
                
                ElementT ndata[4 * 4];
                for (int ndatay = -1; ndatay <= 2; ndatay++)
                {
                    for (int ndatax = -1; ndatax <= 2; ndatax++)
                    {
                        ndata[(ndatay + 1) * 4 + (ndatax + 1)] = image.at(
                            clip(xMappingToOriginFloor + ndatax, 0, image.getWidth() - 1), 
                            clip(yMappingToOriginFloor + ndatay, 0, image.getHeight() - 1));
                    }
                    
                }
                output.at(x, y) = bicubicPolate(ndata, xMappingToOriginFrac, yMappingToOriginFrac);
            }
        }
        return output;
    }

    template<class ElementT, class InputT>
    constexpr static auto bicubicPolate(const ElementT* const ndata, const InputT fracx, const InputT fracy)
    {
        auto x1 = cubicPolate( ndata[0], ndata[1], ndata[2], ndata[3], fracx );
        auto x2 = cubicPolate( ndata[4], ndata[5], ndata[6], ndata[7], fracx );
        auto x3 = cubicPolate( ndata[8], ndata[9], ndata[10], ndata[11], fracx );
        auto x4 = cubicPolate( ndata[12], ndata[13], ndata[14], ndata[15], fracx );

        return clip(cubicPolate( x1, x2, x3, x4, fracy ), 0.0, 255.0);
    }

    template<class InputT1, class InputT2>
    constexpr static auto cubicPolate(const InputT1 v0, const InputT1 v1, const InputT1 v2, const InputT1 v3, const InputT2 frac)
    {
        auto A = (v3-v2)-(v0-v1);
        auto B = (v0-v1)-A;
        auto C = v2-v0;
        auto D = v1;
        return D + frac * (C + frac * (B + frac * A));
    }

    template<class InputT1, class InputT2, class InputT3>
    constexpr static auto clip(const InputT1 input, const InputT2 lowerbound, const InputT3 upperbound)
    {
        if (input < lowerbound)
        {
            return static_cast<InputT1>(lowerbound);
        }
        if (input > upperbound)
        {
            return static_cast<InputT1>(upperbound);
        }
        return input;
    }
}

#endif