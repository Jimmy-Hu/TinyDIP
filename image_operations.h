#include "base_types.h"


namespace TinyDIP
{
    template<class ElementT>
    Image<ElementT> copyResizeBicubic(Image<ElementT> const& image, size_t width, size_t height)
    {
        auto output = Image<ElementT>(width, height);
        auto ratiox = (float)image.getSizeX() / (float)width;
        auto ratioy = (float)image.getSizeY() / (float)height;
        
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
                        ndata[(ndatay + 1) * 4 + (ndatax + 1)] = image.get(
                            clip(xMappingToOriginFloor + ndatax, 0, image.getSizeX() - 1), 
                            clip(yMappingToOriginFloor + ndatay, 0, image.getSizeY() - 1));
                    }
                    
                }
                output.set(x, y, bicubicPolate(ndata, xMappingToOriginFrac, yMappingToOriginFrac));
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
}