#include "base_types.h"


namespace TinyDIP
{
    template<ElementT>
    Image<ElementT> copyResizeBicubic(Image<ElementT> const& image, int width, int height)
    {

    }

    template<class InputT>
    constexpr static auto bicubicPolate(const ElementT* const ndata, const InputT fracx, const InputT fracy)
    {
        auto x1 = cubicPolate( ndata[0], ndata[1], ndata[2], ndata[3], fracx );
        auto x2 = cubicPolate( ndata[4], ndata[5], ndata[6], ndata[7], fracx );
        auto x3 = cubicPolate( ndata[8], ndata[9], ndata[10], ndata[11], fracx );
        auto x4 = cubicPolate( ndata[12], ndata[13], ndata[14], ndata[15], fracx );

        return clip(cubicPolate( x1, x2, x3, x4, fracy ), 0.0, 255.0);
    }
}