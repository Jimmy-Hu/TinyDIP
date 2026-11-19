/* Developed by Jimmy Hu */
/* Shared Implementation Header for SIFT CLI Plugins */

#ifndef TINYDIP_PLUGIN_SIFT_COMMON_H
#define TINYDIP_PLUGIN_SIFT_COMMON_H

#include "plugin_api.h"

namespace handlers
{
    namespace detail_sift
    {
        template <
            typename DoubleImageCallback,
            typename ImageLoaderFun = MetaImageIO::Loader,
            typename OStreamT = std::ostream
        >
        requires(std::invocable<ImageLoaderFun, const std::string_view, Workspace&> &&
                 std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
        constexpr void dispatch_double_elevated_sift_image(
            const std::string_view input_arg,
            Workspace& workspace,
            const std::string_view feature_label,
            DoubleImageCallback&& callback,
            OStreamT& os,
            ImageLoaderFun&& image_loader_fun)
        {
            auto process_image = [&]<typename ImageType>(ImageType&& input_img)
            {
                using DecayedImageType = std::remove_cvref_t<ImageType>;

                if constexpr (TinyDIP::is_bool_data_v<DecayedImageType> || TinyDIP::is_complex_data_v<DecayedImageType>)
                {
                    os << "Error: Input image type [" << get_type_name<DecayedImageType>()
                       << "] does not support " << feature_label << ".\n";
                }
                else
                {
                    using RawScalarT = TinyDIP::get_deep_scalar_t<DecayedImageType>;

                    if constexpr (std::same_as<RawScalarT, double>)
                    {
                        callback(std::forward<ImageType>(input_img));
                    }
                    else if constexpr (requires { TinyDIP::im2double(std::forward<ImageType>(input_img)); })
                    {
                        callback(TinyDIP::im2double(std::forward<ImageType>(input_img)));
                    }
                    else
                    {
                        os << "Error: Input image type [" << get_type_name<DecayedImageType>()
                           << "] cannot be converted to double precision for " << feature_label << ".\n";
                    }
                }
            };

            if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_image))
            {
                os << "Error: Memory variable not found or unsupported type.\n";
            }
        }
    }
}

#endif // TINYDIP_PLUGIN_SIFT_COMMON_H
