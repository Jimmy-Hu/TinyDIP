/* Developed by Jimmy Hu */
/* Shared Implementation Header for SIFT CLI Plugins */

#ifndef TINYDIP_PLUGIN_SIFT_COMMON_H
#define TINYDIP_PLUGIN_SIFT_COMMON_H

#include <concepts>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include "plugin_api.h"

namespace handlers
{
    namespace detail_sift
    {
        template <typename RawScalarT>
        concept is_already_double_sift_scalar =
            std::same_as<RawScalarT, double> ||
            std::same_as<RawScalarT, TinyDIP::RGB_DOUBLE> ||
            std::same_as<RawScalarT, TinyDIP::HSV>;

        template <typename ImgT>
        concept im2double_convertible_image = requires(ImgT&& img)
        {
            TinyDIP::im2double(std::move(img));
        };

        template <typename Callback, typename ImgT>
        concept invocable_after_im2double =
            im2double_convertible_image<ImgT> &&
            std::invocable<Callback&, decltype(TinyDIP::im2double(std::declval<ImgT>()))>;

        template <typename Callback, typename ImgT>
        concept sift_callback_compatible_for_image =
            TinyDIP::is_bool_data_v<ImgT> ||
            TinyDIP::is_complex_data_v<ImgT> ||
            (is_already_double_sift_scalar<TinyDIP::get_deep_scalar_t<ImgT>> &&
             std::invocable<Callback&, ImgT>) ||
            (!is_already_double_sift_scalar<TinyDIP::get_deep_scalar_t<ImgT>> &&
             (!im2double_convertible_image<ImgT> || invocable_after_im2double<Callback, ImgT>));

        template <typename Callback, typename TupleT>
        struct is_double_elevated_sift_callback_for_tuple : std::false_type {};

        template <typename Callback, typename... ImgTs>
        struct is_double_elevated_sift_callback_for_tuple<Callback, std::tuple<ImgTs...>>
            : std::bool_constant<(sift_callback_compatible_for_image<Callback, ImgTs> && ...)> {};

        template <typename Callback, typename TupleT = master_image_types>
        concept double_elevated_sift_callback =
            is_double_elevated_sift_callback_for_tuple<Callback, TupleT>::value;

        template <
            double_elevated_sift_callback DoubleImageCallback,
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

                    if constexpr (is_already_double_sift_scalar<RawScalarT>)
                    {
                        std::invoke(callback, std::forward<ImageType>(input_img));
                    }
                    else if constexpr (requires { TinyDIP::im2double(std::forward<ImageType>(input_img)); })
                    {
                        std::invoke(callback, TinyDIP::im2double(std::forward<ImageType>(input_img)));
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
