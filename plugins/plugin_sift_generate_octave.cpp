/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for sift_generate_octave command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  sift_generate_octave template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void sift_generate_octave(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        if (std::ranges::size(args) < 2)
        {
            os << "Usage: sift_generate_octave <input_img | $var> <output_var | $var> [levels=5] [initial_sigma=1.6] [k=1.414]\n";
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view output_arg = args[1];

        if (!output_arg.starts_with('$'))
        {
            os << "Error: Output must be a memory variable starting with '$'.\n";
            return;
        }

        const std::size_t levels = (std::ranges::size(args) > 2) ? parse_arg<std::size_t>(args[2]) : 5;
        const double initial_sigma = (std::ranges::size(args) > 3) ? parse_arg<double>(args[3]) : 1.6;
        const double k = (std::ranges::size(args) > 4) ? parse_arg<double>(args[4]) : std::numbers::sqrt2_v<double>;

        os << "Generating SIFT octave from " << input_arg << " with " << levels << " levels...\n";

        auto process_octave = [&]<typename ImageType>(ImageType&& input_img)
        {
            using DecayedImageType = std::remove_cvref_t<ImageType>;

            if constexpr (TinyDIP::is_bool_data_v<DecayedImageType> || TinyDIP::is_complex_data_v<DecayedImageType>)
            {
                os << "Error: Input image type [" << get_type_name<DecayedImageType>() << "] does not support SIFT octave generation.\n";
                return;
            }
            else
            {
                auto process_octave_impl = [&]<typename T>(T&& double_img)
                {
                    using ImplDecayedT = std::remove_cvref_t<T>;

                    if constexpr (requires { TinyDIP::SIFT_impl::generate_octave(std::forward<T>(double_img), levels, initial_sigma, k); })
                    {
                        auto result = TinyDIP::SIFT_impl::generate_octave(std::forward<T>(double_img), levels, initial_sigma, k);
                        workspace.store(output_arg.substr(1), std::move(result));
                        os << "Saved SIFT octave to " << output_arg << ".\n";
                    }
                    else
                    {
                        os << "Error: Elevated input image type [" << get_type_name<ImplDecayedT>() << "] does not support generate_octave.\n";
                    }
                };

                using RawScalarT = TinyDIP::get_deep_scalar_t<DecayedImageType>;

                if constexpr (std::same_as<RawScalarT, double>)
                {
                    process_octave_impl(std::forward<ImageType>(input_img));
                }
                else if constexpr (requires { TinyDIP::im2double(std::forward<ImageType>(input_img)); })
                {
                    process_octave_impl(TinyDIP::im2double(std::forward<ImageType>(input_img)));
                }
                else
                {
                    os << "Error: Input image type [" << get_type_name<DecayedImageType>() << "] cannot be converted to double precision for SIFT octave generation.\n";
                }
            }
        };

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_octave))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
        }
    }
}

extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto GeneratorSchema = IOSchema{ -1, 1 };
    constexpr auto TerminatorSchema = IOSchema{ 0, -1 };
    constexpr auto TransformerSchema = IOSchema{ 0, 1 };
    constexpr auto CombinerSchema = IOSchema{ 0, 2 };
    constexpr auto IndependentSchema = IOSchema{ -1, -1 };

    registry.register_command("sift_generate_octave", "Generate a SIFT octave (Difference of Gaussian images).", TransformerSchema, 
        [&](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::sift_generate_octave(workspace, args, os);
        }
    );
}