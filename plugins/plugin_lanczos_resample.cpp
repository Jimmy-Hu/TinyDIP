/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for lanczos_resample command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  lanczos_resample function implementation
    constexpr void lanczos_resample(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<4>(
                "lanczos_resample [execution_policy] <input_img | $var> <output_img | $var> <width> <height> [a=3]", 
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    const std::size_t width = parse_arg<std::size_t>(filtered_args[2]);
                    const std::size_t height = parse_arg<std::size_t>(filtered_args[3]);
                    std::size_t a = 3;
                    
                    if (std::ranges::size(filtered_args) >= 5)
                    {
                        a = parse_arg<std::size_t>(filtered_args[4]);
                    }

                    os << "Resizing " << filtered_args[0] << " to " << width << "x" << height << " with Lanczos radius " << a;
                    if (!std::ranges::empty(policy_str))
                    {
                        os << " (Policy: " << policy_str << ")";
                    }
                    os << "...\n";

                    return [width, height, a, policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            return TinyDIP::lanczos_resample(std::forward<ImageType>(img), width, height, static_cast<int>(a));
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (requires { TinyDIP::lanczos_resample(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height, static_cast<int>(a)); })
                            {
                                return TinyDIP::lanczos_resample(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), width, height, static_cast<int>(a));
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };

                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            );
        transform_handler(workspace, args, os);
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

    registry.register_command("lanczos_resample", "Resize an image using Lanczos resampling.", TransformerSchema,
        handlers::lanczos_resample
    );
}