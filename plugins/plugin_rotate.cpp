/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for rotate command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  rotate function implementation
    constexpr void rotate(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout)
    {
        auto transform_handler = make_meta_transform_handler<3>(
            "rotate [execution_policy] <input_img | $var> <output_img | $var> <angle>", 
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                const double angle = parse_arg<double>(filtered_args[2]);

                os << "Rotating " << filtered_args[0] << " by " << angle;
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [angle, policy_str, &os]<typename ImageType>(ImageType&& img) -> std::any
                {
                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (requires { TinyDIP::rotate_detail_shear_transformation(std::forward<ImageType>(img), angle); })
                        {
                            return TinyDIP::rotate_detail_shear_transformation(std::forward<ImageType>(img), angle);
                        }
                        else
                        {
                            throw std::invalid_argument("Input image type does not support rotate_detail_shear_transformation.");
                            return std::any{};
                        }
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (requires { TinyDIP::rotate_detail_shear_transformation(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), angle); })
                        {
                            return TinyDIP::rotate_detail_shear_transformation(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img), angle);
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

    registry.register_command("rotate", "Rotate an image using shear transformations.", TransformerSchema,
        handlers::rotate
    );
}