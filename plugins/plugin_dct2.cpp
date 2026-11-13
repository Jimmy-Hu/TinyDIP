/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for dct2 command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  dct2 template function implementation
    template <std::ranges::input_range RangeT = std::span<const std::string_view>>
    requires (std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string_view> or
              std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string> or
              std::convertible_to<std::ranges::range_value_t<RangeT>, std::string_view> or
              std::convertible_to<std::ranges::range_value_t<RangeT>, std::string>)
    constexpr void dct2(
        Workspace& workspace,
        const RangeT& args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2>(
            "dct2",
            [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                os << "Calculating DCT-2 for " << filtered_args[0];
                if (!std::ranges::empty(policy_str))
                {
                    os << " (Policy: " << policy_str << ")";
                }
                os << "...\n";

                return [policy_str, &os]<typename ImageType>(ImageType && img) -> std::any
                {
                    auto exec_default = [&]() -> std::any
                    {
                        return TinyDIP::dct2(std::forward<ImageType>(img));
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (requires { TinyDIP::dct2(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img)); })
                        {
                            return TinyDIP::dct2(std::forward<ExecPolicy>(exec_policy), std::forward<ImageType>(img));
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

    get_command_usage_registry().register_usage(
        "dct2",
        "dct2 [execution_policy] <input_image | $var> <output_image | $var>"
    );

    registry.register_command("dct2", "Calculate Discrete Cosine Transformation for an image.", TransformerSchema, 
        handlers::dct2
    );
}