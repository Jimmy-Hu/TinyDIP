/* Developed by Jimmy Hu */
/* Standalone Plugin Module 1 for abs: Fundamental Numeric Images */

#include "plugin_abs.h"

extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto GeneratorSchema = IOSchema{ -1, 1 };
    constexpr auto TerminatorSchema = IOSchema{ 0, -1 };
    constexpr auto TransformerSchema = IOSchema{ 0, 1 };
    constexpr auto CombinerSchema = IOSchema{ 0, 2 };
    constexpr auto IndependentSchema = IOSchema{ -1, -1 };

    // Ensure usage description is registered in the shared CommandUsageRegistry
    get_command_usage_registry().register_usage(
        "abs",
        "abs [execution_policy] <input_data | $var> <output_var | $var>"
    );

    // 1. Fundamental numeric images slice
    registry.register_command(
        "abs",
        "Calculate the absolute value of an image or container.",
        TransformerSchema,
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::abs<image_types>(workspace, args, os);
        }
    );
}