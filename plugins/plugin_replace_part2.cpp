/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for replace */

#include "../main.h"
#include "plugin_api.h"
#include "plugin_replace.h"

extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto GeneratorSchema = IOSchema{ -1, 1 };
    constexpr auto TerminatorSchema = IOSchema{ 0, -1 };
    constexpr auto TransformerSchema = IOSchema{ 0, 1 };
    constexpr auto CombinerSchema = IOSchema{ 0, 2 };
    constexpr auto IndependentSchema = IOSchema{ -1, -1 };

    constexpr std::string_view cmd_name = "replace";
    constexpr std::string_view cmd_desc =
        "Replace occurrences of a target value with a new value in an image or container.";

    registry.register_command(
        cmd_name,
        cmd_desc,
        IOSchema{0, 3},
        [&](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::replace<custom_and_multichannel_image_types, all_custom_scalar_types>(workspace, args, os);
        }
    );
}