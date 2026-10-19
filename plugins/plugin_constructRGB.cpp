/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for constructRGB */

#include "plugin_api.h"


extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto GeneratorSchema = IOSchema{ -1, 1 };
    constexpr auto TerminatorSchema = IOSchema{ 0, -1 };
    constexpr auto TransformerSchema = IOSchema{ 0, 1 };
    constexpr auto CombinerSchema = IOSchema{ 0, 2 };
    constexpr auto IndependentSchema = IOSchema{ -1, -1 };

    registry.register_command("constructRGB", "Merge separate R, G, and B planes into an RGB image.", IndependentSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::construct_rgb(workspace, args, os);
        }
    );
}