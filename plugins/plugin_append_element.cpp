/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for replace */

#include "plugin_api.h"


extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto TransformerSchema = IOSchema{ 0, 1 };
    constexpr auto CombinerSchema = IOSchema{ 0, 2 };

    registry.register_command("append_element", "Append an element to the back of a container using emplace_back.", CombinerSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::append_element(workspace, args, os);
        }
    );
}