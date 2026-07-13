/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for Mathematical Operations */

#include "plugin_api.h"


extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto TransformerSchema = IOSchema{0, 1};
    constexpr auto CombinerSchema    = IOSchema{0, 2};

    registry.register_command("abs", "Calculate the absolute value of an image or container.", TransformerSchema,
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::abs(workspace, args, os);
        }
    );
    
    registry.register_command("add", "Add two images or containers pixel-wise.", CombinerSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::add(workspace, args, os);
        }
    );

    registry.register_command("max", "Calculate the maximum value of an image or container.", TransformerSchema,
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::max(workspace, args, os);
        }
    );
    
    registry.register_command("min", "Calculate the minimum value of an image or container.", TransformerSchema,
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::min(workspace, args, os);
        }
    );

    registry.register_command("subtract", "Subtract two images or containers pixel-wise.", CombinerSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::subtract(workspace, args, os);
        }
    );

    registry.register_command("sum", "Calculate the sum of all elements in an image or container.", TransformerSchema,
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::sum(workspace, args, os);
        }
    );
}