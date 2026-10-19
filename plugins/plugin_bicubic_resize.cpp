/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for bicubic_resize */

#include "plugin_api.h"


extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto TransformerSchema = IOSchema{ 0, 1 };
    constexpr auto CombinerSchema = IOSchema{ 0, 2 };

    registry.register_command("bicubic_resize", "Resize an image using Bicubic interpolation.", TransformerSchema, 
            handlers::bicubic_resize
    );
}