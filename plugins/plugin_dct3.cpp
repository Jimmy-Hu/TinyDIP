/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for dct2 command */

#include "plugin_api.h"


extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto GeneratorSchema = IOSchema{ -1, 1 };
    constexpr auto TerminatorSchema = IOSchema{ 0, -1 };
    constexpr auto TransformerSchema = IOSchema{ 0, 1 };
    constexpr auto CombinerSchema = IOSchema{ 0, 2 };
    constexpr auto IndependentSchema = IOSchema{ -1, -1 };

    registry.register_command("dct3", "Calculate three-dimensional Discrete Cosine Transformation for an image.", TransformerSchema, 
        handlers::dct3
    );
}