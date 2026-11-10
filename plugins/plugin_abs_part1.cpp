/* Developed by Jimmy Hu */
/* Sharded Translation Unit 1 for abs: Fundamental Numeric Images */

#include "plugin_abs.h"

void register_abs_part1(CommandRegistry& registry)
{
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