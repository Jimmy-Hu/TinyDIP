/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for replace */

#include "../main.h"
#include "plugin_api.h"
#include "plugin_replace.h"

// Sliced Image Tuples
using image_types = tuple_map_t<image_t, core_numeric_types>;
using custom_and_multichannel_image_types = tuple_cat_t<
    tuple_map_t<image_t, all_custom_scalar_types>,
    tuple_map_t<image_t, all_multichannel_types>
>;
using complex_image_types = tuple_cat_t<
    tuple_map_t<image_t, all_complex_types>,
    tuple_map_t<image_t, all_complex_multichannel_types>
>;

// Sliced Container Tuples
using sequence_container_types = tuple_cat_t<
    all_vector_types,
    all_deque_types,
    all_list_types,
    all_array_types
>;

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

    registry.register_command(cmd_name, cmd_desc, IOSchema{0, 3},
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::replace(workspace, args, os);
        }
    );
}