/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for sin command */

#include "plugin_trig_common.h"

namespace handlers
{
    namespace detail_sin
    {
        struct SinOp
        {
            template <typename... Args>
            requires requires { TinyDIP::sin(std::declval<Args>()...); }
            static constexpr decltype(auto) apply(Args&&... call_args)
            {
                return TinyDIP::sin(std::forward<Args>(call_args)...);
            }
        };
    }

    template <
        string_token_range RangeT = std::span<const std::string_view>,
        typename OStreamT = std::ostream
    >
    requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
    constexpr void sin(
        Workspace& workspace,
        const RangeT& args,
        OStreamT& os = std::cout)
    {
        detail_trig::execute_trig_command<detail_sin::SinOp>(workspace, args, "sin", os);
    }
}

extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    register_plugin_command(
        registry,
        "sin",
        "Calculate the sine of an image, container, or scalar.",
        TransformerSchema,
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::sin(workspace, args, os);
        }
    );
}
