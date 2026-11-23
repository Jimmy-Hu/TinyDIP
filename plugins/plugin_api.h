/* Developed by Jimmy Hu */
/* Dynamic Module Export Interface for TinyDIP Plugin Architecture */

#ifndef TINYDIP_PLUGIN_API_H
#define TINYDIP_PLUGIN_API_H

#include "../main.h" // Includes CommandRegistry, Workspace, and core logic
#include "command_usage_registry.h"

//  Cross-platform DLL/SO Export Macros
#ifdef _WIN32
    #define TINYDIP_PLUGIN_EXPORT __declspec(dllexport)
#else
    #define TINYDIP_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

//  A standard C-ABI signature strictly preventing C++ compiler name mangling.
//  This allows the dynamic OS loader to locate the exact symbol "register_plugin_commands" natively.
extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry);

#endif//  Helper to register a command handler taking (Workspace&, std::span<const std::string_view>, std::ostream&)
//  without repeating redundant wrapper lambdas or local IOSchema constants in every plugin translation unit.
template <typename HandlerCallable>
inline void register_plugin_command(
    CommandRegistry& registry,
    const std::string_view name,
    const std::string_view description,
    const IOSchema schema,
    HandlerCallable&& handler_callable)
{
    if constexpr (std::same_as<std::remove_cvref_t<HandlerCallable>, CommandHandler>)
    {
        registry.register_command(name, description, schema, std::forward<HandlerCallable>(handler_callable));
    }
    else
    {
        registry.register_command(name, description, schema, CommandHandler(std::forward<HandlerCallable>(handler_callable)));
    }
}

