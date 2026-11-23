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
//  Helper to register a command handler taking (Workspace&, std::span<const std::string_view>, std::ostream&)
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

template <string_token_range RangeT, typename OStreamT = std::ostream>
requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
[[nodiscard]] constexpr std::optional<std::pair<std::string_view, std::string_view>> extract_two_workspace_var_names(
    OStreamT& os,
    const RangeT& args,
    const std::string_view cmd_name)
{
    if (std::ranges::size(args) < 2)
    {
        print_command_usage(os, cmd_name);
        return std::nullopt;
    }

    const std::string_view first_arg = sanitize_string_view(args[0]);
    const std::string_view second_arg = sanitize_string_view(args[1]);
    if (!first_arg.starts_with('$') || !second_arg.starts_with('$'))
    {
        os << "Error: Both arguments must be memory variables starting with '$'.\n";
        return std::nullopt;
    }

    const std::string_view first_name = sanitize_string_view(first_arg.substr(1));
    const std::string_view second_name = sanitize_string_view(second_arg.substr(1));
    if (first_name.empty() || second_name.empty())
    {
        os << "Error: Variable names cannot be empty.\n";
        return std::nullopt;
    }

    return std::pair{ first_name, second_name };
}

