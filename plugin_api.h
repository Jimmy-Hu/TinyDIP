/* Developed by Jimmy Hu */
/* Dynamic Module Export Interface for TinyDIP Plugin Architecture */

#ifndef TINYDIP_PLUGIN_API_H
#define TINYDIP_PLUGIN_API_H

#include "main.h" // Includes CommandRegistry, Workspace, and core logic

//  Cross-platform DLL/SO Export Macros
#ifdef _WIN32
    #define TINYDIP_PLUGIN_EXPORT __declspec(dllexport)
#else
    #define TINYDIP_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

//  A standard C-ABI signature strictly preventing C++ compiler name mangling.
//  This allows the dynamic OS loader to locate the exact symbol "register_plugin_commands" natively.
extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry);

#endif