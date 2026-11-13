/* Developed by Jimmy Hu */
/* Standalone Header for Command Usage Description Management */

#ifndef TINYDIP_COMMAND_USAGE_REGISTRY_H
#define TINYDIP_COMMAND_USAGE_REGISTRY_H

#include <functional>
#include <iostream>
#include <map>
#include <print>
#include <ranges>
#include <string>
#include <string_view>

//  Centralized registry class managing command usage descriptions via a private map
class CommandUsageRegistry
{
private:
    std::map<std::string, std::string, std::less<>> all_description_;

public:
    CommandUsageRegistry()
    {
        all_description_.emplace("abs", "abs [execution_policy] <input_data | $var> <output_var | $var>");
        all_description_.emplace("add", "add [execution_policy] <input1_data | $var> <input2_data | $var> <output_var | $var>");
        all_description_.emplace("append_element", "append_element <input_container | $var> <element_val | $var> <$output_var>");
        all_description_.emplace("batch_add_zeros", "batch_add_zeros <input_dir> <output_dir>");
        all_description_.emplace("bicubic_resize", "bicubic_resize [execution_policy] <input_image | $var> <output_image | $var> <new_width> <new_height>");
        all_description_.emplace("cast", "cast [execution_policy] <input_data | $var> <output_var | $var> <target_type>");
        all_description_.emplace("clear", "clear");
        all_description_.emplace("constructRGB", "constructRGB [execution_policy] <R_img | $var> <G_img | $var> <B_img | $var> <output_img | $var>");
        all_description_.emplace("copy", "copy <$src_var> <$dst_var>");
        all_description_.emplace("count", "count [execution_policy] <input_data | $var> <target_value | $var> [output_var | $var]");
        all_description_.emplace("create_container", "create_container <prototype_element | $var> <output_container | $var>");
        all_description_.emplace("dct2", "dct2 [execution_policy] <input_image | $var> <output_image | $var>");
        all_description_.emplace("dct3", "dct3 [execution_policy] <input_container | $var> <output_container | $var>");
        all_description_.emplace("divides", "divides [execution_policy] <input1_data | $var> <input2_data | $var> <output_var | $var>");
        all_description_.emplace("erase_element", "erase_element <input_container | $var> <output_var | $var> <index>");
        all_description_.emplace("estimate_gaussian_params_2d", "estimate_gaussian_params_2d [execution_policy] <input_img | $var> <output_var | $var> [max_iterations=1000] [tolerance=1e-7]");
        all_description_.emplace("find", "find [execution_policy] <input_container | $var> <target_value | $var> [output_var | $var]");
        all_description_.emplace("help", "help [command_name]");
        all_description_.emplace("histogram", "histogram [execution_policy] <input_image | $var> [output_var | $var] [count_type]");
        all_description_.emplace("idct2", "idct2 [execution_policy] <input_image | $var> <output_image | $var>");
        all_description_.emplace("lanczos_resample", "lanczos_resample [execution_policy] <input_image | $var> <output_image | $var> <new_width> <new_height> [a]");
        all_description_.emplace("max", "max [execution_policy] <input_data | $var> [output_var | $var]");
        all_description_.emplace("min", "min [execution_policy] <input_data | $var> [output_var | $var]");
        all_description_.emplace("multiply", "multiply [execution_policy] <input1_data | $var> <input2_data_or_scalar | $var> <output_var | $var>");
        all_description_.emplace("ones", "ones <$output_var> <width> <height> [type]");
        all_description_.emplace("print", "print <input_data | $var>");
        all_description_.emplace("read", "read <filepath> [$output_var]");
        all_description_.emplace("remove", "remove <$var>");
        all_description_.emplace("rename", "rename <$old_var> <$new_var>");
        all_description_.emplace("replace", "replace [execution_policy] <input_data | $var> <target_value | $var> <new_value | $var> <output_var | $var>");
        all_description_.emplace("size", "size <input_container | $var> [output_var | $var]");
        all_description_.emplace("subtract", "subtract [execution_policy] <input1_data | $var> <input2_data | $var> <output_var | $var>");
        all_description_.emplace("sum", "sum [execution_policy] <input_data | $var> [output_var | $var]");
        all_description_.emplace("vars", "vars");
        all_description_.emplace("write", "write <$input_var> <filepath>");
        all_description_.emplace("zeros", "zeros <$output_var> <width> <height> [type]");
    }

    //  Register or update a command's usage string dynamically (e.g., from plugins)
    void register_usage(const std::string_view command_name, const std::string_view usage_string)
    {
        all_description_.insert_or_assign(std::string(command_name), std::string(usage_string));
    }

    //  Check whether a command name exists in the registry
    bool has_usage(const std::string_view command_name) const noexcept
    {
        return all_description_.find(command_name) != std::ranges::end(all_description_);
    }

    //  Retrieve usage description by command name (falls back to returning the input view if not registered)
    std::string_view get_usage(const std::string_view command_name_or_usage) const noexcept
    {
        if (auto it = all_description_.find(command_name_or_usage); it != std::ranges::end(all_description_))
        {
            return it->second;
        }
        return command_name_or_usage;
    }

    //  Read-only subscript operator for convenient lookup (e.g., registry["count"])
    std::string_view operator[](const std::string_view command_name_or_usage) const noexcept
    {
        return get_usage(command_name_or_usage);
    }

    //  Format and print the usage string along with execution policy hints when applicable
    void print_usage(std::ostream& os, const std::string_view command_name_or_usage) const
    {
        const std::string_view usage_string = get_usage(command_name_or_usage);
        std::print(os, "Usage: {}\n", usage_string);
        if (usage_string.find("[execution_policy]") != std::string_view::npos)
        {
            std::print(os, "       Optional Execution policies: seq, par, par_unseq, unseq\n");
        }
    }
};

//  Global singleton accessor for the CommandUsageRegistry
inline CommandUsageRegistry& get_command_usage_registry()
{
    static CommandUsageRegistry instance;
    return instance;
}

//  Helper to structurally format and output command usage descriptions by command name
inline void print_command_usage(
    std::ostream& os,
    const std::string_view command_name_or_usage)
{
    get_command_usage_registry().print_usage(os, command_name_or_usage);
}

#endif // TINYDIP_COMMAND_USAGE_REGISTRY_H