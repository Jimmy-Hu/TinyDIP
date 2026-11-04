/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for count command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  count template function implementation
    template <
        std::ranges::input_range RangeT,
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires ((std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string_view> or
               std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string> or
               std::convertible_to<std::ranges::range_value_t<RangeT>, std::string_view> or
               std::convertible_to<std::ranges::range_value_t<RangeT>, std::string>) and
              std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void count(
        Workspace& workspace,
        const RangeT& args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        const std::string_view policy_str = extract_policy_string(args);
        const auto filtered_args = args_filter(args);

        if (std::ranges::size(filtered_args) < 2)
        {
            std::print(os, "Usage: count [execution_policy] <input_data | $var> <target_value | $var> [output_var | $var]\n");
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view target_arg = filtered_args[1];
        const std::string_view output_arg = std::ranges::size(filtered_args) > 2 ? filtered_args[2] : "";

        std::print(os, "Counting occurrences of {} in {}", target_arg, input_arg);
        if (!std::ranges::empty(policy_str))
        {
            std::print(os, " (Policy: {})", policy_str);
        }
        std::print(os, "...\n");

        auto process_input = [&]<typename DataT>(DataT&& data)
        {
            using DecayedT = std::remove_cvref_t<DataT>;
            
            // Extract the deep mathematical scalar natively avoiding template evaluation ambiguity
            using ElementT = TinyDIP::get_deep_scalar_t<DecayedT>;

            auto execute_count = [&]<typename TargetT>(TargetT&& target_val)
            {
                using TargetDecayedT = std::remove_cvref_t<TargetT>;
                
                // Strict C++20 cross-type comparison firewall to protect against unconstrained PSTL hard-errors
                if constexpr (requires(const ElementT& e, const TargetDecayedT& t) { { e == t } -> std::convertible_to<bool>; })
                {
                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (TinyDIP::is_Image<DecayedT>::value && requires { TinyDIP::count(std::forward<DataT>(data), std::forward<TargetT>(target_val)); })
                        {
                            return TinyDIP::count(std::forward<DataT>(data), std::forward<TargetT>(target_val));
                        }
                        else if constexpr (std::ranges::input_range<DecayedT> && requires { std::ranges::count(std::forward<DataT>(data), std::forward<TargetT>(target_val)); })
                        {
                            return static_cast<std::size_t>(std::ranges::count(std::forward<DataT>(data), std::forward<TargetT>(target_val)));
                        }
                        else
                        {
                            throw std::invalid_argument("Input type does not support counting operations natively.");
                            return std::any{};
                        }
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (TinyDIP::is_Image<DecayedT>::value && requires { TinyDIP::count(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), std::forward<TargetT>(target_val)); })
                        {
                            return TinyDIP::count(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), std::forward<TargetT>(target_val));
                        }
                        else if constexpr (std::ranges::input_range<DecayedT> && requires { std::count(std::forward<ExecPolicy>(exec_policy), std::ranges::begin(data), std::ranges::end(data), std::forward<TargetT>(target_val)); })
                        {
                            return static_cast<std::size_t>(std::count(std::forward<ExecPolicy>(exec_policy), std::ranges::begin(data), std::ranges::end(data), std::forward<TargetT>(target_val)));
                        }
                        else
                        {
                            throw std::invalid_argument("Input type does not support execution-policy multi-threaded counting operations natively.");
                            return std::any{};
                        }
                    };

                    try
                    {
                        std::any result = dispatch_policy_string(policy_str, exec_policy, exec_default, os);

                        // std::any{} indicates an error occurred and was caught safely in the default blocks
                        if (!result.has_value())
                        {
                            return;
                        }

                        const std::size_t final_count = std::any_cast<std::size_t>(result);

                        if (!std::ranges::empty(output_arg))
                        {
                            if (output_arg.starts_with('$'))
                            {
                                workspace.store(output_arg.substr(1), final_count);
                                std::print(os, "Saved count result to {}\n", output_arg);
                            }
                            else
                            {
                                std::print(os, "Error: Output must be a memory variable starting with '$'.\n");
                            }
                        }
                        else
                        {
                            std::print(os, "Count result: {}\n", final_count);
                        }
                    }
                    catch (const std::exception& e)
                    {
                        std::print(os, "Error calculating count: {}\n", e.what());
                    }
                }
                else
                {
                    throw std::invalid_argument(std::string("Error: Cannot mathematically compare [") + std::string(get_type_name<ElementT>()) + "] with [" + std::string(get_type_name<TargetDecayedT>()) + "].");
                }
            };

            // Firewall: Intercept target parameter logic explicitly to prevent interpreting raw literal numbers as missing filenames
            if (target_arg.starts_with('$'))
            {
                bool found = false;
                auto try_var = [&]<typename T>() -> bool
                {
                    if (auto* ptr = workspace.template retrieve<T>(target_arg.substr(1)))
                    {
                        execute_count(*ptr);
                        return true;
                    }
                    return false;
                };

                // Match dynamically strictly against scalars to prevent accidental matrix comparisons
                found = match_any_type<master_scalar_types>(try_var);
                if (!found)
                {
                    std::print(os, "Error: Target variable not found or unsupported type: {}\n", target_arg);
                }
            }
            else
            {
                try
                {
                    // Perfectly parse string literal into the container's native underlying scalar type
                    ElementT parsed_target = parse_arg<ElementT>(target_arg);
                    execute_count(parsed_target);
                }
                catch (const std::exception& e)
                {
                    std::print(os, "Error parsing target value '{}' as scalar: {}\n", target_arg, e.what());
                }
            }
        };

        if (!dispatch_data_operation<master_data_types>(input_arg, workspace, image_loader_fun, process_input))
        {
            std::print(os, "Error: Memory variable not found or unsupported type for input: {}\n", input_arg);
        }
    }
}

extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    // Define human-readable pipeline schema routing constants locally
    constexpr auto GeneratorSchema = IOSchema{ -1, 1 };
    constexpr auto TerminatorSchema = IOSchema{ 0, -1 };
    constexpr auto TransformerSchema = IOSchema{ 0, 1 };
    constexpr auto CombinerSchema = IOSchema{ 0, 2 };
    constexpr auto IndependentSchema = IOSchema{ -1, -1 };

    registry.register_command("count", "Count occurrences of a specific value in an image or container.", CombinerSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::count(workspace, args, os);
        }
    );
}