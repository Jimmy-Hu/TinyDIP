/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for find command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  find template function implementation
    template <
        std::ranges::input_range RangeT,
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires ((std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string_view> or
               std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string> or
               std::convertible_to<std::ranges::range_value_t<RangeT>, std::string_view> or
               std::convertible_to<std::ranges::range_value_t<RangeT>, std::string>) and
              std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void find(
        Workspace& workspace,
        const RangeT& args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        const std::string_view policy_str = extract_policy_string(args);
        const auto filtered_args = args_filter(args);

        if (std::ranges::size(filtered_args) < 2)
        {
            std::print(os, "Usage: find [execution_policy] <input_container | $var> <target_value | $var> [output_var | $var]\n");
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view target_arg = filtered_args[1];
        const std::string_view output_arg = std::ranges::size(filtered_args) > 2 ? filtered_args[2] : "";

        std::print(os, "Finding {} in {}", target_arg, input_arg);
        if (!std::ranges::empty(policy_str))
        {
            std::print(os, " (Policy: {})", policy_str);
        }
        std::print(os, "...\n");

        auto process_input = [&]<typename DataT, typename DecayedT = std::remove_cvref_t<DataT>>(DataT&& data)
        {
            // Outer range firewall: prevents instantiating range traits on non-range types like TinyDIP::Image
            if constexpr (std::ranges::input_range<DecayedT>)
            {
                using ElementT = std::ranges::range_value_t<DecayedT>;

                auto execute_find = [&]<typename TargetT>(TargetT&& target_val)
                {
                    using TargetDecayedT = std::remove_cvref_t<TargetT>;

                    // Strict C++20 cross-type and proxy-reference comparison firewall to protect unconstrained PSTL std::find
                    if constexpr (std::equality_comparable_with<ElementT, TargetDecayedT> &&
                                  requires(std::ranges::range_reference_t<DecayedT> ref, const TargetDecayedT& t)
                                  {
                                      { ref == t } -> std::convertible_to<bool>;
                                      { t == ref } -> std::convertible_to<bool>;
                                  })
                    {
                        auto exec_default = [&]() -> std::any
                        {
                            if constexpr (requires { std::ranges::find(data, target_val); })
                            {
                                auto it = std::ranges::find(data, target_val);
                                if (it != std::ranges::end(data))
                                {
                                    return std::optional<std::size_t>{
                                        static_cast<std::size_t>(std::ranges::distance(std::ranges::begin(data), it))
                                    };
                                }
                                return std::optional<std::size_t>{std::nullopt};
                            }
                            else
                            {
                                throw std::invalid_argument("Input container does not support sequential find operations natively.");
                                return std::any{};
                            }
                        };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (std::ranges::forward_range<DecayedT> &&
                                          requires { std::find(std::forward<ExecPolicy>(exec_policy), std::ranges::begin(data), std::ranges::end(data), target_val); })
                            {
                                auto it = std::find(
                                    std::forward<ExecPolicy>(exec_policy),
                                    std::ranges::begin(data),
                                    std::ranges::end(data),
                                    target_val
                                );
                                if (it != std::ranges::end(data))
                                {
                                    return std::optional<std::size_t>{
                                        static_cast<std::size_t>(std::ranges::distance(std::ranges::begin(data), it))
                                    };
                                }
                                return std::optional<std::size_t>{std::nullopt};
                            }
                            else
                            {
                                throw std::invalid_argument("Input container does not support execution-policy multi-threaded find operations natively.");
                                return std::any{};
                            }
                        };

                        try
                        {
                            std::any result = dispatch_policy_string(policy_str, exec_policy, exec_default, os);

                            if (!result.has_value())
                            {
                                return;
                            }

                            const auto found_index_opt = std::any_cast<std::optional<std::size_t>>(result);

                            if (found_index_opt.has_value())
                            {
                                const std::size_t found_index = *found_index_opt;
                                if (!std::ranges::empty(output_arg))
                                {
                                    if (output_arg.starts_with('$'))
                                    {
                                        workspace.store(output_arg.substr(1), found_index);
                                        std::print(os, "Found at index {}. Saved find result to {}\n", found_index, output_arg);
                                    }
                                    else
                                    {
                                        std::print(os, "Error: Output must be a memory variable starting with '$'.\n");
                                    }
                                }
                                else
                                {
                                    std::print(os, "Found at index: {}\n", found_index);
                                }
                            }
                            else
                            {
                                std::print(os, "Value {} not found in {}.\n", target_arg, input_arg);
                            }
                        }
                        catch (const std::exception& e)
                        {
                            std::print(os, "Error executing find: {}\n", e.what());
                        }
                    }
                    else
                    {
                        std::print(os, "Error: Cannot mathematically compare container element [{}] with target [{}].\n",
                                   get_type_name<ElementT>(), get_type_name<TargetDecayedT>());
                    }
                };

                // Firewall: Intercept target parameter logic explicitly to support both workspace variables and raw literals
                if (target_arg.starts_with('$'))
                {
                    bool found = false;
                    auto try_var = [&]<typename T>() -> bool
                    {
                        if (auto* ptr = workspace.template retrieve<T>(target_arg.substr(1)))
                        {
                            execute_find(*ptr);
                            return true;
                        }
                        return false;
                    };

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
                        ElementT parsed_target = parse_arg<ElementT>(target_arg);
                        execute_find(parsed_target);
                    }
                    catch (const std::exception& e)
                    {
                        std::print(os, "Error parsing target value '{}' as scalar: {}\n", target_arg, e.what());
                    }
                }
            }
            else
            {
                std::print(os, "Error: Input [{}] is not a container and does not support find operations.\n", get_type_name<DecayedT>());
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

    registry.register_command("find", "Find the first occurrence index of a specific value in a container.", CombinerSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::find(workspace, args, os);
        }
    );
}