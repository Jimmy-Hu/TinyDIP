/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for replace */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  replace template function implementation
    template <
        std::ranges::input_range RangeT,
        typename ImageLoaderFun = MetaImageIO::Loader,
        typename ImageSaverFun = MetaImageIO::Saver
    >
    requires ((std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string_view> or
               std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string> or
               std::convertible_to<std::ranges::range_value_t<RangeT>, std::string_view> or
               std::convertible_to<std::ranges::range_value_t<RangeT>, std::string>) and
              std::invocable<ImageLoaderFun, const std::string_view, Workspace&> &&
              std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::RGB>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<double>&&> &&
              std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::HSV>&&>)
    constexpr void replace(
        Workspace& workspace,
        const RangeT& args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{},
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        const std::string_view policy_str = extract_policy_string(args);
        const auto filtered_args = args_filter(args);

        if (std::ranges::size(filtered_args) < 4)
        {
            std::print(os, "Usage: replace [execution_policy] <input_data | $var> <target_value | $var> <new_value | $var> <output_var | $var>\n");
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view target_arg = filtered_args[1];
        const std::string_view new_val_arg = filtered_args[2];
        const std::string_view output_arg = filtered_args[3];

        std::print(os, "Replacing occurrences of {} with {} in {}", target_arg, new_val_arg, input_arg);
        if (!std::ranges::empty(policy_str))
        {
            std::print(os, " (Policy: {})", policy_str);
        }
        std::print(os, "...\n");

        auto process_input =
            [&]<typename DataT,
                typename DecayedT = std::remove_cvref_t<DataT>,
                typename ElementT = TinyDIP::get_deep_scalar_t<DecayedT>
        >(DataT &&data)
        {
            auto execute_replace = [&]<
                typename TargetT,
                typename NewT,
                typename TargetDecayedT = std::remove_cvref_t<TargetT>,
                typename NewDecayedT = std::remove_cvref_t<NewT>
            >(TargetT&& target_val, NewT&& new_val)
            {
                // Core SFINAE Firewall immunizing unconstrained PSTL algorithms from undefined operator overloads
                if constexpr (requires(const ElementT& e, const TargetDecayedT& t) { { e == t } -> std::convertible_to<bool>; } &&
                              std::convertible_to<NewDecayedT, ElementT>)
                {
                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr ((TinyDIP::is_Image<DecayedT>::value) && (requires { TinyDIP::replace(std::forward<DataT>(data), std::forward<TargetT>(target_val), std::forward<NewT>(new_val)); }))
                        {
                            return TinyDIP::replace(std::forward<DataT>(data), std::forward<TargetT>(target_val), std::forward<NewT>(new_val));
                        }
                        else if constexpr ((std::ranges::input_range<DecayedT>) && (requires { std::ranges::replace(std::declval<DecayedT&>(), std::forward<TargetT>(target_val), std::forward<NewT>(new_val)); }))
                        {
                            DecayedT result = data;
                            std::ranges::replace(result, std::forward<TargetT>(target_val), std::forward<NewT>(new_val));
                            return result;
                        }
                        else
                        {
                            throw std::invalid_argument("Input type does not support sequential replacement natively.");
                            return std::any{};
                        }
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr ((TinyDIP::is_Image<DecayedT>::value) && (requires { TinyDIP::replace(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), std::forward<TargetT>(target_val), std::forward<NewT>(new_val)); }))
                        {
                            return TinyDIP::replace(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), std::forward<TargetT>(target_val), std::forward<NewT>(new_val));
                        }
                        else if constexpr ((std::ranges::input_range<DecayedT>) && (requires { std::replace(std::forward<ExecPolicy>(exec_policy), std::declval<DecayedT&>().begin(), std::declval<DecayedT&>().end(), std::forward<TargetT>(target_val), std::forward<NewT>(new_val)); }))
                        {
                            DecayedT result = data;
                            std::replace(std::forward<ExecPolicy>(exec_policy), std::ranges::begin(result), std::ranges::end(result), std::forward<TargetT>(target_val), std::forward<NewT>(new_val));
                            return result;
                        }
                        else
                        {
                            throw std::invalid_argument("Input type does not support execution-policy multi-threaded replacement natively.");
                            return std::any{};
                        }
                    };

                    try
                    {
                        std::any result = dispatch_policy_string(policy_str, exec_policy, exec_default, os);

                        if (!result.has_value()) return;

                        bool handled = false;
                        auto try_save_output = [&]<typename OutT>() -> bool
                        {
                            if (result.type() == typeid(OutT))
                            {
                                image_saver_fun(output_arg, workspace, std::move(std::any_cast<OutT&>(result)));
                                std::print(os, "Saved to {}\n", output_arg);
                                handled = true;
                                return true;
                            }
                            return false;
                        };

                        if (!match_any_type<master_data_types>(try_save_output))
                        {
                            std::print(os, "Error: Output type from processor is unknown or unsupported. Type Name: [{}]\n", result.type().name());
                        }
                    }
                    catch (const std::exception& e)
                    {
                        std::print(os, "Error calculating replace: {}\n", e.what());
                    }
                }
                else
                {
                    std::print(os, "Error: Cannot mathematically compare or assign [{}] with target [{}] and new value [{}].\n", 
                               get_type_name<ElementT>(), get_type_name<TargetDecayedT>(), get_type_name<NewDecayedT>());
                }
            };

            // Safely cascades argument parsing for the new value payload
            auto evaluate_args = [&]<typename TupleT1, typename TargetT>(TargetT&& target_val)
            {
                if (new_val_arg.starts_with('$'))
                {
                    bool found_new = false;
                    auto try_new_var = [&]<typename NewT>() -> bool
                    {
                        if (auto* ptr = workspace.template retrieve<NewT>(new_val_arg.substr(1)))
                        {
                            execute_replace(std::forward<TargetT>(target_val), *ptr);
                            return true;
                        }
                        return false;
                    };
                    found_new = match_any_type<TupleT1>(try_new_var);
                    if (!found_new)
                    {
                        std::print(os, "Error: New value variable not found or unsupported type: {}\n", new_val_arg);
                    }
                }
                else
                {
                    try
                    {
                        ElementT parsed_new = parse_arg<ElementT>(new_val_arg);
                        execute_replace(std::forward<TargetT>(target_val), parsed_new);
                    }
                    catch (const std::exception& e)
                    {
                        std::print(os, "Error parsing new value '{}' as scalar: {}\n", new_val_arg, e.what());
                    }
                }
            };

            // Bootstraps argument parsing with the target payload
            if (target_arg.starts_with('$'))
            {
                bool found_target = false;
                auto try_target_var = [&]<typename TargetT>() -> bool
                {
                    if (auto* ptr = workspace.template retrieve<TargetT>(target_arg.substr(1)))
                    {
                        evaluate_args.template operator()<core_numeric_types>(*ptr);
                        return true;
                    }
                    return false;
                };
                found_target = match_any_type<master_scalar_types>(try_target_var);
                if (!found_target)
                {
                    std::print(os, "Error: Target variable not found or unsupported type: {}\n", target_arg);
                }
            }
            else
            {
                try
                {
                    ElementT parsed_target = parse_arg<ElementT>(target_arg);
                    evaluate_args.template operator()<core_numeric_types>(parsed_target);
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

    registry.register_command("replace", "Replace specific value in an image or container to a new value.", IOSchema{ 0, 3 },
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::replace(workspace, args, os);
        }
    );
}