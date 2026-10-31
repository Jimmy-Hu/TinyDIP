/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for im2uint8 command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  im2uint8 function implementation
    constexpr void im2uint8(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2, master_data_types>(
                "im2uint8 [execution_policy] <input_data | $var> <output_var | $var>",
                [](const auto& filtered_args, const std::string_view policy_str, std::ostream& os)
                {
                    os << "Converting " << filtered_args[0] << " to uint8";
                    if (!std::ranges::empty(policy_str))
                    {
                        os << " (Policy: " << policy_str << ")";
                    }
                    os << "...\n";

                    return[policy_str, &os]<typename DataT>(DataT && data) -> std::any
                    {
                        using DecayedDataT = std::remove_cvref_t<DataT>;

                        auto exec_default = [&]() -> std::any
                            {
                                if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                                {
                                    if constexpr (requires { TinyDIP::im2uint8(std::forward<DataT>(data)); })
                                    {
                                        return TinyDIP::im2uint8(std::forward<DataT>(data));
                                    }
                                    else
                                    {
                                        throw std::invalid_argument("Input image type does not support im2uint8 conversion.");
                                        return std::any{};
                                    }
                                }
                                else if constexpr (std::ranges::input_range<DecayedDataT>)
                                {
                                    if constexpr (requires { TinyDIP::im2uint8(*std::ranges::begin(data)); })
                                    {
                                        return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                            [](auto&& element)
                                            {
                                                return TinyDIP::im2uint8(std::forward<decltype(element)>(element));
                                            },
                                            std::forward<DataT>(data)
                                        );
                                    }
                                    else
                                    {
                                        throw std::invalid_argument("Input container type does not support im2uint8 conversion.");
                                        return std::any{};
                                    }
                                }
                                else
                                {
                                    throw std::invalid_argument("Input data type does not support im2uint8 conversion.");
                                    return std::any{};
                                }
                            };

                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy && exec_policy) -> std::any
                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                        {
                            if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                            {
                                if constexpr (requires { TinyDIP::im2uint8(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                                {
                                    return TinyDIP::im2uint8(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
                                }
                                else
                                {
                                    if (!std::ranges::empty(policy_str))
                                    {
                                        os << "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n";
                                    }
                                    return exec_default();
                                }
                            }
                            else if constexpr (std::ranges::input_range<DecayedDataT>)
                            {
                                if constexpr (requires { TinyDIP::im2uint8(*std::ranges::begin(data)); })
                                {
                                    return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                        std::forward<ExecPolicy>(exec_policy),
                                        [](auto&& element)
                                        {
                                            return TinyDIP::im2uint8(std::forward<decltype(element)>(element));
                                        },
                                        std::forward<DataT>(data)
                                    );
                                }
                                else
                                {
                                    if (!std::ranges::empty(policy_str))
                                    {
                                        os << "Warning: Execution policy requested but not supported for this data type/operation. Falling back to default.\n";
                                    }
                                    return exec_default();
                                }
                            }
                            else
                            {
                                if (!std::ranges::empty(policy_str))
                                {
                                    os << "Warning: Execution policy requested but not supported for this data type/operation. Falling back to default.\n";
                                }
                                return exec_default();
                            }
                        };

                        return dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    };
                }
            );
        transform_handler(workspace, args, os);
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

    registry.register_command("im2uint8", "Convert an image or container to 8-bit unsigned integers.", TransformerSchema,
        handlers::im2uint8
    );
}