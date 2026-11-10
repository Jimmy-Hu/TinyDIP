/* Developed by Jimmy Hu */
/* Isolated Header for abs Operation */

#ifndef TINYDIP_PLUGIN_ABS_H
#define TINYDIP_PLUGIN_ABS_H

#include "plugin_api.h"

namespace handlers
{
    //  abs template function implementation
    template <
        typename CheckingTypes = master_data_types,
        typename OutputCheckingTypes = CheckingTypes,
        std::ranges::input_range RangeT = std::span<const std::string_view>
    >
    requires (std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string_view> or
              std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string> or
              std::convertible_to<std::ranges::range_value_t<RangeT>, std::string_view> or
              std::convertible_to<std::ranges::range_value_t<RangeT>, std::string>)
    constexpr auto abs(
        Workspace& workspace,
        const RangeT& args,
        std::ostream& os = std::cout
    )
    {
        auto transform_handler = make_meta_transform_handler<2, CheckingTypes, OutputCheckingTypes>(
            "abs",
            [](std::span<const std::string_view> filtered_args, const std::string_view policy_str, std::ostream& os)
            {
                std::print(os, "Calculating abs of {}", filtered_args[0]);
                if (not std::ranges::empty(policy_str))
                {
                    std::print(os, " (Policy: {})", policy_str);
                }
                std::print(os, "...\n");

                return [policy_str, &os]<typename DataT>(DataT&& data) -> std::any
                {
                    using DecayedDataT = std::remove_cvref_t<DataT>;

                    auto exec_default = [&]() -> std::any
                    {
                        if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            if constexpr (requires { TinyDIP::abs(std::forward<DataT>(data)); })
                            {
                                return TinyDIP::abs(std::forward<DataT>(data));
                            }
                            else
                            {
                                throw std::invalid_argument("Input image type does not support abs operation.");
                            }
                        }
                        else if constexpr (std::ranges::input_range<DecayedDataT>)
                        {
                            return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                [](auto&& element)
                                {
                                    return TinyDIP::generic_abs(std::forward<decltype(element)>(element));
                                },
                                std::forward<DataT>(data)
                            );
                        }
                        else
                        {
                            return TinyDIP::generic_abs(std::forward<DataT>(data));
                        }

                        return std::any{};
                    };

                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        if constexpr (TinyDIP::is_Image<DecayedDataT>::value)
                        {
                            if constexpr (requires { TinyDIP::abs(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data)); })
                            {
                                return TinyDIP::abs(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data));
                            }
                            else
                            {
                                if (not std::ranges::empty(policy_str))
                                {
                                    std::print(os, "Warning: Execution policy requested but not supported for this image type/operation. Falling back to default.\n");
                                }
                                return exec_default();
                            }
                        }
                        else if constexpr (std::ranges::input_range<DecayedDataT>)
                        {
                            return TinyDIP::recursive_transform<TinyDIP::recursive_depth<DecayedDataT>()>(
                                std::forward<ExecPolicy>(exec_policy),
                                [](auto&& element)
                                {
                                    return TinyDIP::generic_abs(std::forward<decltype(element)>(element));
                                },
                                std::forward<DataT>(data)
                            );
                        }
                        else
                        {
                            if (not std::ranges::empty(policy_str))
                            {
                                std::print(os, "Warning: Execution policy requested but not supported for this data type/operation. Falling back to default.\n");
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

// abs(Image<complex<T>>) produces Image<T> or Image<MultiChannel<T, 3>>
using complex_abs_output_types = tuple_cat_t<
    image_types,
    custom_and_multichannel_image_types
>;

// Sliced Container Tuples
using sequence_container_types = tuple_cat_t<
    all_vector_types,
    all_deque_types,
    all_list_types,
    all_array_types
>;

#endif // TINYDIP_PLUGIN_ABS_H