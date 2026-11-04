/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for replace */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //   extract_container_element type trait implementation
    template <typename T>
    struct extract_container_element
    {
        using type = TinyDIP::get_deep_scalar_t<T>;
    };

    template <typename ElemT>
    struct extract_container_element<TinyDIP::Image<ElemT>>
    {
        using type = ElemT;
    };

    template <std::ranges::input_range RangeT>
    requires (not TinyDIP::is_Image<RangeT>::value)
    struct extract_container_element<RangeT>
    {
        using type = std::ranges::range_value_t<RangeT>;
    };

    template <typename T>
    using extract_container_element_t = typename extract_container_element<std::remove_cvref_t<T>>::type;

    //  replace template function implementation (Optimized for Chain of Responsibility)
    template <
        typename CheckingDataTypes = master_data_types,
        typename CheckingScalarTypes = core_numeric_types,
        std::ranges::input_range RangeT,
        typename ImageLoaderFun = MetaImageIO::Loader,
        typename ImageSaverFun = MetaImageIO::Saver
    >
    requires (
        (std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string_view> or
         std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string> or
         std::convertible_to<std::ranges::range_value_t<RangeT>, std::string_view> or
         std::convertible_to<std::ranges::range_value_t<RangeT>, std::string>) and
        std::invocable<ImageLoaderFun, const std::string_view, Workspace&> and
        std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::RGB>&&> and
        std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<double>&&> and
        std::invocable<ImageSaverFun, const std::string_view, Workspace&, TinyDIP::Image<TinyDIP::HSV>&&>
    )
    constexpr void replace(
        Workspace& workspace,
        const RangeT& args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{},
        ImageSaverFun&& image_saver_fun = ImageSaverFun{}
    )
    {
        const std::string_view policy_str = extract_policy_string(args);
        const auto filtered_args = args_filter(args);

        if (std::ranges::size(filtered_args) < 4)
        {
            std::print(
                os,
                "Usage: replace [execution_policy] <input_data | $var> <target_value | $var> <new_value | $var> <output_var | $var>\n"
            );
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view target_arg = filtered_args[1];
        const std::string_view new_val_arg = filtered_args[2];
        const std::string_view output_arg = filtered_args[3];

        auto process_input = [&]<typename DataT>(DataT&& data)
        {
            using DecayedT = std::remove_cvref_t<DataT>;
            using ActualElementT = extract_container_element_t<DecayedT>;

            if constexpr (not std::equality_comparable<ActualElementT>)
            {
                throw std::invalid_argument(
                    std::string("Element type [") +
                    std::string(get_type_name<ActualElementT>()) +
                    "] is not equality comparable."
                );
            }
            else
            {
                // Decoupled 1D scalar resolver: avoids O(M^2) nested template instantiations
                auto resolve_scalar_arg = [&](const std::string_view arg_sv) -> ActualElementT
                {
                    if (arg_sv.starts_with('$'))
                    {
                        const std::string_view var_name = arg_sv.substr(1);

                        // Fast path: exact element type match (handles RGB, MultiChannel, Complex, etc. in O(1))
                        if (const auto* exact_ptr = workspace.template retrieve<ActualElementT>(var_name))
                        {
                            return *exact_ptr;
                        }

                        // Fallback path: convertible numeric scalar match in O(M)
                        ActualElementT converted_val{};
                        bool matched = match_any_type<CheckingScalarTypes>(
                            [&]<typename ScalarT>() -> bool
                            {
                                if constexpr (std::convertible_to<ScalarT, ActualElementT>)
                                {
                                    if (const auto* ptr = workspace.template retrieve<ScalarT>(var_name))
                                    {
                                        converted_val = static_cast<ActualElementT>(*ptr);
                                        return true;
                                    }
                                }
                                return false;
                            }
                        );

                        if (matched)
                        {
                            return converted_val;
                        }

                        throw std::runtime_error(
                            std::string("Variable ") + std::string(arg_sv) +
                            " not found or not convertible to [" +
                            std::string(get_type_name<ActualElementT>()) + "]."
                        );
                    }
                    else
                    {
                        return parse_arg<ActualElementT>(arg_sv);
                    }
                };

                const ActualElementT target_val = resolve_scalar_arg(target_arg);
                const ActualElementT new_val = resolve_scalar_arg(new_val_arg);

                std::print(
                    os,
                    "Replacing occurrences of {} with {} in {}",
                    target_arg,
                    new_val_arg,
                    input_arg
                );
                if (not std::ranges::empty(policy_str))
                {
                    std::print(os, " (Policy: {})", policy_str);
                }
                std::print(os, "...\n");

                auto exec_default = [&]() -> std::any
                {
                    if constexpr (
                        TinyDIP::is_Image<DecayedT>::value and
                        requires { TinyDIP::replace(std::forward<DataT>(data), target_val, new_val); }
                    )
                    {
                        return TinyDIP::replace(std::forward<DataT>(data), target_val, new_val);
                    }
                    else if constexpr (
                        std::ranges::input_range<DecayedT> and
                        requires(DecayedT& r) { std::ranges::replace(r, target_val, new_val); }
                    )
                    {
                        DecayedT result = data;
                        std::ranges::replace(result, target_val, new_val);
                        return result;
                    }
                    else
                    {
                        throw std::invalid_argument("Input type does not support sequential replacement.");
                    }
                };

                auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_policy) -> std::any
                    requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                {
                    if constexpr (
                        TinyDIP::is_Image<DecayedT>::value and
                        requires { TinyDIP::replace(std::forward<ExecPolicy>(exec_policy), std::forward<DataT>(data), target_val, new_val); }
                    )
                    {
                        return TinyDIP::replace(
                            std::forward<ExecPolicy>(exec_policy),
                            std::forward<DataT>(data),
                            target_val,
                            new_val
                        );
                    }
                    else if constexpr (
                        std::ranges::input_range<DecayedT> and
                        requires(DecayedT& r)
                        {
                            std::replace(
                                std::forward<ExecPolicy>(exec_policy),
                                std::ranges::begin(r),
                                std::ranges::end(r),
                                target_val,
                                new_val
                            );
                        }
                    )
                    {
                        DecayedT result = data;
                        std::replace(
                            std::forward<ExecPolicy>(exec_policy),
                            std::ranges::begin(result),
                            std::ranges::end(result),
                            target_val,
                            new_val
                        );
                        return result;
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Input type does not support execution-policy multi-threaded replacement."
                        );
                    }
                };

                std::any result = dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                if (result.has_value())
                {
                    // Output type is guaranteed to be DecayedT; no O(N) lookup needed
                    image_saver_fun(output_arg, workspace, std::move(std::any_cast<DecayedT&>(result)));
                    std::print(os, "Saved to {}\n", output_arg);
                }
            }
        };

        if (not dispatch_data_operation<CheckingDataTypes>(input_arg, workspace, image_loader_fun, process_input))
        {
            // Throw std::invalid_argument so CommandRegistry::execute advances to the next handler in the chain
            throw std::invalid_argument(
                std::string("Input '") + std::string(input_arg) +
                "' is not handled by this type slice."
            );
        }
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

// Sliced Container Tuples
using sequence_container_types = tuple_cat_t<
    all_vector_types,
    all_deque_types,
    all_list_types,
    all_array_types
>;