/* Developed by Jimmy Hu */
/* Shared Implementation Header for Trigonometric CLI Plugins (sin, cos, tan) */

#ifndef TINYDIP_PLUGIN_TRIG_COMMON_H
#define TINYDIP_PLUGIN_TRIG_COMMON_H

#include "plugin_api.h"

namespace handlers
{
    namespace detail_trig
    {
        using trig_scalar_types = tuple_unique_t<leaf_scalar_types>;

        using trig_input_types = tuple_unique_t<tuple_cat_t<
            master_data_types,
            master_image_container_types,
            trig_scalar_types
        >>;

        using trig_scalar_output_types = tuple_unique_t<tuple_cat_t<
            core_floating_point_types,
            tuple_map_t<multichannel_t, core_floating_point_types>,
            all_complex_types,
            all_complex_multichannel_types
        >>;

        using trig_image_output_types = tuple_unique_t<tuple_cat_t<
            tuple_map_t<image_t, core_floating_point_types>,
            std::tuple<TinyDIP::Image<TinyDIP::RGB_DOUBLE>, TinyDIP::Image<TinyDIP::HSV>>,
            tuple_map_t<image_t, tuple_map_t<multichannel_t, core_floating_point_types>>,
            tuple_map_t<image_t, all_complex_types>,
            tuple_map_t<image_t, all_complex_multichannel_types>
        >>;

        using trig_container_output_types = tuple_unique_t<tuple_cat_t<
            tuple_map_t<std::vector, core_floating_point_types>,
            tuple_map_t<std::deque, core_floating_point_types>,
            tuple_map_t<std::list, core_floating_point_types>,
            generate_arrays_t<core_floating_point_types, 3, 4>,
            tuple_map_t<std::vector, trig_image_output_types>,
            tuple_map_t<std::deque, trig_image_output_types>,
            tuple_map_t<std::list, trig_image_output_types>
        >>;

        using trig_output_types = tuple_unique_t<tuple_cat_t<
            trig_image_output_types,
            trig_container_output_types,
            trig_scalar_output_types
        >>;

        template <
            typename TrigOp,
            string_token_range RangeT = std::span<const std::string_view>,
            typename OStreamT = std::ostream
        >
        requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
        constexpr void execute_trig_command(
            Workspace& workspace,
            const RangeT& args,
            const std::string_view cmd_name,
            OStreamT& os = std::cout)
        {
            // Support direct numeric scalar literals on the CLI (e.g., sin 1.57079632679 $out)
            {
                const std::string_view policy_str = extract_policy_string(args);
                const auto filtered = args_filter(args);

                if (filtered.size() >= 2 && !filtered[0].starts_with('$'))
                {
                    double literal_val = 0.0;
                    if (try_parse_non_file_scalar(filtered[0], literal_val))
                    {
                        const std::string_view output_arg = filtered[1];
                        const auto out_var = require_workspace_var_name(
                            os,
                            output_arg,
                            std::string("Error: Output for scalar ") + std::string(cmd_name) + " must be a memory variable starting with '$'.\n"
                        );
                        if (!out_var)
                        {
                            return;
                        }

                        print_policy_banner(
                            os,
                            std::string("Calculating ") + std::string(cmd_name) + " of " + std::string(filtered[0]),
                            policy_str
                        );

                        std::any result_any = dispatch_callable_with_policy(
                            policy_str,
                            os,
                            "Scalar trig operation failed.",
                            [](auto&&... call_args)
                                requires requires { TrigOp::apply(std::forward<decltype(call_args)>(call_args)...); }
                            {
                                return TrigOp::apply(std::forward<decltype(call_args)>(call_args)...);
                            },
                            literal_val
                        );
                        workspace.store(*out_var, std::any_cast<double>(result_any));
                        os << "Saved to " << output_arg << "\n";
                        return;
                    }
                }
            }

            const std::string op_label = std::string(cmd_name) + " operation";
            const std::string banner_prefix = std::string("Calculating ") + std::string(cmd_name) + " of ";
            execute_unary_transform_command<TrigOp, true, trig_input_types, trig_output_types>(
                workspace,
                args,
                os,
                cmd_name,
                op_label,
                banner_prefix
            );
        }
    }
}

#endif // TINYDIP_PLUGIN_TRIG_COMMON_H
