/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for histogram command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  histogram template function implementation
    template <
        std::ranges::input_range RangeT,
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires ((std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string_view> or
               std::same_as<std::remove_cvref_t<std::ranges::range_value_t<RangeT>>, std::string> or
               std::convertible_to<std::ranges::range_value_t<RangeT>, std::string_view> or
               std::convertible_to<std::ranges::range_value_t<RangeT>, std::string>) and
              std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void histogram(
        Workspace& workspace,
        const RangeT& args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        const std::string_view policy_str = extract_policy_string(args);
        const auto filtered_args = args_filter(args);

        if (std::ranges::empty(filtered_args))
        {
            std::print(os, "Usage: histogram [execution_policy] <input_image | $var> [output_var | $var]\n");
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        const std::string_view output_arg = std::ranges::size(filtered_args) > 1 ? filtered_args[1] : std::string_view{};

        std::print(os, "Generating histogram for {}", input_arg);
        if (!std::ranges::empty(policy_str))
        {
            std::print(os, " (Policy: {})", policy_str);
        }
        std::print(os, "...\n");

        auto process_input = [&]<typename DataT>(DataT&& input_img)
        {
            using DecayedT = std::remove_cvref_t<DataT>;
            using ElementT = TinyDIP::get_element_type_t<DecayedT>;

            if constexpr (TinyDIP::is_Image<DecayedT>::value)
            {
                if constexpr (std::totally_ordered<ElementT>)
                {
                    try
                    {
                        TinyDIP::Histogram<ElementT> hist(input_img);

                        if (!std::ranges::empty(output_arg))
                        {
                            if (output_arg.starts_with('$'))
                            {
                                const std::size_t total_count = hist.getCountSum();
                                const std::size_t bin_count = hist.size();
                                workspace.store(output_arg.substr(1), std::move(hist));
                                std::print(os, "Saved histogram (bins: {}, total count: {}) to {}\n", bin_count, total_count, output_arg);
                            }
                            else
                            {
                                std::print(os, "Error: Output must be a memory variable starting with '$'.\n");
                            }
                        }
                        else
                        {
                            std::size_t non_zero_bins = 0;
                            if constexpr (TinyDIP::Histogram<ElementT>::is_dense_v)
                            {
                                for (std::size_t i = 0; i < hist.size(); ++i)
                                {
                                    const auto c = hist.getCount(static_cast<ElementT>(i));
                                    if (c > 0)
                                    {
                                        std::print(os, "  [{}] : {}\n", i, c);
                                        ++non_zero_bins;
                                    }
                                }
                            }
                            else
                            {
                                for (const auto& [key, c] : hist)
                                {
                                    if (c > 0)
                                    {
                                        if constexpr (sizeof(ElementT) == 1 && std::is_integral_v<ElementT>)
                                        {
                                            std::print(os, "  [{}] : {}\n", +key, c);
                                        }
                                        else
                                        {
                                            std::print(os, "  [{}] : {}\n", key, c);
                                        }
                                        ++non_zero_bins;
                                    }
                                }
                            }
                            std::print(os, "Histogram summary: {} active bin(s), total count = {}\n", non_zero_bins, hist.getCountSum());
                        }
                    }
                    catch (const std::exception& e)
                    {
                        std::print(os, "Error generating histogram: {}\n", e.what());
                    }
                }
                else
                {
                    std::print(os, "Error: Image element type [{}] does not support ordering (<) required for Histogram. Please use a single-channel numeric image.\n", get_type_name<ElementT>());
                }
            }
            else
            {
                std::print(os, "Error: Histogram operation requires an Image input.\n");
            }
        };

        if (!dispatch_data_operation<master_image_types>(input_arg, workspace, image_loader_fun, process_input))
        {
            std::print(os, "Error: Memory variable not found or unsupported image type for input: {}\n", input_arg);
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

    registry.register_command("histogram", "Generate the histogram of a single-channel image.", TransformerSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::histogram(workspace, args, os);
        }
    );
}
