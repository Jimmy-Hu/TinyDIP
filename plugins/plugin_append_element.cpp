/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for replace */

#include "plugin_api.h"

namespace handlers
{
    //  append_element template function implementation
    template <
        typename ImageLoaderFun = MetaImageIO::Loader
    >
    requires (std::invocable<ImageLoaderFun, const std::string_view, Workspace&>)
    constexpr void append_element(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageLoaderFun&& image_loader_fun = ImageLoaderFun{})
    {
        if (std::ranges::size(args) < 3)
        {
            print_command_usage(os, "append_element");
            return;
        }

        const std::string_view input_arg = args[0];
        const std::string_view elem_arg = args[1];
        const std::string_view output_arg = args[2];

        if (!output_arg.starts_with('$'))
        {
            os << "Error: Output must be a memory variable starting with '$'.\n";
            return;
        }

        os << "Appending element " << elem_arg << " to " << input_arg << "...\n";

        auto process_append = [&]<typename CandidateType>(CandidateType&& candidate)
        {
            using DecayedT = std::remove_cvref_t<CandidateType>;

            // Mathematically verify that the type is a registered container that supports back insertion
            if constexpr (is_vector_v<DecayedT> || is_deque_v<DecayedT> || is_list_v<DecayedT>)
            {
                using ElementT = typename DecayedT::value_type;
                
                if (!elem_arg.starts_with('$'))
                {
                    os << "Error: Element must be a memory variable starting with '$'.\n";
                    return;
                }

                const std::string_view elem_name = elem_arg.substr(1);
                
                // Directly retrieve the perfectly matched complex object from the workspace
                if (const ElementT* element_ptr = workspace.template retrieve<ElementT>(elem_name))
                {
                    DecayedT new_container = candidate; // Copy original container completely independently
                    
                    // Utilizing emplace_back to directly construct the complex object in place
                    // optimizing memory allocation and avoiding unnecessary temporary copies natively.
                    new_container.emplace_back(*element_ptr);
                    
                    workspace.store(output_arg.substr(1), std::move(new_container));
                    os << "Saved updated container to " << output_arg << ".\n";
                }
                else
                {
                    os << "Error: Element variable $" << elem_name << " not found or type mismatch. Expected exact type: [" << get_type_name<ElementT>() << "].\n";
                }
            }
            else
            {
                os << "Error: Input type [" << get_type_name<DecayedT>() << "] is not a supported container type that supports appending.\n";
            }
        };

        // Leverage tuple_cat_t to flawlessly support both scalar containers and newly registered image containers!
        using AllContainerTypes = tuple_cat_t<master_data_types, master_image_container_types>;

        if (!dispatch_data_operation<AllContainerTypes>(input_arg, workspace, image_loader_fun, process_append))
        {
            os << "Error: Memory variable not found or unsupported type.\n";
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

    registry.register_command("append_element", "Append an element to the back of a container using emplace_back.", CombinerSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::append_element(workspace, args, os);
        }
    );
}