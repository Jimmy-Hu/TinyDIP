/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for rand command */

#include "../main.h"
#include "plugin_api.h"

namespace handlers
{
    //  rand_generator template function implementation
    template <
        std::invocable<const std::string_view, Workspace&, TinyDIP::Image<double>&&> ImageSaverFun = MetaImageIO::Saver
    >
    constexpr void rand_generator(
        Workspace& workspace,
        std::span<const std::string_view> args,
        std::ostream& os = std::cout,
        ImageSaverFun&& image_saver_fun = ImageSaverFun{})
    {
        auto dispatch_generation = [&]
        (std::uniform_random_bit_generator auto&& urbg, const std::string_view& out_path, std::span<const std::size_t> sz)
        {
            std::uniform_real_distribution<double> dist{};
            using UrbgType = std::remove_cvref_t<decltype(urbg)>;
            using DistType = decltype(dist);

            RandomGenerator<UrbgType, DistType> gen{urbg, dist};

            //  Calling the dynamic range-based generate overload directly from TinyDIP.
            auto output_img = TinyDIP::generate(gen, sz);

            // Dynamically save image via the injected saver abstraction
            image_saver_fun(out_path, workspace, std::move(output_img));
            os << "Saved to " << out_path << "\n";
        };

        std::map<std::string_view, std::function<void(const std::string_view&, std::span<const std::size_t>)>> urbg_mapping = {
            {"knuth_b",       [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::knuth_b{std::random_device{}()}, out_path, sz); }
            },
            {"minstd_rand",   [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::minstd_rand{std::random_device{}()}, out_path, sz); }
            },
            {"minstd_rand0",  [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::minstd_rand0{std::random_device{}()}, out_path, sz); }
            },
            {"mt19937",       [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::mt19937{std::random_device{}()}, out_path, sz); }
            },
            {"mt19937_64",    [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::mt19937_64{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux24",      [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::ranlux24{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux24_base", [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::ranlux24_base{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux48",      [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::ranlux48{std::random_device{}()}, out_path, sz); }
            },
            {"ranlux48_base", [&]
                (const std::string_view& out_path, std::span<const std::size_t> sz)
                { dispatch_generation(std::ranlux48_base{std::random_device{}()}, out_path, sz); }
            }
        };

        auto print_available_urbgs = [&]()
        {
            os << "Available URBGs: ";
            for (auto it = std::ranges::begin(urbg_mapping); it != std::ranges::end(urbg_mapping); ++it)
            {
                os << it->first;
                if (std::next(it) != std::ranges::end(urbg_mapping))
                {
                    os << ", ";
                }
            }
            os << '\n';
        };

        if (std::ranges::size(args) < 3)
        {
            os << "Usage: rand <urbg_type> <output_bmp | $var> <dim1> [dim2] [dim3] ...\n";
            print_available_urbgs();
            return;
        }

        const std::string_view urbg_type = args[0];
        const std::string_view output_arg = args[1];
        
        std::vector<std::size_t> sizes;
        sizes.reserve(std::ranges::size(args) - 2);
        
        for (std::size_t i = 2; i < std::ranges::size(args); ++i)
        {
            sizes.emplace_back(parse_arg<std::size_t>(args[i]));
        }

        os << "Generating random image with dimensions: ";
        for (const auto& size : sizes)
        {
            os << size << " ";
        }
        os << "using URBG '" << urbg_type << "'...\n";

        if (auto it = urbg_mapping.find(urbg_type); it != std::ranges::end(urbg_mapping))
        {
            it->second(output_arg, sizes);
        }
        else
        {
            os << "Error: Unknown URBG type '" << urbg_type << "'.\n";
            print_available_urbgs();
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

    registry.register_command("rand", "Generate random multi-dimensional image with specified URBG.", GeneratorSchema, 
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::rand_generator(workspace, args, os);
        }
    );
}