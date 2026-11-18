/* Developed by Jimmy Hu */
/* Isolated Compilation Unit for cast command */

#include <variant>
#include "plugin_api.h"

namespace cast_detail
{
    template <typename T>
    struct multichannel_traits
    {
        static constexpr bool is_multichannel = false;
        using channel_type = void;
        static constexpr std::size_t channel_count = 0;

        template <typename NewChannelT>
        using rebind = void;
    };

    template <typename ChannelT, std::size_t N>
    struct multichannel_traits<TinyDIP::MultiChannel<ChannelT, N>>
    {
        static constexpr bool is_multichannel = true;
        using channel_type = ChannelT;
        static constexpr std::size_t channel_count = N;

        template <typename NewChannelT>
        using rebind = TinyDIP::MultiChannel<NewChannelT, N>;
    };

    template <typename T>
    inline constexpr bool is_color_triplet_struct_v =
        std::same_as<T, TinyDIP::RGB> ||
        std::same_as<T, TinyDIP::RGB_DOUBLE> ||
        std::same_as<T, TinyDIP::HSV>;

    template <typename ContainerT, typename NewElemT>
    struct container_cast_traits
    {
        static constexpr bool supported = is_sequence_container_v<ContainerT>;
        static constexpr bool is_fixed_array = is_std_array_v<ContainerT>;
        static constexpr bool is_random_access = is_vector_v<ContainerT> || is_deque_v<ContainerT> || is_std_array_v<ContainerT>;
        using old_elem_type = std::ranges::range_value_t<ContainerT>;
        using type = TinyDIP::rebind_container_t<ContainerT, NewElemT>;
    };

    template <typename FunT, typename TargetT>
    concept has_template_operator = requires(FunT& func)
    {
        func.template operator()<TargetT>();
    };

    using dispatchable_target_types = tuple_unique_t<std::tuple<
        bool, unsigned char, int, float, double, std::size_t,
        std::complex<float>, std::complex<double>, std::complex<long double>,
        TinyDIP::RGB, TinyDIP::RGB_DOUBLE, TinyDIP::HSV,
        TinyDIP::MultiChannel<bool, 3>,
        TinyDIP::MultiChannel<unsigned char, 3>,
        TinyDIP::MultiChannel<int, 3>,
        TinyDIP::MultiChannel<float, 3>,
        TinyDIP::MultiChannel<double, 3>,
        TinyDIP::MultiChannel<std::size_t, 3>
#ifdef TINYDIP_FULL_BUILD
        , char, std::int8_t, short, unsigned short, unsigned int,
        long, unsigned long, long long, unsigned long long,
        std::ptrdiff_t, long double
#endif
    >>;

    template <typename FunT, typename TupleT>
    struct supports_all_target_types_impl : std::false_type {};

    template <typename FunT, typename... Ts>
    struct supports_all_target_types_impl<FunT, std::tuple<Ts...>>
        : std::bool_constant<(has_template_operator<FunT, Ts> && ...)> {};

    template <typename FunT>
    concept supports_cast_target_types =
        supports_all_target_types_impl<FunT, dispatchable_target_types>::value;

    inline const auto& get_target_type_map()
    {
        static const auto type_map = build_canonical_type_alias_map<dispatchable_target_types>(
            [](auto& add)
            {
                add.template operator()<std::complex<float>>({"complex_float", "cfloat"});
                add.template operator()<std::complex<double>>({"complex_double", "cdouble", "complex"});
                add.template operator()<std::complex<long double>>({"complex_ldouble", "cldouble"});
                add.template operator()<TinyDIP::RGB_DOUBLE>({"rgbd"});
                add.template operator()<TinyDIP::MultiChannel<bool, 3>>({"mc_bool", "multichannel<bool>"});
                add.template operator()<TinyDIP::MultiChannel<unsigned char, 3>>({"mc_uint8", "mc_uchar", "multichannel<uint8>", "multichannel<uint8,3>", "multichannel<unsigned char>"});
                add.template operator()<TinyDIP::MultiChannel<int, 3>>({"mc_int", "multichannel<int>"});
                add.template operator()<TinyDIP::MultiChannel<float, 3>>({"mc_float", "multichannel<float>"});
                add.template operator()<TinyDIP::MultiChannel<double, 3>>({"mc_double", "multichannel<double>"});
                add.template operator()<TinyDIP::MultiChannel<std::size_t, 3>>({"mc_size_t", "multichannel<size_t>", "multichannel<size_t,3>"});
            }
        );
        return type_map;
    }

    template <supports_cast_target_types CallbackFun>
    inline bool dispatch_target_type(const std::string_view raw_type_str, CallbackFun&& callback)
    {
        const auto& type_map = get_target_type_map();
        const std::string key = canonicalize_target_type_key(raw_type_str);
        if (const auto it = type_map.find(key); it != type_map.end())
        {
            std::visit(
                [&]<typename TargetT>(std::type_identity<TargetT>)
                    requires has_template_operator<CallbackFun, TargetT>
                {
                    callback.template operator()<TargetT>();
                },
                it->second
            );
            return true;
        }
        return false;
    }

    inline bool is_recognized_target_type(const std::string_view raw_type_str)
    {
        const auto& type_map = get_target_type_map();
        return type_map.find(canonicalize_target_type_key(raw_type_str)) != type_map.end();
    }

    template <typename OutPixelT, std::size_t N, typename DstChT, typename SrcElemT>
    constexpr OutPixelT cast_pixel_channels(const SrcElemT& px)
    {
        OutPixelT dst{};
        for (std::size_t i = 0; i < N; ++i)
        {
            dst.channels[i] = static_cast<DstChT>(px.channels[i]);
        }
        return dst;
    }

    // Returns a pixel-channel casting functor when TargetT/SrcElemT form a valid MultiChannel conversion,
    // or std::nullptr_t otherwise — shared by both cast_image_default and cast_image_policy.
    template <typename TargetT, typename SrcElemT>
    constexpr auto make_multichannel_pixel_caster()
    {
        if constexpr (multichannel_traits<TargetT>::is_multichannel &&
                      is_color_triplet_struct_v<SrcElemT> &&
                      multichannel_traits<TargetT>::channel_count == 3 &&
                      requires(const decltype(std::declval<SrcElemT>().channels[0])& c)
                      {
                          static_cast<typename multichannel_traits<TargetT>::channel_type>(c);
                      })
        {
            using DstChT = typename multichannel_traits<TargetT>::channel_type;
            return [](const SrcElemT& px) constexpr { return cast_pixel_channels<TargetT, 3, DstChT>(px); };
        }
        else if constexpr (multichannel_traits<TargetT>::is_multichannel &&
                           multichannel_traits<SrcElemT>::is_multichannel &&
                           multichannel_traits<SrcElemT>::channel_count == multichannel_traits<TargetT>::channel_count &&
                           requires(const typename multichannel_traits<SrcElemT>::channel_type& c)
                           {
                               static_cast<typename multichannel_traits<TargetT>::channel_type>(c);
                           })
        {
            using DstChT = typename multichannel_traits<TargetT>::channel_type;
            constexpr std::size_t DstN = multichannel_traits<TargetT>::channel_count;
            return [](const SrcElemT& px) constexpr { return cast_pixel_channels<TargetT, DstN, DstChT>(px); };
        }
        else if constexpr (!multichannel_traits<TargetT>::is_multichannel &&
                           multichannel_traits<SrcElemT>::is_multichannel &&
                           (TinyDIP::check_tuple_element_type<TargetT, core_numeric_types>::value ||
                            TinyDIP::check_tuple_element_type<TargetT, all_complex_types>::value) &&
                           requires(const typename multichannel_traits<SrcElemT>::channel_type& c)
                           {
                               static_cast<TargetT>(c);
                           })
        {
            constexpr std::size_t N = multichannel_traits<SrcElemT>::channel_count;
            using OutPixelT = typename multichannel_traits<SrcElemT>::template rebind<TargetT>;
            return [](const SrcElemT& px) constexpr { return cast_pixel_channels<OutPixelT, N, TargetT>(px); };
        }
        else
        {
            return nullptr;
        }
    }

    // Cast a single TinyDIP::Image<SrcElemT> without an execution policy
    template <typename TargetT, typename SrcElemT>
    constexpr auto cast_image_default(const TinyDIP::Image<SrcElemT>& src_img)
    {
        constexpr auto mc_caster = make_multichannel_pixel_caster<TargetT, SrcElemT>();
        if constexpr (std::same_as<SrcElemT, TargetT>)
        {
            return TinyDIP::Image<TargetT>(src_img);
        }
        else if constexpr (requires { src_img.template cast<TargetT>(); })
        {
            return src_img.template cast<TargetT>();
        }
        else if constexpr (!std::same_as<std::remove_cvref_t<decltype(mc_caster)>, std::nullptr_t>)
        {
            return TinyDIP::pixelwise_transform(mc_caster, src_img);
        }
        else if constexpr (is_color_triplet_struct_v<TargetT> && is_color_triplet_struct_v<SrcElemT>)
        {
            return TinyDIP::convert_image<TargetT>(src_img);
        }
        else if constexpr (is_color_triplet_struct_v<TargetT> &&
                           multichannel_traits<SrcElemT>::is_multichannel &&
                           multichannel_traits<SrcElemT>::channel_count == 3)
        {
            using SrcChT = typename multichannel_traits<SrcElemT>::channel_type;
            if constexpr (requires(const SrcChT& c) { static_cast<double>(c); static_cast<TinyDIP::GrayScale>(c); })
            {
                return TinyDIP::convert_image<TargetT>(src_img);
            }
        }
    }

    template <typename TargetT, typename SrcElemT>
    inline constexpr bool can_cast_image_v =
        !std::same_as<decltype(cast_image_default<TargetT>(std::declval<const TinyDIP::Image<SrcElemT>&>())), void>;

    // Cast a single TinyDIP::Image<SrcElemT> with an execution policy
    template <typename TargetT, typename ExPo, typename SrcElemT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>> && can_cast_image_v<TargetT, SrcElemT>)
    constexpr auto cast_image_policy(ExPo&& exec_policy, const TinyDIP::Image<SrcElemT>& src_img)
    {
        constexpr auto mc_caster = make_multichannel_pixel_caster<TargetT, SrcElemT>();
        if constexpr (std::same_as<SrcElemT, TargetT>)
        {
            return TinyDIP::Image<TargetT>(src_img);
        }
        else if constexpr (requires { src_img.template cast<TargetT>(std::forward<ExPo>(exec_policy)); })
        {
            return src_img.template cast<TargetT>(std::forward<ExPo>(exec_policy));
        }
        else if constexpr (!std::same_as<std::remove_cvref_t<decltype(mc_caster)>, std::nullptr_t>)
        {
            return TinyDIP::pixelwise_transform(std::forward<ExPo>(exec_policy), mc_caster, src_img);
        }
        else
        {
            return cast_image_default<TargetT>(src_img);
        }
    }

    template <typename T>
    struct leaf_image_type
    {
        using type = T;
    };

    template <std::ranges::input_range R>
    requires (!TinyDIP::is_Image<R>::value)
    struct leaf_image_type<R> : leaf_image_type<std::ranges::range_value_t<R>> {};

    template <typename T>
    using leaf_image_type_t = typename leaf_image_type<std::remove_cvref_t<T>>::type;

    using all_castable_image_container_types = tuple_cat_t<
        master_image_container_types,
        master_nested_image_container_types
    >;

    using castable_scalar_types = tuple_unique_t<tuple_cat_t<
        core_numeric_types,
        all_complex_types
    >>;
}

namespace handlers
{
    //  cast template function implementation
    template <typename OStreamT = std::ostream>
    requires(std::derived_from<std::remove_cvref_t<OStreamT>, std::ostream>)
    inline void cast(
        Workspace& workspace,
        std::span<const std::string_view> args,
        OStreamT& os = std::cout)
    {
        const std::string_view policy_str = extract_policy_string(args);
        const auto filtered_args = args_filter(args);

        if (std::ranges::size(filtered_args) < 3)
        {
            print_command_usage(os, "cast");
            return;
        }

        const std::string_view input_arg = filtered_args[0];
        std::string_view output_arg = filtered_args[1];
        std::string combined_type_storage;
        std::string_view type_arg = filtered_args[2];

        if (std::ranges::size(filtered_args) >= 4)
        {
            combined_type_storage = std::string(sanitize_string_view(filtered_args[2]));
            for (std::size_t i = 3; i < std::ranges::size(filtered_args); ++i)
            {
                combined_type_storage.push_back(' ');
                combined_type_storage.append(sanitize_string_view(filtered_args[i]));
            }
            type_arg = combined_type_storage;
        }
        else if (!output_arg.starts_with('$') &&
                 cast_detail::is_recognized_target_type(output_arg) &&
                 !cast_detail::is_recognized_target_type(type_arg))
        {
            std::swap(output_arg, type_arg);
        }

        if (!cast_detail::is_recognized_target_type(type_arg))
        {
            std::print(
                os,
                "Error: Unsupported target type '{}' for 'cast'.\n"
                "Supported types: bool, uint8 (uchar / grayscale), int, float, double, size_t, "
                "complex_float, complex_double, complex_ldouble, rgb, rgb_double, hsv, "
                "mc_bool, mc_uint8, mc_int, mc_float, mc_double, mc_size_t "
                "(or TinyDIP::Image<T> forms such as TinyDIP::Image<float>, TinyDIP::Image<int>).\n",
                sanitize_string_view(type_arg)
            );
            return;
        }

        auto save_image_result = [&]<typename OutImgT>(OutImgT&& out_img)
        {
            using DecayedOutImgT = std::remove_cvref_t<OutImgT>;
            MetaImageIO::Saver{}(output_arg, workspace, std::forward<OutImgT>(out_img));
            std::print(os, "Saved {} to {}\n", get_type_name<DecayedOutImgT>(), output_arg);
        };

        auto save_memory_only_result = [&]<typename OutT>(OutT&& out_val)
        {
            using DecayedOutT = std::remove_cvref_t<OutT>;
            if (!output_arg.starts_with('$'))
            {
                throw std::invalid_argument(
                    "Direct file writing is only supported for Image<RGB>, Image<double>, and Image<HSV>. "
                    "Please specify a workspace variable starting with '$'."
                );
            }
            workspace.store(output_arg.substr(1), std::forward<OutT>(out_val));
            std::print(os, "Saved {} to {}\n", get_type_name<DecayedOutT>(), output_arg);
        };

        auto print_cast_banner = [&]()
        {
            std::print(os, "Casting {} to {}", input_arg, sanitize_string_view(type_arg));
            if (!std::ranges::empty(policy_str))
            {
                std::print(os, " (Policy: {})", policy_str);
            }
            std::print(os, "...\n");
        };

        auto run_on_image = [&]<typename SrcImgT>(const SrcImgT& src_img)
        {
            using SrcElemT = TinyDIP::get_element_type_t<SrcImgT>;
            cast_detail::dispatch_target_type(type_arg, [&]<typename TargetT>()
            {
                if constexpr (cast_detail::can_cast_image_v<TargetT, SrcElemT>)
                {
                    print_cast_banner();
                    auto exec_default = [&]() -> std::any
                    {
                        return cast_detail::cast_image_default<TargetT>(src_img);
                    };
                    auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_pol) -> std::any
                        requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                    {
                        return cast_detail::cast_image_policy<TargetT>(std::forward<ExecPolicy>(exec_pol), src_img);
                    };

                    using OutImgT = decltype(cast_detail::cast_image_default<TargetT>(src_img));
                    std::any result_any = dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                    save_image_result(std::move(std::any_cast<OutImgT&>(result_any)));
                }
                else
                {
                    if constexpr (cast_detail::is_color_triplet_struct_v<SrcElemT> &&
                                  (std::is_arithmetic_v<TargetT> || TinyDIP::is_complex_data_v<TargetT>))
                    {
                        throw std::invalid_argument(
                            std::string("Cannot cast 3-channel image ") + std::string(get_type_name<SrcImgT>()) +
                            " directly to single-channel scalar '" + std::string(get_type_name<TargetT>()) +
                            "'. Use 'rgb_double', 'rgb', 'hsv', 'mc_float' / 'mc_int', or extract a plane with getRplane/getGplane/getBplane first."
                        );
                    }
                    else
                    {
                        throw std::invalid_argument(
                            std::string("Cannot cast ") + std::string(get_type_name<SrcImgT>()) +
                            " to target type '" + std::string(get_type_name<TargetT>()) + "'."
                        );
                    }
                }
            });
        };

        try
        {
            if (input_arg.starts_with('$'))
            {
                const std::string_view var_name = input_arg.substr(1);

                // 1. Check master_image_types
                const bool matched_image = match_any_type<master_image_types>([&]<typename ImgT>() -> bool
                {
                    if (const ImgT* img_ptr = workspace.retrieve<ImgT>(var_name))
                    {
                        run_on_image(*img_ptr);
                        return true;
                    }
                    return false;
                });
                if (matched_image)
                {
                    return;
                }

                // 2. Check 1-level and 2-level image containers uniformly via recursive_transform<Depth>
                const bool matched_img_container = match_any_type<cast_detail::all_castable_image_container_types>([&]<typename ContainerT>() -> bool
                {
                    if (const ContainerT* cont_ptr = workspace.retrieve<ContainerT>(var_name))
                    {
                        constexpr std::size_t Depth = TinyDIP::recursive_depth<ContainerT>();
                        using SliceImgT = cast_detail::leaf_image_type_t<ContainerT>;
                        using SrcElemT = TinyDIP::get_element_type_t<SliceImgT>;

                        cast_detail::dispatch_target_type(type_arg, [&]<typename TargetT>()
                        {
                            if constexpr (cast_detail::can_cast_image_v<TargetT, SrcElemT>)
                            {
                                print_cast_banner();
                                auto exec_default = [&]() -> std::any
                                {
                                    return TinyDIP::recursive_transform<Depth>(
                                        [](const SliceImgT& slice) { return cast_detail::cast_image_default<TargetT>(slice); },
                                        *cont_ptr
                                    );
                                };
                                auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_pol) -> std::any
                                    requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                                {
                                    return TinyDIP::recursive_transform<Depth>(
                                        std::forward<ExecPolicy>(exec_pol),
                                        [&](const SliceImgT& slice) { return cast_detail::cast_image_policy<TargetT>(exec_pol, slice); },
                                        *cont_ptr
                                    );
                                };

                                using OutContainerT = decltype(TinyDIP::recursive_transform<Depth>(
                                    [](const SliceImgT& slice) { return cast_detail::cast_image_default<TargetT>(slice); },
                                    *cont_ptr
                                ));
                                std::any result_any = dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                                save_memory_only_result(std::move(std::any_cast<OutContainerT&>(result_any)));
                            }
                            else
                            {
                                throw std::invalid_argument(
                                    std::string("Cannot cast image container ") + std::string(get_type_name<ContainerT>()) +
                                    " to element type '" + std::string(get_type_name<TargetT>()) + "'."
                                );
                            }
                        });
                        return true;
                    }
                    return false;
                });
                if (matched_img_container)
                {
                    return;
                }

                // 3. Check 1D sequence containers (vector, deque, list, array of core_numeric_types)
                const bool matched_seq_container = match_any_type<sequence_container_types>([&]<typename SeqT>() -> bool
                {
                    if (const SeqT* seq_ptr = workspace.retrieve<SeqT>(var_name))
                    {
                        cast_detail::dispatch_target_type(type_arg, [&]<typename TargetT>()
                        {
                            if constexpr (TinyDIP::check_tuple_element_type<TargetT, core_numeric_types>::value)
                            {
                                using Rebound = cast_detail::container_cast_traits<SeqT, TargetT>;
                                if constexpr (Rebound::supported)
                                {
                                    using OldElemT = typename Rebound::old_elem_type;
                                    using OutSeqT = typename Rebound::type;
                                    if constexpr (requires(const OldElemT& v) { static_cast<TargetT>(v); })
                                    {
                                        print_cast_banner();
                                        auto cast_elem = [](const auto& elem) { return static_cast<TargetT>(elem); };
                                        auto exec_default = [&]() -> std::any
                                        {
                                            return TinyDIP::recursive_transform<1>(cast_elem, *seq_ptr);
                                        };
                                        auto exec_policy = [&]<typename ExecPolicy>(ExecPolicy&& exec_pol) -> std::any
                                            requires std::is_execution_policy_v<std::remove_cvref_t<ExecPolicy>>
                                        {
                                            if constexpr (!Rebound::is_fixed_array &&
                                                          Rebound::is_random_access &&
                                                          !std::same_as<OldElemT, bool> &&
                                                          !std::same_as<TargetT, bool>)
                                            {
                                                return TinyDIP::recursive_transform<1>(
                                                    std::forward<ExecPolicy>(exec_pol),
                                                    cast_elem,
                                                    *seq_ptr
                                                );
                                            }
                                            else
                                            {
                                                return exec_default();
                                            }
                                        };

                                        std::any result_any = dispatch_policy_string(policy_str, exec_policy, exec_default, os);
                                        save_memory_only_result(std::move(std::any_cast<OutSeqT&>(result_any)));
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            std::string("Cannot cast container ") + std::string(get_type_name<SeqT>()) +
                                            " to element type '" + std::string(get_type_name<TargetT>()) + "'."
                                        );
                                    }
                                }
                            }
                            else
                            {
                                throw std::invalid_argument(
                                    std::string("1D sequence containers only support core numeric target types; got '") +
                                    std::string(get_type_name<TargetT>()) + "'."
                                );
                            }
                        });
                        return true;
                    }
                    return false;
                });
                if (matched_seq_container)
                {
                    return;
                }

                // 4. Check scalar workspace variables (core_numeric_types and all_complex_types)
                const bool matched_scalar = match_any_type<cast_detail::castable_scalar_types>([&]<typename ScalarT>() -> bool
                {
                    if (const ScalarT* scalar_ptr = workspace.retrieve<ScalarT>(var_name))
                    {
                        cast_detail::dispatch_target_type(type_arg, [&]<typename TargetT>()
                        {
                            if constexpr ((TinyDIP::check_tuple_element_type<TargetT, core_numeric_types>::value ||
                                           TinyDIP::check_tuple_element_type<TargetT, all_complex_types>::value) &&
                                          requires(const ScalarT& s) { static_cast<TargetT>(s); })
                            {
                                print_cast_banner();
                                TargetT casted_val = static_cast<TargetT>(*scalar_ptr);
                                save_memory_only_result(std::move(casted_val));
                            }
                            else
                            {
                                throw std::invalid_argument(
                                    std::string("Cannot cast scalar ") + std::string(get_type_name<ScalarT>()) +
                                    " to target type '" + std::string(get_type_name<TargetT>()) + "'."
                                );
                            }
                        });
                        return true;
                    }
                    return false;
                });
                if (matched_scalar)
                {
                    return;
                }

                std::print(os, "Error: Memory variable '{}' not found or holds an unsupported data type for 'cast'.\n", input_arg);
            }
            else
            {
                const bool loaded = dispatch_data_operation<saveable_image_types>(
                    input_arg,
                    workspace,
                    MetaImageIO::Loader{},
                    [&]<typename DiskImgT>(DiskImgT&& disk_img)
                    {
                        run_on_image(disk_img);
                    }
                );
                if (!loaded)
                {
                    std::print(os, "Error: Could not load '{}' for 'cast'.\n", input_arg);
                }
            }
        }
        catch (const std::exception& e)
        {
            std::print(os, "Error during cast: {}\n", e.what());
        }
    }
}

extern "C" TINYDIP_PLUGIN_EXPORT void register_plugin_commands(CommandRegistry& registry)
{
    register_plugin_command(
        registry,
        "cast",
        "Cast an image, container, or scalar to a specified target data type.",
        TransformerSchema,
        [](Workspace& workspace, std::span<const std::string_view> args, std::ostream& os)
        {
            handlers::cast(workspace, args, os);
        }
    );
}
