/* Developed by Jimmy Hu */

#ifndef TINYDIP_BASIC_FUNCTIONS_H
#define TINYDIP_BASIC_FUNCTIONS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <complex>
#include <concepts>
#include <coroutine>
#include <deque>
#include <execution>
#include <exception>
#ifdef __cpp_lib_format
#include <format>
#endif
#include <functional>
#include <future>
#if __cplusplus >= 202302L || _HAS_CXX23 
#include <generator>
#endif
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <version>
#include "base_types.h"
//#define USE_BOOST_MULTIDIMENSIONAL_ARRAY
#ifdef USE_BOOST_MULTIDIMENSIONAL_ARRAY
#include <boost/multi_array.hpp>
#include <boost/multi_array/algorithm.hpp>
#include <boost/multi_array/base.hpp>
#include <boost/multi_array/collection_concept.hpp>
#endif

namespace TinyDIP
{
    template<typename T>
    concept is_back_inserterable = requires(T x)
    {
        std::back_inserter(x);
    };

    template<typename T>
    concept is_inserterable = requires(T x)
    {
        std::inserter(x, std::ranges::end(x));
    };

    template<typename T>
    concept is_minusable = requires(T x) { x - x; };

    template<typename T1, typename T2>
    concept is_minusable2 = requires(T1 x1, T2 x2) { x1 - x2; };

    template<typename T>
    concept is_multiplicable = requires(T x)
    {
        x * x;
    };

    template<typename T>
    concept is_divisible = requires(T x)
    {
        x / x;
    };

    #ifdef USE_BOOST_MULTIDIMENSIONAL_ARRAY
    template<typename T>
    concept is_multi_array = requires(T x)
    {
        x.num_dimensions();
        x.shape();
        boost::multi_array(x);
    };

    template<typename T1, typename T2>
    concept is_multi_array_dimensionality_equal =
    is_multi_array<T1> && is_multi_array<T2> && requires(T1 x, T2 y)
    {
        T1::dimensionality == T2::dimensionality;
    };
    #endif

    template<typename T1, typename T2>
    concept is_std_powable = requires(T1 x1, T2 x2)
    {
        std::pow(x1, x2);
    };

    //  Reference: https://stackoverflow.com/a/64287611/6667035
    template <typename T>
    struct is_complex : std::false_type {};

    template <typename T>
    struct is_complex<std::complex<T>> : std::true_type {};

    //  Reference: https://stackoverflow.com/a/48458312/6667035
    template <typename>
    struct is_tuple : std::false_type {};

    template <typename ...T>
    struct is_tuple<std::tuple<T...>> : std::true_type {};

    //  is_MultiChannel struct implementation
    template <typename>
    struct is_MultiChannel : std::false_type {};

    template <std::size_t N, typename T>
    struct is_MultiChannel<MultiChannel<T, N>> : std::true_type {};

    template <typename, typename>
    struct check_tuple_element_type {};

    template <typename TargetType, typename ...ElementT>
    struct check_tuple_element_type<TargetType, std::tuple<ElementT...>>
        : std::bool_constant<(std::is_same_v<ElementT, TargetType> || ...)>
    {
    };

    //  recursive_unwrap_type_t struct implementation
    template<std::size_t, typename, typename...>
    struct recursive_unwrap_type { };

    template<class T>
    struct recursive_unwrap_type<0, T>
    {
        using type = T;
    };

    template<class...Ts1, template<class...>class Container1, typename... Ts>
    struct recursive_unwrap_type<1, Container1<Ts1...>, Ts...>
    {
        using type = std::ranges::range_value_t<Container1<Ts1...>>;
    };

    template<std::size_t unwrap_level, class...Ts1, template<class...>class Container1, typename... Ts>
    requires (  std::ranges::input_range<Container1<Ts1...>> &&
                requires { typename recursive_unwrap_type<
                                        unwrap_level - 1,
                                        std::ranges::range_value_t<Container1<Ts1...>>,
                                        std::ranges::range_value_t<Ts>...>::type; })                //  The rest arguments are ranges
    struct recursive_unwrap_type<unwrap_level, Container1<Ts1...>, Ts...>
    {
        using type = typename recursive_unwrap_type<
            unwrap_level - 1,
            std::ranges::range_value_t<Container1<Ts1...>>
            >::type;
    };

    template<std::size_t unwrap_level, typename T1, typename... Ts>
    using recursive_unwrap_type_t = typename recursive_unwrap_type<unwrap_level, T1, Ts...>::type;
    
    //  recursive_invoke_result_t implementation
    template<std::size_t, typename, typename>
    struct recursive_invoke_result { };

    template<typename T, typename F>
    struct recursive_invoke_result<0, F, T> { using type = std::invoke_result_t<F, T>; };

    template<std::size_t unwrap_level, std::copy_constructible F, template<typename...> typename Container, typename... Ts>
    requires (std::ranges::input_range<Container<Ts...>> &&
            requires { typename recursive_invoke_result<unwrap_level - 1, F, std::ranges::range_value_t<Container<Ts...>>>::type; })
    struct recursive_invoke_result<unwrap_level, F, Container<Ts...>>
    {
        using type = Container<typename recursive_invoke_result<unwrap_level - 1, F, std::ranges::range_value_t<Container<Ts...>>>::type>;
    };

    template<std::size_t unwrap_level, std::copy_constructible F, typename T>
    using recursive_invoke_result_t = typename recursive_invoke_result<unwrap_level, F, T>::type;

    //  recursive_variadic_invoke_result_t implementation
    template<std::size_t, typename, typename, typename...>
    struct recursive_variadic_invoke_result { };

    template<std::copy_constructible F, class...Ts1, template<class...>class Container1, typename... Ts>
    struct recursive_variadic_invoke_result<1, F, Container1<Ts1...>, Ts...>
    {
        using type = Container1<std::invoke_result_t<F,
            std::ranges::range_value_t<Container1<Ts1...>>,
            std::ranges::range_value_t<Ts>...>>;
    };

    template<std::size_t unwrap_level, std::copy_constructible F, class...Ts1, template<class...>class Container1, typename... Ts>
    requires (  std::ranges::input_range<Container1<Ts1...>> &&
                requires { typename recursive_variadic_invoke_result<
                                        unwrap_level - 1,
                                        F,
                                        std::ranges::range_value_t<Container1<Ts1...>>,
                                        std::ranges::range_value_t<Ts>...>::type; })                //  The rest arguments are ranges
    struct recursive_variadic_invoke_result<unwrap_level, F, Container1<Ts1...>, Ts...>
    {
        using type = Container1<
            typename recursive_variadic_invoke_result<
            unwrap_level - 1,
            F,
            std::ranges::range_value_t<Container1<Ts1...>>,
            std::ranges::range_value_t<Ts>...
            >::type>;
    };

    template<std::size_t unwrap_level, std::copy_constructible F, typename T1, typename... Ts>
    using recursive_variadic_invoke_result_t = typename recursive_variadic_invoke_result<unwrap_level, F, T1, Ts...>::type;

    //  recursive_array_invoke_result struct implementation
    template<std::size_t, typename, typename, typename...>
    struct recursive_array_invoke_result { };

    template<   typename F, 
                template<class, std::size_t> class Container,
                typename T,
                std::size_t N,
                typename... Ts>
    struct recursive_array_invoke_result<1, F, Container<T, N>, Ts...>
    {
        using type = Container<
            std::invoke_result_t<F, std::ranges::range_value_t<Container<T, N>>,
            std::ranges::range_value_t<Ts>...>,
            N>;
    };

    template<   std::size_t unwrap_level,
                typename F, 
                template<class, std::size_t> class Container,
                typename T,
                std::size_t N,
                typename... Ts>
    requires (  std::ranges::input_range<Container<T, N>> &&
                requires { typename recursive_array_invoke_result<
                                        unwrap_level - 1,
                                        F,
                                        std::ranges::range_value_t<Container<T, N>>,
                                        std::ranges::range_value_t<Ts>...>::type; })                //  The rest arguments are ranges
    struct recursive_array_invoke_result<unwrap_level, F, Container<T, N>, Ts...>
    {
        using type = Container<
            typename recursive_array_invoke_result<
            unwrap_level - 1,
            F,
            std::ranges::range_value_t<Container<T, N>>,
            std::ranges::range_value_t<Ts>...
            >::type, N>;
    };

    template<   std::size_t unwrap_level,
                typename F,
                template<class, std::size_t> class Container,
                typename T,
                std::size_t N,
                typename... Ts>
    using recursive_array_invoke_result_t = typename recursive_array_invoke_result<unwrap_level, F, Container<T, N>, Ts...>::type;

    //  Reference: https://stackoverflow.com/a/58067611/6667035
    template <typename T>
    concept arithmetic = std::is_arithmetic_v<T> or is_complex<T>::value;

    constexpr bool is_integer()
    {
        return false;
    }

    //  Reference: https://codereview.stackexchange.com/q/288432/231235
    template<std::floating_point T>
    constexpr bool is_integer(T input)
    {
        T integer_part;
        return std::modf(input, &integer_part) == 0;
    }

    template<std::integral T>
    constexpr bool is_integer(T input)
    {
        return true;
    }

    #ifdef USE_BOOST_MULTIDIMENSIONAL_ARRAY
    //  Multiplication
    template<is_multiplicable T1, is_multiplicable T2>
    constexpr auto element_wise_multiplication(const T1& input1, const T2& input2)
    {
        return input1 * input2;
    }

    template<is_multi_array T1, is_multi_array T2>
    requires (is_multi_array_dimensionality_equal<T1, T2>)
    constexpr auto element_wise_multiplication(const T1& input1, const T2& input2)
    {
        if (input1.num_dimensions() != input2.num_dimensions())     //  Dimensions are different, unable to perform element-wise add operation
            throw std::logic_error("Array dimensions are different");
        if (*input1.shape() != *input2.shape())                     //  Shapes are different, unable to perform element-wise add operation
            throw std::logic_error("Array shapes are different");
        boost::multi_array output(input1);                          //  drawback to be improved: avoiding copying whole input1 array into output, but with appropriate memory allocation
        for (decltype(+input1.shape()[0]) i = 0; i < input1.shape()[0]; ++i)
            output[i] = element_wise_multiplication(input1[i], input2[i]);
        return output;
    }

    //  Division
    template<is_divisible T1, is_divisible T2>
    constexpr auto element_wise_division(const T1& input1, const T2& input2)
    {
        if (input2 == 0)
            throw std::logic_error("Divide by zero exception");     //  Handle the case of dividing by zero exception
        return input1 / input2;
    }

    template<is_multi_array T1, is_multi_array T2>
    requires (is_multi_array_dimensionality_equal<T1, T2>)
    constexpr auto element_wise_division(const T1& input1, const T2& input2)
    {
        if (input1.num_dimensions() != input2.num_dimensions())     //  Dimensions are different, unable to perform element-wise add operation
            throw std::logic_error("Array dimensions are different");
        if (*input1.shape() != *input2.shape())                     //  Shapes are different, unable to perform element-wise add operation
            throw std::logic_error("Array shapes are different");
        boost::multi_array output(input1);                          //  drawback to be improved: avoiding copying whole input1 array into output, but with appropriate memory allocation
        for (decltype(+input1.shape()[0]) i = 0; i < input1.shape()[0]; ++i)
            output[i] = element_wise_division(input1[i], input2[i]);
        return output;
    }
    #endif
    
    //  print_tuple template function implementation
    //  Copy from https://stackoverflow.com/a/41171552
    template<class TupType, std::size_t... I>
    void print_tuple(const TupType& _tup, std::index_sequence<I...>)
    {
        std::cout << "(";
        #ifdef __cpp_lib_format
        (..., (std::cout << (I == 0 ? "" : ", ") << std::format("{}", std::get<I>(_tup))));
        #else
        (..., (std::cout << (I == 0 ? "" : ", ") << std::get<I>(_tup)));
        #endif
        std::cout << ")\n";
    }

    //  print_tuple template function implementation
    template<class... T>
    void print_tuple(const std::tuple<T...>& _tup)
    {
        print_tuple(_tup, std::make_index_sequence<sizeof...(T)>());
    }

    //  recursive_depth template function implementation
    template<typename T>
    constexpr std::size_t recursive_depth()
    {
        return std::size_t{0};
    }

    template<std::ranges::input_range Range>
    constexpr std::size_t recursive_depth()
    {
        return recursive_depth<std::ranges::range_value_t<Range>>() + std::size_t{1};
    }

    //  recursive_depth template function implementation with target type
    template<typename T_Base, typename T>
    constexpr std::size_t recursive_depth()
    {
        return std::size_t{0};
    }

    template<typename T_Base, std::ranges::input_range Range>
    requires (!std::same_as<Range, T_Base>)
    constexpr std::size_t recursive_depth()
    {
        return recursive_depth<T_Base, std::ranges::range_value_t<Range>>() + std::size_t{1};
    }

    //  is_recursive_invocable template function implementation
    template<std::size_t unwrap_level, class F, class... T>
    requires(unwrap_level <= recursive_depth<T...>())
    static constexpr bool is_recursive_invocable()
    {
        if constexpr (unwrap_level == 0) {
            return std::invocable<F, T...>;
        } else {
            return is_recursive_invocable<
                        unwrap_level - 1,
                        F,
                        std::ranges::range_value_t<T>...>();
        }
    }

    //  recursive_invocable concept
    template<std::size_t unwrap_level, class F, class... T>
    concept recursive_invocable =
            is_recursive_invocable<unwrap_level, F, T...>();

    //  is_recursive_project_invocable template function implementation
    template<std::size_t unwrap_level, class Proj, class F, class... T>
    requires(unwrap_level <= recursive_depth<T...>() &&
            recursive_invocable<unwrap_level, Proj, T...>)
    static constexpr bool is_recursive_project_invocable()
    {
        if constexpr (unwrap_level == 0) {
            return std::invocable<F, std::invoke_result_t<Proj, T...>>;
        } else {
            return is_recursive_project_invocable<
                        unwrap_level - 1,
                        Proj,
                        F,
                        std::ranges::range_value_t<T>...>();
        }
    }

    //  recursive_project_invocable concept
    template<std::size_t unwrap_level, class Proj, class F, class... T>
    concept recursive_project_invocable =
            is_recursive_project_invocable<unwrap_level, Proj, F, T...>();

    /*  recursive_all_of template function implementation with unwrap level
    */
    template<std::size_t unwrap_level, class T, class Proj = std::identity, class UnaryPredicate>
    requires(   recursive_project_invocable<unwrap_level, Proj, UnaryPredicate, T>)
    constexpr auto recursive_all_of(T&& value, UnaryPredicate&& p, Proj&& proj = {}) {
        if constexpr (unwrap_level > 0)
        {
            return std::ranges::all_of(value, [&](auto&& element) {
                return recursive_all_of<unwrap_level - 1>(element, p, proj);
            });
        }
        else
        {
            return std::invoke(p, std::invoke(proj, value));
        }
    }

    /*  recursive_find template function implementation with unwrap level
    */
    template<std::size_t unwrap_level, class R, class T, class Proj = std::identity>
    requires(recursive_invocable<unwrap_level, Proj, R>)
    constexpr auto recursive_find(R&& range, T&& target, Proj&& proj = {})
    {
        if constexpr (unwrap_level)
        {
            return std::ranges::find_if(range, [&](auto& element) {
                return recursive_find<unwrap_level - 1>(element, target, proj);
            }) != std::ranges::end(range);
        }
        else
        {
            return target == std::invoke(proj, range);
        }
    }

    /*  recursive_find template function implementation with unwrap level, execution policy
    */
    template<std::size_t unwrap_level, class ExecutionPolicy, class R, class T, class Proj = std::identity>
    requires(recursive_invocable<unwrap_level, Proj, R>&&
            std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr auto recursive_find(ExecutionPolicy execution_policy, R&& range, T&& target, Proj&& proj = {})
    {
        if constexpr (unwrap_level)
        {
            return std::find_if(execution_policy,
                        std::ranges::begin(range),
                        std::ranges::end(range),
                        [&](auto& element) {
                return recursive_find<unwrap_level - 1>(execution_policy, element, target, proj);
            }) != std::ranges::end(range);
        }
        else
        {
            return target == std::invoke(proj, range);
        }
    }

    /*  recursive_find_if template function implementation with unwrap level
    */
    template<std::size_t unwrap_level, class T, class Proj = std::identity, class UnaryPredicate>
    requires(   recursive_project_invocable<unwrap_level, Proj, UnaryPredicate, T>)
    constexpr auto recursive_find_if(T&& value, UnaryPredicate&& p, Proj&& proj = {}) {
        if constexpr (unwrap_level > 0)
        {
            return std::ranges::find_if(value, [&](auto& element) {
                return recursive_find_if<unwrap_level - 1>(element, p, proj);
            }) != std::ranges::end(value);
        }
        else
        {
            return std::invoke(p, std::invoke(proj, value));
        }
    }

    //  recursive_any_of template function implementation with unwrap level
    template<std::size_t unwrap_level, class T, class Proj = std::identity, class UnaryPredicate>
    requires(recursive_project_invocable<unwrap_level, Proj, UnaryPredicate, T>)
    constexpr auto recursive_any_of(T&& value, UnaryPredicate&& p, Proj&& proj = {})
    {
        return recursive_find_if<unwrap_level>(value, p, proj);
    }
    
    //  recursive_none_of template function implementation with unwrap level
    template<std::size_t unwrap_level, class T, class Proj = std::identity, class UnaryPredicate>
    constexpr auto recursive_none_of(T&& value, UnaryPredicate&& p, Proj&& proj = {})
    {
        return !recursive_any_of<unwrap_level>(value, p, proj);
    }

    template<std::size_t index = 1, typename Arg, typename... Args>
    constexpr static auto& get_from_variadic_template(const Arg& first, const Args&... inputs)
    {
        if constexpr (index > 1)
            return get_from_variadic_template<index - 1>(inputs...);
        else
            return first;
    }

    //  first_of template function implementation
    template<typename... Args>
    constexpr static auto& first_of(const Args&... inputs) {
        return get_from_variadic_template<1>(inputs...);
    }

    template<std::size_t, typename, typename...>
    struct get_from_variadic_template_struct { };

    template<typename T1, typename... Ts>
    struct get_from_variadic_template_struct<1, T1, Ts...>
    {
        using type = T1;
    };

    template<std::size_t index, typename T1, typename... Ts>
    requires ( requires { typename get_from_variadic_template_struct<index - 1, Ts...>::type; })
    struct get_from_variadic_template_struct<index, T1, Ts...>
    {
        using type = typename get_from_variadic_template_struct<index - 1, Ts...>::type;
    };

    template<std::size_t index, typename... Ts>
    using get_from_variadic_template_t = typename get_from_variadic_template_struct<index, Ts...>::type;

    //  recursive_count implementation

    //  recursive_count implementation (the version with unwrap_level)
    template<std::size_t unwrap_level, class T>
    requires(unwrap_level <= recursive_depth<T>())
    constexpr auto recursive_count(const T& input, const auto& target)
    {
        if constexpr (unwrap_level > 0)
        {
            return std::transform_reduce(std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [&target](auto&& element) {
                return recursive_count<unwrap_level - 1>(element, target);
                });
        }
        else
        {
            return (input == target) ? 1 : 0;
        }
    }

    //  recursive_count implementation (the version with unwrap_level and execution policy)
    template<class ExPo, std::size_t unwrap_level, class T>
    requires(unwrap_level <= recursive_depth<T>() &&
             std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr auto recursive_count(ExPo execution_policy, const T& input, const auto& target)
    {
        if constexpr (unwrap_level > 0)
        {
            return std::transform_reduce(std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [&target](auto&& element) {
                return recursive_count<unwrap_level - 1>(element, target);
                });
        }
        else
        {
            return (input == target) ? 1 : 0;
        }
    }

    //  recursive_count_if implementation
    template<class T, std::invocable<T> Pred>
    constexpr std::size_t recursive_count_if(const T& input, const Pred& predicate)
    {
        return predicate(input) ? std::size_t{1} : std::size_t{0};
    }

    template<std::ranges::input_range Range, class Pred>
    requires (!std::invocable<Pred, Range>)
    constexpr auto recursive_count_if(const Range& input, const Pred& predicate)
    {
        return std::transform_reduce(std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [predicate](auto&& element) {
            return recursive_count_if(element, predicate);
            });
    }

    template<std::size_t unwrap_level, class T, class Pred>
    requires(unwrap_level <= recursive_depth<T>())
    constexpr auto recursive_count_if(const T& input, const Pred& predicate)
    {
        if constexpr (unwrap_level > 0)
        {
            return std::transform_reduce(std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [predicate](auto&& element) {
                return recursive_count_if<unwrap_level - 1>(element, predicate);
                });
        }
        else
        {
            return predicate(input) ? 1 : 0;
        }
        
    }

    //  recursive_count_if implementation (with execution policy)
    template<class ExPo, class T, std::invocable<T> Pred>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr std::size_t recursive_count_if(ExPo execution_policy, const T& input, const Pred& predicate)
    {
        return predicate(input) ? std::size_t{1} : 0;
    }

    template<class ExPo, std::ranges::input_range Range, class Pred>
    requires ((std::is_execution_policy_v<std::remove_cvref_t<ExPo>>) && (!std::invocable<Pred, Range>))
    constexpr auto recursive_count_if(ExPo execution_policy, const Range& input, const Pred& predicate)
    {
        return std::transform_reduce(execution_policy, std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [predicate](auto&& element) {
            return recursive_count_if(element, predicate);
            });
    }

    //  recursive_count_if implementation (the version with unwrap_level)
    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred>
    requires(unwrap_level <= recursive_depth<T>())
    constexpr auto recursive_count_if(const T& input, const Pred& predicate)
    {
        if constexpr (unwrap_level > 1)
        {
            return std::transform_reduce(std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [predicate](auto&& element) {
                return recursive_count_if<unwrap_level - 1>(element, predicate);
                });
        }
        else
        {
            return std::count_if(std::ranges::cbegin(input), std::ranges::cend(input), predicate);
        }
    }

    //  batch_recursive_count_if implementation (the version with unwrap_level)
    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1>
    requires(unwrap_level <= recursive_depth<T>())
    constexpr auto batch_recursive_count_if(const T& input, const Pred1& predicate1)
    {
        std::vector<decltype(recursive_count_if<unwrap_level>(input, predicate1))> output;
        output.push_back(recursive_count_if<unwrap_level>(input, predicate1));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class... Preds>
    requires(unwrap_level <= recursive_depth<T>())
    constexpr auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Preds&... predicates)
    {
        auto output1 = batch_recursive_count_if<unwrap_level>(input, predicate1);
        auto output2 = batch_recursive_count_if<unwrap_level>(input, predicates...);
        output1.insert(std::ranges::cend(output1), std::ranges::cbegin(output2), std::ranges::cend(output2));
        return output1;
    }

    //  recursive_max template function implementation
    template<class T, class Proj = std::identity,
        std::indirect_strict_weak_order<
                std::projected<const T*, Proj>> Comp = std::ranges::less>
    requires(!std::ranges::input_range<T>)          //  non-range overloading
    static inline T recursive_max(T inputNumber, Comp comp = {}, Proj proj = {})
    {
        return std::invoke(proj, inputNumber);
    }

    template<std::ranges::input_range T, class Proj = std::identity,
             std::indirect_strict_weak_order<
             std::projected<const T*, Proj>> Comp = std::ranges::less>
    static inline auto recursive_max(const T& numbers, Comp comp = {}, Proj proj = {})
    {
        auto output = recursive_max(numbers.at(0), comp, proj);
        for (auto& element : numbers)
        {
            output = std::ranges::max(
                output,
                recursive_max(element, comp, proj),
                comp,
                proj);
        }
        return output;
    }

    //  recursive_min template function implementation
    template<class T, class Proj = std::identity,
             std::indirect_strict_weak_order<
             std::projected<const T*, Proj>> Comp = std::ranges::less>
    requires(!std::ranges::input_range<T>)          //  non-range overloading
    static inline T recursive_min(T inputNumber, Comp comp = {}, Proj proj = {})
    {
        return std::invoke(proj, inputNumber);
    }

    template<std::ranges::input_range T, class Proj = std::identity,
             std::indirect_strict_weak_order<
             std::projected<const T*, Proj>> Comp = std::ranges::less>
    static inline auto recursive_min(const T& numbers, Comp comp = {}, Proj proj = {})
    {
        auto output = recursive_min(numbers.at(0), comp, proj);
        for (auto& element : numbers)
        {
            output = std::ranges::min(
                output,
                recursive_min(element, comp, proj),
                comp,
                proj);
        }
        return output;
    }

    //  recursive_print template function implementation
    template<typename T>
    constexpr void recursive_print(const T& input, const std::size_t level = 0)
    {
        #ifdef __cpp_lib_format
        std::cout << std::string(level, ' ') << std::format("{}", input) << '\n';
        #else
        std::cout << std::string(level, ' ') << input << '\n';
        #endif
    }

    template<std::ranges::input_range Range>
    constexpr void recursive_print(const Range& input, const std::size_t level = 0)
    {
        std::cout << std::string(level, ' ') << "Level " << level << ":" << std::endl;
        std::ranges::for_each(input, [level](auto&& element) {
            recursive_print(element, level + 1);
        });
    }

    //  recursive_size template function implementation
    template<class T> requires (!std::ranges::range<T>)
    constexpr auto recursive_size(const T& input)
    {
        return std::size_t{1};
    }

    template<std::ranges::range Range> requires (!(std::ranges::input_range<std::ranges::range_value_t<Range>>))
    constexpr auto recursive_size(const Range& input)
    {
        return std::ranges::size(input);
    }

    template<std::ranges::range Range> requires (std::ranges::input_range<std::ranges::range_value_t<Range>>)
    constexpr auto recursive_size(const Range& input)
    {
        return std::transform_reduce(std::ranges::begin(input), std::ranges::end(input), std::size_t{}, std::plus<std::size_t>(), [](auto& element) {
            return recursive_size(element);
            });
    }

    //  recursive_size template function implementation (the version with unwrap_level)
    template<std::size_t unwrap_level, std::ranges::range T>
    requires(unwrap_level <= recursive_depth<T>())
    constexpr auto recursive_size(const T input)
    {
        if constexpr (unwrap_level > 1)
        {
            return recursive_size<unwrap_level - 1>(input.at(0));
        }
        else
        {
            return std::ranges::size(input);
        }
    }
    
    /*  recursive_reduce_all template function performs operation on input container exhaustively
    */
    template<arithmetic T>
    constexpr auto recursive_reduce_all(const T& input)
    {
        return input;
    }

    //  recursive_reduce template function implementation
    //  Reference: https://codereview.stackexchange.com/a/251310/231235
    template<std::size_t unwrap_level = 1, class T, class ValueType, class Function = std::plus<ValueType>>
    constexpr auto recursive_reduce(const T& input, ValueType init, const Function& f = std::plus<ValueType>())
    {
        if constexpr (unwrap_level > 0)
        {
            for (const auto& element : input) {
                init = recursive_reduce<unwrap_level - 1>(element, init, f);
            }
            return init;
        }
        else
        {
            return std::invoke(f, init, input);
        }
    }

    template<typename T>
    concept is_recursive_reduceable = requires(T x)
    {
        recursive_reduce(x, T{});
    };

    template<typename T>
    concept is_recursive_sizeable = requires(T x)
    {
        recursive_size(x);
    };

    //  arithmetic_mean implementation
    template<class T = double, is_recursive_sizeable Container>
    constexpr auto arithmetic_mean(const Container& input)
    {
        if (recursive_size(input) == 0) //  Check the case of dividing by zero exception
        {
            throw std::logic_error("Divide by zero exception"); //  Handle the case of dividing by zero exception
        }
        return (recursive_reduce(input, T{})) / (recursive_size(input));
    }

    //  recursive_for_each function implementation
    template<std::size_t unwrap_level = 1, class UnaryFunction, typename Range>
    requires(unwrap_level <= recursive_depth<Range>())
    constexpr UnaryFunction recursive_for_each(UnaryFunction op, Range& input)
    {
        if constexpr (unwrap_level > 1)
        {
            std::for_each(
                std::ranges::begin(input),
                std::ranges::end(input),
                [&](auto&& element) { return recursive_for_each<unwrap_level - 1>(op, element); }
            );
            return op;
        }
        else
        {
            std::for_each(
                std::ranges::cbegin(input),
                std::ranges::cend(input),
                op);
            return op;
        }
    }

    //  recursive_for_each function implementation (with execution policy)
    template<std::size_t unwrap_level = 1, class ExPo, class UnaryFunction, typename Range>
    requires (unwrap_level <= recursive_depth<Range>() &&
              std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr UnaryFunction recursive_for_each(ExPo execution_policy, UnaryFunction op, Range& input)
    {
        if constexpr (unwrap_level > 1)
        {
            std::for_each(
                execution_policy,
                std::ranges::begin(input),
                std::ranges::end(input),
                [&](auto&& element) { return recursive_for_each<unwrap_level - 1>(execution_policy, op, element); }
            );
            return op;
        }
        else
        {
            std::for_each(
                execution_policy,
                std::ranges::cbegin(input),
                std::ranges::cend(input),
                op);
            return op;
        }
    }

    namespace impl {
        
        template<class F, class Proj = std::identity>
        struct recursive_for_each_state {
            F f;
            Proj proj;
        };

        //  recursive_foreach_all template function implementation
        template<class T, class State>
        requires (recursive_depth<T>() == 0)
        constexpr void recursive_foreach_all(T& value, State& state) {
            std::invoke(state.f, std::invoke(state.proj, value));
        }

        template<class T, class State>
        requires (recursive_depth<T>() != 0)
        constexpr void recursive_foreach_all(T& inputRange, State& state) {
            for (auto& item: inputRange)
                impl::recursive_foreach_all(item, state);
        }

        //  recursive_reverse_foreach_all template function implementation
        template<class T, class State>
        requires (recursive_depth<T>() == 0)
        constexpr void recursive_reverse_foreach_all(T& value, State& state) {
            std::invoke(state.f, std::invoke(state.proj, value));
        }

        template<class T, class State>
        requires (recursive_depth<T>() != 0)
        constexpr void recursive_reverse_foreach_all(T& inputRange, State& state) {
            for (auto& item: inputRange | std::views::reverse)
                impl::recursive_reverse_foreach_all(item, state);
        }
    }

    template<class T, class Proj = std::identity, class F>
    constexpr auto recursive_reverse_foreach_all(T& inputRange, F f, Proj proj = {})
    {
        impl::recursive_for_each_state state(std::move(f), std::move(proj));
        impl::recursive_reverse_foreach_all(inputRange, state);
        return std::make_pair(inputRange.end(), std::move(state.f));
    }

    //  recursive_fold_right_all template function implementation
    //  https://codereview.stackexchange.com/q/287842/231235
    template<class T, class I, class F>
    constexpr auto recursive_fold_right_all(const T& inputRange, I init, F f)
    {
        recursive_reverse_foreach_all(inputRange, [&](auto& value) {
            init = std::invoke(f, value, init);
        });

        return init;
    }

    //  recursive_replace_copy_if template function implementation
    template<std::size_t unwrap_level = 1, std::ranges::input_range Range, class UnaryPredicate, class T>
    requires(unwrap_level <= recursive_depth<Range>() &&
            (recursive_invocable<unwrap_level, UnaryPredicate, Range>))
    constexpr auto recursive_replace_copy_if(const Range& input, const UnaryPredicate& unary_predicate, const T& new_value)
    {
        if constexpr(unwrap_level == 1)
        {
            Range output{};
            std::ranges::replace_copy_if(
                std::ranges::cbegin(input),
                std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                unary_predicate,
                new_value);
            return output;
        }
        else
        {
            Range output{};

            std::ranges::transform(
                std::ranges::cbegin(input),
                std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                [&](auto&& element) { return recursive_replace_copy_if<unwrap_level - 1>(element, unary_predicate, new_value); }
            );
            return output;
        }

    }

    //  recursive_remove_copy_if function implementation with unwrap level
    template <std::size_t unwrap_level, std::ranges::input_range Range, class UnaryPredicate>
    requires(recursive_invocable<unwrap_level, UnaryPredicate, Range> &&
             is_inserterable<Range> &&
             unwrap_level > 0 &&
             unwrap_level <= recursive_depth<Range>())
    constexpr auto recursive_remove_copy_if(const Range& input, const UnaryPredicate& unary_predicate)
    {
        if constexpr(unwrap_level > 1)
        {
            Range output{};
        
            std::ranges::transform(
                std::ranges::cbegin(input),
                std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                [&unary_predicate](auto&& element) { return recursive_remove_copy_if<unwrap_level - 1>(element, unary_predicate); }
                );
            return output;
        }
        else
        {
            Range output{};
            std::ranges::remove_copy_if(std::ranges::cbegin(input), std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                unary_predicate);
            return output;
        }
    }

    //  recursive_remove_copy_if function implementation with unwrap level, execution policy
    template<std::size_t unwrap_level, class ExPo, std::ranges::input_range Range, class UnaryPredicate>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>> &&
             recursive_invocable<unwrap_level, UnaryPredicate, Range> &&
             is_inserterable<Range> &&
             unwrap_level > 0 &&
             unwrap_level <= recursive_depth<Range>())
    constexpr auto recursive_remove_copy_if(ExPo execution_policy, const Range& input, const UnaryPredicate& unary_predicate)
    {
        if constexpr(unwrap_level > 1)
        {
            Range output{};
        
            std::ranges::transform(
                std::ranges::cbegin(input),
                std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                [&](auto&& element) {
                    return recursive_remove_copy_if<unwrap_level - 1>(execution_policy, element, unary_predicate); 
                    }
                );
            return output;
        }
        else
        {
            Range output{};
            std::remove_copy_if(execution_policy, std::ranges::cbegin(input), std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                unary_predicate);
            return output;
        }
    }

    //  recursive_remove_copy function implementation with unwrap level
    template <std::size_t unwrap_level, std::ranges::input_range Range, class T>
    requires(is_inserterable<Range> &&
             unwrap_level > 0 &&
             unwrap_level <= recursive_depth<Range>())
    constexpr auto recursive_remove_copy(const Range& input, const T& value)
    {
        return recursive_remove_copy_if<unwrap_level>(input, [&](auto&& element) { return element == value; });
    }

    //  recursive_remove_copy function implementation with unwrap level, execution policy
    template <std::size_t unwrap_level, class ExPo, std::ranges::input_range Range, class T>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>> &&
             is_inserterable<Range> &&
             unwrap_level > 0 &&
             unwrap_level <= recursive_depth<Range>())
    constexpr auto recursive_remove_copy(ExPo execution_policy, const Range& input, const T& value)
    {
        return recursive_remove_copy_if<unwrap_level>(execution_policy, input, [&](auto&& element) { return element == value; });
    }

    template<typename OutputIt, std::copy_constructible NAryOperation, typename InputIt, typename... InputIts>
    OutputIt transform(OutputIt d_first, NAryOperation op, InputIt first, InputIt last, InputIts... rest) {
        while (first != last) {
            *d_first++ = op(*first++, (*rest++)...);
        }
        return d_first;
    }

    //  recursive_transform template function for the multiple parameters cases (the version with unwrap_level)
    template<std::size_t unwrap_level = 1, std::copy_constructible F, class Arg1, class... Args>
    requires(unwrap_level <= recursive_depth<Arg1>())
    constexpr auto recursive_transform(const F& f, const Arg1& arg1, const Args&... args)
    {
        if constexpr (unwrap_level > 0)
        {
            recursive_variadic_invoke_result_t<unwrap_level, F, Arg1, Args...> output{};
            transform(
                std::inserter(output, std::ranges::end(output)),
                [&f](auto&& element1, auto&&... elements) { return recursive_transform<unwrap_level - 1>(f, element1, elements...); },
                std::ranges::cbegin(arg1),
                std::ranges::cend(arg1),
                std::ranges::cbegin(args)...
            );
            return output;
        }
        else if constexpr (std::regular_invocable<F, Arg1, Args...>)
        {
            return std::invoke(f, arg1, args...);
        }
        else
        {
            static_assert(!std::regular_invocable<F, Arg1, Args...>, "The function passed to recursive_transform() cannot be invoked"
                                                                     "with the element types at the given recursion level.");
        }
    }

    //  recursive_transform template function for std::array (the version with unwrap_level)
    template< std::size_t unwrap_level = 1,
                template<class, std::size_t> class Container,
                typename T,
                std::size_t N,
                typename F,
                class... Args>
    requires (std::ranges::input_range<Container<T, N>>)
    constexpr auto recursive_transform(const F& f, const Container<T, N>& arg1, const Args&... args)
    {
        if constexpr (unwrap_level > 0)
        {
            recursive_array_invoke_result_t<unwrap_level, F, Container, T, N, Args...> output{};
            
            transform(
                std::ranges::begin(output),
                [&f](auto&& element1, auto&&... elements) { return recursive_transform<unwrap_level - 1>(f, element1, elements...); },
                std::ranges::cbegin(arg1),
                std::ranges::cend(arg1),
                std::ranges::cbegin(args)...
            );
            return output;
        }
        else if constexpr(std::regular_invocable<F, Container<T, N>, Args...>)
        {
            return std::invoke(f, arg1, args...);
        }
        else
        {
            static_assert(!std::regular_invocable<F, Container<T, N>, Args...>, "The function passed to recursive_transform() cannot be invoked"
                                                                                "with the element types at the given recursion level.");
        }
    }

    //  recursive_transform template function implementation (the version with unwrap_level, with execution policy)
    template<std::size_t unwrap_level = 1, class ExPo, class T, class F>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>> &&
              unwrap_level <= recursive_depth<T>())
    constexpr auto recursive_transform(ExPo execution_policy, const F& f, const T& input)
    {
        if constexpr (unwrap_level > 0)
        {
            recursive_invoke_result_t<unwrap_level, F, T> output{};
            output.resize(input.size());
            std::mutex mutex;
            std::transform(execution_policy, std::ranges::cbegin(input), std::ranges::cend(input), std::ranges::begin(output),
                [&](auto&& element)
                {
                    std::lock_guard lock(mutex);
                    return recursive_transform<unwrap_level - 1>(execution_policy, f, element);
                });
            return output;
        }
        else if constexpr (std::regular_invocable<F, T>)
        {
            return std::invoke(f, input);
        }
        else
        {
            static_assert(!std::regular_invocable<F, T>,    "The function passed to recursive_transform() cannot be invoked"
                                                            "with the element types at the given recursion level.");
        }
    }

    #ifdef USE_BOOST_ITERATOR
    #include <boost/iterator/zip_iterator.hpp>

    //  recursive_transform template function for the binary operation cases (the version with unwrap_level, with execution policy)
    template<std::size_t unwrap_level = 1, class ExPo, class T1, class T2, class F>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>> &&
              unwrap_level <= recursive_depth<T>())
    constexpr auto recursive_transform(ExPo execution_policy, const F& f, const T1& input1, const T2& input2)
    {
        if constexpr (unwrap_level > 0)
        {
            recursive_variadic_invoke_result_t<unwrap_level, F, T1, T2> output{};
            assert(input1.size() == input2.size());
            std::mutex mutex;

            //  Reference: https://stackoverflow.com/a/10457201/6667035
            //  Reference: https://www.boost.org/doc/libs/1_76_0/libs/iterator/doc/zip_iterator.html
            std::for_each(execution_policy,
                boost::make_zip_iterator(
                    boost::make_tuple(std::ranges::cbegin(input1), std::ranges::cbegin(input2))
                ),
                boost::make_zip_iterator(
                    boost::make_tuple(std::ranges::cend(input1), std::ranges::cend(input2))
                ),
                [&](auto&& elements)
                {
                    auto result = recursive_transform<unwrap_level - 1>(execution_policy, f, boost::get<0>(elements), boost::get<1>(elements));
                    std::lock_guard lock(mutex);
                    output.emplace_back(std::move(result));
                }
            );

            return output;
        }
        else
        {
            return std::invoke(f, input1, input2);
        }
    }
    #endif

    //  recursive_copy_if template function implementation
    template <std::ranges::input_range Range, std::invocable<std::ranges::range_value_t<Range>> UnaryPredicate>
    constexpr auto recursive_copy_if(const Range& input, const UnaryPredicate& unary_predicate)
    {
        Range output{};
        std::ranges::copy_if(std::ranges::cbegin(input), std::ranges::cend(input),
            std::inserter(output, std::ranges::end(output)),
            unary_predicate);
        return output;
    }

    template <
        std::ranges::input_range Range,
        class UnaryPredicate>
    constexpr auto recursive_copy_if(const Range& input, const UnaryPredicate& unary_predicate)
    {
        Range output{};

        std::ranges::transform(
            std::ranges::cbegin(input),
            std::ranges::cend(input),
            std::inserter(output, std::ranges::end(output)),
            [&unary_predicate](auto&& element) { return recursive_copy_if(element, unary_predicate); }
        );
        return output;
    }

    //  recursive_copy_if function implementation with unwrap level
    //  Reference: https://codereview.stackexchange.com/q/291308/231235
    template <std::size_t unwrap_level, std::ranges::input_range Range, class UnaryPredicate>
    requires(recursive_invocable<unwrap_level, UnaryPredicate, Range> &&
            is_inserterable<Range> &&
            unwrap_level > 0)
    constexpr auto recursive_copy_if(const Range& input, const UnaryPredicate& unary_predicate)
    {
        if constexpr(unwrap_level > 1)
        {
            Range output{};
        
            std::ranges::transform(
                std::ranges::cbegin(input),
                std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                [&unary_predicate](auto&& element) { return recursive_copy_if<unwrap_level - 1>(element, unary_predicate); }
                );
            return output;
        }
        else
        {
            Range output{};
            std::ranges::copy_if(std::ranges::cbegin(input), std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                unary_predicate);
            return output;
        }
    }

    //  recursive_transform_reduce template function implementation
    template<class Input, class T, class UnaryOp, class BinaryOp = std::plus<T>>
    constexpr auto recursive_transform_reduce(const Input& input, T init, const UnaryOp& unary_op, const BinaryOp& binop = std::plus<T>())
    {
        return binop(init, unary_op(input));
    }

    template<std::ranges::range Input, class T, class UnaryOp, class BinaryOp = std::plus<T>>
    constexpr auto recursive_transform_reduce(const Input& input, T init, const UnaryOp& unary_op, const BinaryOp& binop = std::plus<T>())
    {
        return std::transform_reduce(std::ranges::begin(input), std::ranges::end(input), init, binop, [&](auto& element) {
            return recursive_transform_reduce(element, T{}, unary_op, binop);
            });
    }

    //  recursive_transform_reduce template function implementation with execution policy
    template<class ExPo, class Input, class T, class UnaryOp, class BinaryOp = std::plus<T>>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>&&
              !std::ranges::range<Input>)
    constexpr auto recursive_transform_reduce(ExPo execution_policy, const Input& input, T init, const UnaryOp& unary_op, const BinaryOp& binop = std::plus<T>())
    {
        return binop(init, unary_op(input));
    }

    template<class ExPo, std::ranges::range Input, class T, class UnaryOp, class BinaryOp = std::plus<T>>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr auto recursive_transform_reduce(ExPo execution_policy, const Input& input, T init, const UnaryOp& unary_op, const BinaryOp& binop = std::plus<T>())
    {
        return std::transform_reduce(execution_policy, std::ranges::begin(input), std::ranges::end(input), init, binop, [&](auto& element) {
            return recursive_transform_reduce(execution_policy, element, T{}, unary_op, binop);
            });
    }

    //  recursive_transform_reduce template function implementation
    template<std::size_t unwrap_level, class Input, class T, class UnaryOp = std::identity, class BinaryOp = std::plus<T>>
    requires(recursive_invocable<unwrap_level, UnaryOp, Input>)
    constexpr auto recursive_transform_reduce(const Input& input, T init = {}, const UnaryOp& unary_op = {}, const BinaryOp& binop = std::plus<T>())
    {
        if constexpr (unwrap_level > 0)
        {
            return std::transform_reduce(std::ranges::begin(input), std::ranges::end(input), init, binop, [&](auto& element) {
                return recursive_transform_reduce<unwrap_level - 1>(element, T{}, unary_op, binop);
            });
        }
        else
        {
            return std::invoke(unary_op, input);
        }
    }

    //  recursive_transform_reduce template function implementation with execution policy
    template<std::size_t unwrap_level, class ExecutionPolicy, class Input, class T, class UnaryOp = std::identity, class BinaryOp = std::plus<T>>
    requires(recursive_invocable<unwrap_level, UnaryOp, Input>&&
            std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr auto recursive_transform_reduce(ExecutionPolicy execution_policy, const Input& input, T init = {}, const UnaryOp& unary_op = {}, const BinaryOp& binop = std::plus<T>())
    {
        if constexpr (unwrap_level > 0)
        {
            return std::transform_reduce(
                execution_policy,
                std::ranges::begin(input),
                std::ranges::end(input),
                init,
                binop,
                [&](auto& element) {
                return recursive_transform_reduce<unwrap_level - 1>(execution_policy, element, T{}, unary_op, binop);
            });
        }
        else
        {
            return std::invoke(unary_op, input);
        }
    }

    //  two_input_map_reduce Template Function Implementation
    template<
        class ExecutionPolicy,
        std::ranges::input_range Input1,
        std::ranges::input_range Input2,
        class T,
        class BinaryOp1 = std::minus<T>,
        class BinaryOp2 = std::plus<T>
    >
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto two_input_map_reduce(
        ExecutionPolicy execution_policy,
        const Input1& input1,
        const Input2& input2,
        const T init = {},
        const BinaryOp1& binop1 = std::minus<T>(),
        const BinaryOp2& binop2 = std::plus<T>())
    {
        if (input1.size() != input2.size())
        {
            throw std::runtime_error("Size mismatched!");
        }
        #if _HAS_CXX23
        auto transformed = std::views::zip(input1, input2)
                     | std::views::transform([&](auto input) {
                           return std::invoke(binop1, std::get<0>(input), std::get<1>(input));
                       });
        return std::reduce(
            execution_policy,
            transformed.begin(), transformed.end(),
            init,
            binop2
        );
        #else
        T output = init;
        for (std::size_t i = 0; i < input1.size(); ++i)
        {
            output = std::invoke(binop2, output, std::invoke(binop1, input1.at(i), input2.at(i)));
        }
        return output;
        #endif
    }

    template<typename T>
    concept can_calculate_variance_of = requires(const T & value)
    {
        (std::pow(value, 2) - value) / std::size_t{ 1 };
    };

    template<typename T>
    struct recursive_iter_value_t_detail
    {
        using type = T;
    };

    template <std::ranges::range T>
    struct recursive_iter_value_t_detail<T>
        : recursive_iter_value_t_detail<std::iter_value_t<T>>
    { };

    template<typename T>
    using recursive_iter_value_t = typename recursive_iter_value_t_detail<T>::type;

    //  population_variance function implementation (with recursive_transform_reduce template function)
    template<class T = double, is_recursive_sizeable Container>
    requires (can_calculate_variance_of<recursive_iter_value_t<Container>>)
    constexpr auto population_variance(const Container& input)
    {
        if (recursive_size(input) == 0) //  Check the case of dividing by zero exception
        {
            throw std::logic_error("Divide by zero exception"); //  Handle the case of dividing by zero exception
        }
        auto mean = arithmetic_mean<T>(input);
        return recursive_transform_reduce(std::execution::par,
            input, T{}, [mean](auto& element) {
                return std::pow(element - mean, 2);
            }, std::plus<T>()) / recursive_size(input);
    }

    //  population_variance function implementation (with recursive_transform_reduce template function, execution policy)
    template<class ExecutionPolicy, class T = double, is_recursive_sizeable Container>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
                can_calculate_variance_of<recursive_iter_value_t<Container>>)
    constexpr auto population_variance(ExecutionPolicy execution_policy, const Container& input)
    {
        if (recursive_size(input) == 0) //  Check the case of dividing by zero exception
        {
            throw std::logic_error("Divide by zero exception"); //  Handle the case of dividing by zero exception
        }
        auto mean = arithmetic_mean<T>(input);
        return recursive_transform_reduce(execution_policy,
            input, T{}, [mean](auto& element) {
                return std::pow(element - mean, 2);
            }, std::plus<T>()) / recursive_size(input);
    }

    //  population_standard_deviation template function implementation
    template<class T = double, is_recursive_sizeable Container>
    requires (can_calculate_variance_of<recursive_iter_value_t<Container>>)
    constexpr auto population_standard_deviation(const Container& input)
    {
        return population_standard_deviation(std::execution::par, input);
    }

    //  population_standard_deviation template function implementation
    template<class ExecutionPolicy, class T = double, is_recursive_sizeable Container>
    requires (  std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>> &&
                can_calculate_variance_of<recursive_iter_value_t<Container>>)
    constexpr auto population_standard_deviation(ExecutionPolicy execution_policy, const Container& input)
    {
        if (recursive_size(input) == 0) //  Check the case of dividing by zero exception
        {
            throw std::logic_error("Divide by zero exception"); //  Handle the case of dividing by zero exception
        }
        return std::pow(population_variance(execution_policy, input), 0.5);
    }

    //  square_sum template function implementation
    template<class ExecutionPolicy, class T = double, std::ranges::input_range Container>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr auto square_sum(ExecutionPolicy execution_policy, const Container& input)
    {
        return recursive_transform_reduce(execution_policy,
            input, T{}, [&](auto& element) {
                return std::pow(element, 2);
            }, std::plus<T>());
    }

    //  square_sum template function implementation
    template<class Container>
    constexpr auto square_sum(const Container& input)
    {
        return square_sum(std::execution::seq, input);
    }

    //  root_mean_square template function implementation
    template<class ExecutionPolicy, is_recursive_sizeable Container>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    constexpr static auto root_mean_square(ExecutionPolicy execution_policy, const Container& input)
    {
        return std::sqrt(square_sum(execution_policy, input) / recursive_size(input));
    }

    //  root_mean_square template function implementation
    template<is_recursive_sizeable Container>
    constexpr static auto root_mean_square(const Container& input)
    {
        return root_mean_square(std::execution::seq, input);
    }

    template<std::size_t dim, class T>
    constexpr auto n_dim_vector_generator(T input, std::size_t times)
    {
        if constexpr (dim == 0)
        {
            return input;
        }
        else
        {
            auto element = n_dim_vector_generator<dim - 1>(input, times);
            std::vector<decltype(element)> output(times, element);
            return output;
        }
    }

    template<std::size_t dim, std::size_t times, class T>
    constexpr auto n_dim_array_generator(T input)
    {
        if constexpr (dim == 0)
        {
            return input;
        }
        else
        {
            auto element = n_dim_array_generator<dim - 1, times>(input);
            std::array<decltype(element), times> output;
            std::fill(std::ranges::begin(output), std::ranges::end(output), element);
            return output;
        }
    }

    template<std::size_t dim, class T>
    constexpr auto n_dim_deque_generator(T input, std::size_t times)
    {
        if constexpr (dim == 0)
        {
            return input;
        }
        else
        {
            auto element = n_dim_deque_generator<dim - 1>(input, times);
            std::deque<decltype(element)> output(times, element);
            return output;
        }
    }

    template<std::size_t dim, class T>
    constexpr auto n_dim_list_generator(T input, std::size_t times)
    {
        if constexpr (dim == 0)
        {
            return input;
        }
        else
        {
            auto element = n_dim_list_generator<dim - 1>(input, times);
            std::list<decltype(element)> output(times, element);
            return output;
        }
    }

    template<std::size_t dim, class T, template<class...> class Container = std::vector>
    constexpr auto n_dim_container_generator(T input, std::size_t times)
    {
        if constexpr (dim == 0)
        {
            return input;
        }
        else
        {
            return Container(times, n_dim_container_generator<dim - 1, T, Container>(input, times));
        }
    }

    namespace impl {
        struct recursive_flatten_fn
        {
            //  recursive_flatten template function implementation
            template<std::ranges::range T, class OutputContainer>
            constexpr auto operator()(const T& input, OutputContainer& output_container) const
            {
                output_container.append_range(input);
                return output_container;
            }

            template<std::ranges::range Container, class OutputContainer>
            requires (std::ranges::range<std::ranges::range_value_t<Container>>)
            constexpr auto operator()(const Container& input, OutputContainer& output_container) const
            {
                for (const auto& element : input) {
                    output_container = operator()(element, output_container);
                }
                return output_container;
            }
        };

        inline constexpr recursive_flatten_fn recursive_flatten;
    }

    //  recursive_flatten template function implementation with unwrap level
    template<std::size_t unwrap_level, class T, class OutputContainer>
    constexpr auto recursive_flatten(const T& input, OutputContainer& output_container)
    {
        if constexpr (unwrap_level > 1)
        {
            for (const auto& element : input) {
                output_container = recursive_flatten<unwrap_level - 1>(element, output_container);
            }
            return output_container;
        }
        else
        {
            #if __cplusplus > 202302L || _HAS_CXX23 
            output_container.append_range(input);
            #else
            for (const auto& element : input)
            {
                output_container.emplace_back(element);
            }
            #endif
            return output_container;
        }
    }

    #if __cplusplus > 202302L || _HAS_CXX23 
    //  recursive_flatten_view template function implementation with unwrap level
    //  Reference: https://codereview.stackexchange.com/q/295937/231235
    template<std::size_t unwrap_level, typename T>
    static std::generator<const recursive_unwrap_type_t<unwrap_level, T>&> recursive_flatten_view(const T& input)
    {
        if constexpr (unwrap_level > 0)
        {
            for (const auto& element : input)
                for (const auto& value : recursive_flatten_view<unwrap_level - 1>(element))
                    co_yield value;
        }
        else
        {
            co_yield input;
        }
    }
    #else                   //  Suboptimal solution
    //  recursive_flatten_view template function implementation with unwrap level
    //  Reference: https://codereview.stackexchange.com/q/291793/231235
    template<std::size_t unwrap_level, std::ranges::range Container>
    requires (std::ranges::range<std::ranges::range_value_t<Container>>)
    constexpr static auto recursive_flatten_view(const Container& input)
    {
        recursive_unwrap_type_t<recursive_depth<Container>() - 1, Container> output_container;
        return std::views::all(recursive_flatten<unwrap_level>(input, output_container));
    }
    #endif

    //  recursive_minmax template function implementation
    //  Reference: https://codereview.stackexchange.com/q/288208/231235
    template<std::size_t unwrap_level = 1, std::ranges::forward_range T, class Proj = std::identity,
                    std::indirect_strict_weak_order<
                    std::projected<std::ranges::iterator_t<T>, Proj>> Comp = std::ranges::less>
    constexpr static auto recursive_minmax(T&& numbers, Comp comp = {}, Proj proj = {})
    {
        if constexpr (unwrap_level > 1)
        {
            return std::ranges::minmax(recursive_flatten_view<unwrap_level>(numbers), comp, proj);
        }
        else
        {
            return std::ranges::minmax(numbers, comp, proj);
        }
    }

    //  hypot Template Function Implementation
    template<typename... Args>
    constexpr auto hypot(Args... args)
    {
        return std::sqrt((std::pow(args, 2.0) + ...));
    }

    //  Multichannel Concept Implementation
    template<typename T>
    concept Multichannel = requires(T a)
    {
        { a.channels }; // or whatever is best to check for multiple channels
    };

    //  apply_multichannel Template Function Implementation
    template<std::size_t channel_count = 3, class ElementT, class Lambda, typename... Args>
    [[nodiscard]] constexpr static auto apply_multichannel(const MultiChannel<ElementT, channel_count>& input, Lambda f, Args... args)
    {
        MultiChannel<decltype(std::invoke(f, input.channels[0], args...)), channel_count> output;
        std::transform(std::ranges::cbegin(input.channels), std::ranges::cend(input.channels), std::ranges::begin(output.channels),
            [&](auto&& input) { return std::invoke(f, input, args...); });
        return output;
    }

    //  apply_multichannel Template Function Implementation (the version with execution policy)
    template<std::size_t channel_count = 3, class ExecutionPolicy, class ElementT, class Lambda, typename... Args>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto apply_multichannel(ExecutionPolicy&& execution_policy, const MultiChannel<ElementT, channel_count>& input, Lambda f, Args... args)
    {
        MultiChannel<decltype(std::invoke(f, input.channels[0], args...)), channel_count> output;
        std::transform(std::forward<ExecutionPolicy>(execution_policy), std::ranges::cbegin(input.channels), std::ranges::cend(input.channels), std::ranges::begin(output.channels),
            [&](auto&& input) { return std::invoke(f, input, args...); });
        return output;
    }

    //  apply_multichannel Template Function Implementation
    template<std::size_t channel_count = 3, class T, class Lambda, typename... Args>
    requires((std::same_as<T, RGB>) || (std::same_as<T, RGB_DOUBLE>) || (std::same_as<T, HSV>))
    [[nodiscard]] constexpr static auto apply_multichannel(const T& input, Lambda f, Args... args)
    {
        MultiChannel<decltype(std::invoke(f, input.channels[0], args...)), channel_count> output;
        std::transform(std::ranges::cbegin(input.channels), std::ranges::cend(input.channels), std::ranges::begin(output.channels),
            [&](auto&& input) { return std::invoke(f, input, args...); });
        return output;
    }

    //  apply_multichannel Template Function Implementation (the version with execution policy)
    template<std::size_t channel_count = 3, class ExecutionPolicy, class T, class Lambda, typename... Args>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)&&
             ((std::same_as<T, RGB>) || (std::same_as<T, RGB_DOUBLE>) || (std::same_as<T, HSV>))
    [[nodiscard]] constexpr static auto apply_multichannel(ExecutionPolicy&& execution_policy, const T& input, Lambda f, Args... args)
    {
        MultiChannel<decltype(std::invoke(f, input.channels[0], args...)), channel_count> output;
        std::transform(std::forward<ExecutionPolicy>(execution_policy), std::ranges::cbegin(input.channels), std::ranges::cend(input.channels), std::ranges::begin(output.channels),
            [&](auto&& input) { return std::invoke(f, input, args...); });
        return output;
    }

    //  append_aux Template Function Implementation
    //  https://stackoverflow.com/a/41398948/6667035
    template <typename T, std::size_t N, std::size_t... I>
    constexpr std::array<T, N + 1> append_aux(std::array<T, N> a, T t, std::index_sequence<I...>)
    {
        return std::array<T, N + 1>{ a[I]..., t };
    }

    //  append Template Function Implementation
    //  https://stackoverflow.com/a/41398948/6667035
    template <typename T, std::size_t N>
    constexpr std::array<T, N + 1> append(std::array<T, N> a, T t)
    {
        return append_aux(a, t, std::make_index_sequence<N>());
    }

    //  abs Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto abs(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return abs(_input); });
        }
        else
        {
            return std::abs(input);
        }
    }

    //  abs Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto abs(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return abs(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else
        {
            return std::abs(input);
        }
    }

    //  pow Template Function Implementation
    template<typename T, typename ExpT = double>
    [[nodiscard]] constexpr static auto pow(const T& input, const ExpT exp)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input, auto&& input_exp) {return pow(_input, input_exp); }, exp);
        }
        else
        {
            return std::pow(input, exp);
        }
    }

    //  pow Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T, typename ExpT = double>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto pow(ExecutionPolicy&& execution_policy, const T& input, const ExpT exp)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input, auto&& input_exp) {return pow(std::forward<ExecutionPolicy>(execution_policy), _input, input_exp); }, exp);
        }
        else
        {
            return std::pow(input, exp);
        }
    }
    
    //  sqrt Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto sqrt(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return sqrt(_input); });
        }
        else
        {
            return std::sqrt(input);
        }
    }

    //  sqrt Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto sqrt(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return sqrt(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else
        {
            return std::sqrt(input);
        }
    }

    //  cbrt Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto cbrt(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return cbrt(_input); });
        }
        else
        {
            return std::cbrt(input);
        }
    }

    //  cbrt Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto cbrt(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return cbrt(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else
        {
            return std::cbrt(input);
        }
    }

    //  sin Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto sin(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return sin(_input); });
        }
        else
        {
            return std::sin(input);
        }
    }

    //  sin Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto sin(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return sin(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else
        {
            return std::sin(input);
        }
    }

    //  cos Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto cos(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return cos(_input); });
        }
        else
        {
            return std::cos(input);
        }
    }

    //  cos Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto cos(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return cos(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else
        {
            return std::cos(input);
        }
    }

    //  tan Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto tan(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return tan(_input); });
        }
        else
        {
            return std::tan(input);
        }
    }

    //  tan Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto tan(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return tan(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else
        {
            return std::tan(input);
        }
    }

    //  cot Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto cot(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return cot(_input); });
        }
        else if constexpr (is_complex<T>::value)
        {
            return static_cast<T>(1) / tan(input);
        }
        else
        {
            return 1 / std::tan(input);
        }
    }

    //  cot Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto cot(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return cot(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else if constexpr (is_complex<T>::value)
        {
            return static_cast<T>(1) / tan(input);
        }
        else
        {
            return 1 / std::tan(input);
        }
    }

    //  sec Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto sec(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return 1 / cos(_input); });
        }
        else if constexpr (is_complex<T>::value)
        {
            return static_cast<T>(1) / cos(input);
        }
        else
        {
            return 1 / std::cos(input);
        }
    }

    //  sec Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto sec(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return 1 / cos(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else if constexpr (is_complex<T>::value)
        {
            return static_cast<T>(1) / cos(input);
        }
        else
        {
            return 1 / std::cos(input);
        }
    }

    //  csc Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto csc(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return 1 / std::sin(_input); });
        }
        else
        {
            return 1 / std::sin(input);
        }
    }

    //  csc Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto csc(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return 1 / std::sin(_input); });
        }
        else
        {
            return 1 / std::sin(input);
        }
    }

    //  asin Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto asin(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return asin(_input); });
        }
        else
        {
            return std::asin(input);
        }
    }

    //  asin Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto asin(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return std::asin(_input); });
        }
        else
        {
            return std::asin(input);
        }
    }

    //  acos Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto acos(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return std::acos(_input); });
        }
        else
        {
            return std::acos(input);
        }
    }

    //  acos Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto acos(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return std::acos(_input); });
        }
        else
        {
            return std::acos(input);
        }
    }

    //  atan Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto atan(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return std::atan(_input); });
        }
        else
        {
            return std::atan(input);
        }
    }

    //  atan Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto atan(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return std::atan(_input); });
        }
        else
        {
            return std::atan(input);
        }
    }

    //  sinh Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto sinh(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return std::sinh(_input); });
        }
        else
        {
            return std::sinh(input);
        }
    }

    //  sinh Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto sinh(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return std::sinh(_input); });
        }
        else
        {
            return std::sinh(input);
        }
    }

    //  cosh Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto cosh(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return std::cosh(_input); });
        }
        else
        {
            return std::cosh(input);
        }
    }

    //  cosh Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto cosh(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return std::cosh(_input); });
        }
        else
        {
            return std::cosh(input);
        }
    }

    //  tanh Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto tanh(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return std::tanh(_input); });
        }
        else
        {
            return std::tanh(input);
        }
    }

    //  tanh Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto tanh(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return std::tanh(_input); });
        }
        else
        {
            return std::tanh(input);
        }
    }

    //  asinh Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto asinh(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return std::asinh(_input); });
        }
        else
        {
            return std::asinh(input);
        }
    }

    //  asinh Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto asinh(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return std::asinh(_input); });
        }
        else
        {
            return std::asinh(input);
        }
    }

    //  acosh Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto acosh(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return acosh(_input); });
        }
        else
        {
            return std::acosh(input);
        }
    }

    //  acosh Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto acosh(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return acosh(_input); });
        }
        else
        {
            return std::acosh(input);
        }
    }

    //  atanh Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto atanh(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return atanh(_input); });
        }
        else
        {
            return std::atanh(input);
        }
    }

    //  atanh Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto atanh(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return atanh(_input); });
        }
        else
        {
            return std::atanh(input);
        }
    }

    //  erf Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto erf(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return erf(_input); });
        }
        else
        {
            return std::erf(input);
        }
    }

    //  erf Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto erf(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(execution_policy, input, [&](auto&& _input) {return erf(_input); });
        }
        else
        {
            return std::erf(input);
        }
    }

    //  erfc Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto erfc(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return erfc(_input); });
        }
        else
        {
            return std::erfc(input);
        }
    }

    //  erfc Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto erfc(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(std::forward<ExecutionPolicy>(execution_policy), input, [&](auto&& _input) {return erfc(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else
        {
            return std::erfc(input);
        }
    }

    //  lgamma Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto lgamma(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return lgamma(_input); });
        }
        else
        {
            return std::lgamma(input);
        }
    }

    //  lgamma Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto lgamma(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return lgamma(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::lgamma(input);
        }
    }

    //  tgamma Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto tgamma(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return tgamma(_input); });
        }
        else
        {
            return std::tgamma(input);
        }
    }

    //  tgamma Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto tgamma(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return tgamma(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::tgamma(input);
        }
    }

    //  ceil Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto ceil(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return ceil(_input); });
        }
        else
        {
            return std::ceil(input);
        }
    }

    //  ceil Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto ceil(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return ceil(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::ceil(input);
        }
    }

    //  floor Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto floor(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return floor(_input); });
        }
        else
        {
            return std::floor(input);
        }
    }

    //  floor Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto floor(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return floor(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::floor(input);
        }
    }

    //  trunc Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto trunc(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return trunc(_input); });
        }
        else
        {
            return std::trunc(input);
        }
    }

    //  trunc Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto trunc(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return trunc(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::trunc(input);
        }
    }

    //  nearbyint Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto nearbyint(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return nearbyint(_input); });
        }
        else
        {
            return std::nearbyint(input);
        }
    }

    //  nearbyint Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto nearbyint(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return nearbyint(std::forward<ExecutionPolicy>(execution_policy), _input); });
        }
        else
        {
            return std::nearbyint(input);
        }
    }

    //  rint Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto rint(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return rint(_input); });
        }
        else
        {
            return std::rint(input);
        }
    }

    //  rint Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto rint(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return rint(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::rint(input);
        }
    }

    //  ilogb Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto ilogb(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return ilogb(_input); });
        }
        else
        {
            return std::ilogb(input);
        }
    }

    //  ilogb Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto ilogb(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return ilogb(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::ilogb(input);
        }
    }

    //  logb Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto logb(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return logb(_input); });
        }
        else
        {
            return std::logb(input);
        }
    }

    //  logb Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto logb(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return logb(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::logb(input);
        }
    }

    //  fpclassify Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto fpclassify(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return fpclassify(_input); });
        }
        else
        {
            return std::fpclassify(input);
        }
    }

    //  fpclassify Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto fpclassify(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return fpclassify(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::fpclassify(input);
        }
    }

    //  isfinite Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto isfinite(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return isfinite(_input); });
        }
        else
        {
            return std::isfinite(input);
        }
    }

    //  isfinite Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto isfinite(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return isfinite(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::isfinite(input);
        }
    }

    //  isinf Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto isinf(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return isinf(_input); });
        }
        else
        {
            return std::isinf(input);
        }
    }

    //  isinf Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto isinf(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return isinf(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::isinf(input);
        }
    }

    //  isnan Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto isnan(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return isnan(_input); });
        }
        else
        {
            return std::isnan(input);
        }
    }

    //  isnan Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto isnan(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return isnan(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::isnan(input);
        }
    }

    //  isnormal Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto isnormal(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return isnormal(_input); });
        }
        else
        {
            return std::isnormal(input);
        }
    }

    //  isnormal Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto isnormal(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return isnormal(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::isnormal(input);
        }
    }

    //  signbit Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto signbit(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return signbit(_input); });
        }
        else
        {
            return std::signbit(input);
        }
    }

    //  signbit Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto signbit(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return signbit(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::signbit(input);
        }
    }

    //  exp Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto exp(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return exp(_input); });
        }
        else
        {
            return std::exp(input);
        }
    }

    //  exp Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto exp(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return exp(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::exp(input);
        }
    }

    //  exp2 Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto exp2(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return exp2(_input); });
        }
        else
        {
            return std::exp2(input);
        }
    }

    //  exp2 Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto exp2(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return exp2(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::exp2(input);
        }
    }

    //  expm1 Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto expm1(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return expm1(_input); });
        }
        else
        {
            return std::expm1(input);
        }
    }

    //  expm1 Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto expm1(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return expm1(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::expm1(input);
        }
    }

    //  log Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto log(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return log(_input); });
        }
        else
        {
            return std::log(input);
        }
    }

    //  log Template Function Implementation (the version with execution policy)
    template<class ExecutionPolicy, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
    [[nodiscard]] constexpr static auto log(ExecutionPolicy&& execution_policy, const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(
                std::forward<ExecutionPolicy>(execution_policy),
                input,
                [&](auto&& _input) {return log(std::forward<ExecutionPolicy>(execution_policy), _input); }
            );
        }
        else
        {
            return std::log(input);
        }
    }

    //  log10 Template Function Implementation
    template<typename T>
    [[nodiscard]] constexpr static auto log10(const T& input)
    {
        if constexpr (Multichannel<T>)
        {
            return apply_multichannel(input, [&](auto&& _input) {return log10(_input); });
        }
        else
        {
            return std::log10(input);
        }
    }


    // sum_first_element Template Function Implementation
    template <typename FirstT, typename SecondT, class Function = std::plus<FirstT>>
    requires(std::regular_invocable<Function, FirstT, FirstT>)
    constexpr static FirstT sum_first_element(const std::vector<std::pair<FirstT, SecondT>>& pairs, const Function& f = Function{})
    {
        FirstT sum{};
        for (auto const& [first, second] : pairs)
        {
            sum = std::invoke(f, sum, first);
        }
        return sum;
    }

    //  sum_first_element Template Function Implementation (with execution policy)
    template <class ExPo, typename FirstT, typename SecondT, class Function = std::plus<FirstT>>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>> and
             std::regular_invocable<Function, FirstT, FirstT>)
    constexpr static FirstT sum_first_element(ExPo&& execution_policy, const std::vector<std::pair<FirstT, SecondT>>& pairs, const Function& f = Function{})
    {
        const std::size_t size = pairs.size();
        std::vector<FirstT> first_elements(size);
        std::transform(
            std::forward<ExPo>(execution_policy),
            std::ranges::cbegin(pairs),
            std::ranges::cend(pairs),
            std::ranges::begin(first_elements),
            [](auto const& pair) { return pair.first; });
        return std::reduce(std::forward<ExPo>(execution_policy), std::ranges::cbegin(first_elements), std::ranges::cend(first_elements), FirstT{}, f);
    }

    // sum_first_element Template Function Implementation
    template <typename KeyT, typename ValueT, class Function = std::plus<KeyT>>
    requires(std::regular_invocable<Function, KeyT, KeyT>)
    constexpr static KeyT sum_first_element(const std::map<KeyT, ValueT>& map, const Function& f = Function{})
    {
        KeyT sum{};
        for (const auto& [key, value] : map)
        {
            sum = std::invoke(f, sum, key);
        }
        return sum;
    }

    //  sum_second_element Template Function Implementation
    template <typename FirstT, typename SecondT, class Function = std::plus<SecondT>>
    requires(std::regular_invocable<Function, SecondT, SecondT>)
    constexpr static SecondT sum_second_element(const std::vector<std::pair<FirstT, SecondT>>& pairs, const Function& f = Function{})
    {
        SecondT sum{};
        for (auto const& [first, second] : pairs)
        {
            sum = std::invoke(f, sum, second);
        }
        return sum;
    }

    //  sum_second_element Template Function Implementation (with execution policy)
    template <class ExPo, typename FirstT, typename SecondT, class Function = std::plus<SecondT>>
    requires(std::is_execution_policy_v<std::remove_cvref_t<ExPo>> and
             std::regular_invocable<Function, SecondT, SecondT>)
    constexpr static SecondT sum_second_element(ExPo&& execution_policy, const std::vector<std::pair<FirstT, SecondT>>& pairs, const Function& f = Function{})
    {
        const std::size_t size = pairs.size();
        std::vector<SecondT> second_elements(size);
        std::transform(
            std::forward<ExPo>(execution_policy),
            std::ranges::cbegin(pairs),
            std::ranges::cend(pairs),
            std::ranges::begin(second_elements),
            [&](auto const& pair) { return pair.second; });
        return std::reduce(std::forward<ExPo>(execution_policy), std::ranges::cbegin(second_elements), std::ranges::cend(second_elements), SecondT{}, f);
    }

    //  sum_second_element Template Function Implementation
    template <typename KeyT, typename ValueT, class Function = std::plus<ValueT>>
    requires(std::regular_invocable<Function, ValueT, ValueT>)
    constexpr static ValueT sum_second_element(const std::map<KeyT, ValueT>& map, const Function& f = Function{})
    {
        ValueT sum{};
        for (const auto& [key, value] : map)
        {
            sum = std::invoke(f, sum, value);
        }
        return sum;
    }
    //  Formatter class implementation
    //  Copy from https://stackoverflow.com/a/12262626/6667035
    class Formatter
    {
    public:
        Formatter() {}
        ~Formatter() {}

        template <typename Type>
        Formatter& operator << (const Type& value)
        {
            stream_ << value;
            return *this;
        }

        std::string str() const { return stream_.str(); }
        operator std::string() const { return stream_.str(); }

        enum ConvertToString
        {
            to_str
        };
        std::string operator >> (ConvertToString) { return stream_.str(); }

    private:
        std::stringstream stream_;

        Formatter(const Formatter&);
        Formatter& operator = (Formatter&);
    };
}

#endif