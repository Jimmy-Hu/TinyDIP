/* Developed by Jimmy Hu */

#ifndef BasicFunctions_H
#define BasicFunctions_H

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <concepts>
#include <deque>
#include <execution>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
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

    //  recursive_unwrap_type_t struct implementation
    template<std::size_t, typename, typename...>
    struct recursive_unwrap_type { };

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

    //  Reference: https://stackoverflow.com/a/58067611/6667035
    template <typename T>
    concept arithmetic = std::is_arithmetic_v<T>;

    constexpr bool is_integer()
    {
        return false;
    }

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

    //  recursive_depth function implementation
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
            if constexpr (std::invocable<F, T...>)
                return true;
            else
                return false;
        } else if constexpr (unwrap_level > 0) {
            return is_recursive_invocable<
                        unwrap_level - 1,
                        F,
                        std::ranges::range_value_t<T>...>();
        } else {
            return false;
        }
    }

    //  recursive_invocable concept
    template<std::size_t unwrap_level, class F, class T>
    concept recursive_invocable =
            is_recursive_invocable<unwrap_level, F, T>();

    //  is_recursive_project_invocable template function implementation
    template<std::size_t unwrap_level, class Proj, class F, class T>
    requires(unwrap_level <= recursive_depth<T>() &&
            recursive_invocable<unwrap_level, Proj, T>)
    static constexpr bool is_recursive_project_invocable()
    {
        if constexpr (unwrap_level == 0) {
            if constexpr (std::invocable<F, std::invoke_result_t<Proj, T>>)
                return true;
            else
                return false;
        } else if constexpr (unwrap_level > 0) {
            return is_recursive_project_invocable<
                        unwrap_level - 1,
                        Proj,
                        F,
                        std::ranges::range_value_t<T>>();
        } else {
            return false;
        }
    }

    //  recursive_project_invocable concept
    template<std::size_t unwrap_level, class Proj, class F, class T>
    concept recursive_project_invocable =
            is_recursive_project_invocable<unwrap_level, Proj, F, T>();

    /*  recursive_all_of template function implementation with unwrap level
    */
    template<std::size_t unwrap_level, class T, class Proj = std::identity, class UnaryPredicate>
    requires(   unwrap_level <= recursive_depth<T>() &&
                recursive_invocable<unwrap_level, Proj, T> &&
                recursive_project_invocable<unwrap_level, Proj, UnaryPredicate, T>)
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

    /*  recursive_find_if template function implementation with unwrap level
    */
    template<std::size_t unwrap_level, class T, class Proj = std::identity, class UnaryPredicate>
    requires(   unwrap_level <= recursive_depth<T>() &&
                recursive_invocable<unwrap_level, Proj, T> &&
                recursive_project_invocable<unwrap_level, Proj, UnaryPredicate, T>)
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
    requires(   unwrap_level <= recursive_depth<T>() &&
                recursive_invocable<unwrap_level, Proj, T> &&
                recursive_project_invocable<unwrap_level, Proj, UnaryPredicate, T>)
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

    template<typename... Args>
    constexpr static auto& first_of(Args&... inputs) {
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

    //  recursive_print implementation
    template<typename T>
    constexpr void recursive_print(const T& input, const std::size_t level = 0)
    {
        std::cout << std::string(level, ' ') << input << '\n';
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

    //  recursive_count implementation (the version with unwrap_level)
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

    //  recursive_reduce implementation
    //  Reference: https://codereview.stackexchange.com/a/251310/231235
    template<class T, class ValueType, class Function = std::plus<ValueType>>
    constexpr auto recursive_reduce(const T& input, ValueType init, const Function& f)
    {
        return f(init, input);
    }

    template<std::ranges::range Container, class ValueType, class Function = std::plus<ValueType>>
    constexpr auto recursive_reduce(const Container& input, ValueType init, const Function& f = std::plus<ValueType>())
    {
        for (const auto& element : input) {
            init = recursive_reduce(element, init, f);
        }
        return init;
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

    template<typename OutputIt, std::copy_constructible NAryOperation, typename InputIt, typename... InputIts>
    OutputIt transform(OutputIt d_first, NAryOperation op, InputIt first, InputIt last, InputIts... rest) {
        while (first != last) {
            *d_first++ = op(*first++, (*rest++)...);
        }
        return d_first;
    }

    //  recursive_transform for the multiple parameters cases (the version with unwrap_level)
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
        else
        {
            return std::invoke(f, arg1, args...);
        }
    }

    //  recursive_transform implementation (the version with unwrap_level, with execution policy)
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
        else
        {
            return std::invoke(f, input);
        }
    }

    #ifdef USE_BOOST_ITERATOR
    #include <boost/iterator/zip_iterator.hpp>

    //  recursive_transform for the binary operation cases (the version with unwrap_level, with execution policy)
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
            return f(input1, input2);
        }
    }
    #endif

    //  recursive_copy_if function 
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
        requires (!std::invocable<UnaryPredicate, std::ranges::range_value_t<Range>>)
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
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
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

    //  population_standard_deviation implementation
    template<class T = double, is_recursive_sizeable Container>
    requires (can_calculate_variance_of<recursive_iter_value_t<Container>>)
    constexpr auto population_standard_deviation(const Container& input)
    {
        if (recursive_size(input) == 0) //  Check the case of dividing by zero exception
        {
            throw std::logic_error("Divide by zero exception"); //  Handle the case of dividing by zero exception
        }
        return std::pow(population_variance(input), 0.5);
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
}

#endif