/* Developed by Jimmy Hu */

#ifndef BasicFunctions_H
#define BasicFunctions_H

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
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

    #ifdef USE_BOOST_MULTIDIMENSIONAL_ARRAY
    template<typename T>
    concept is_multi_array = requires(T x)
    {
        x.num_dimensions();
        x.shape();
        boost::multi_array(x);
    };
    #endif

    template<typename T1, typename T2>
    concept is_std_powable = requires(T1 x1, T2 x2)
    {
        std::pow(x1, x2);
    };

    //  recursive_depth function implementation
    template<typename T>
    constexpr std::size_t recursive_depth()
    {
        return 0;
    }

    template<std::ranges::input_range Range>
    constexpr std::size_t recursive_depth()
    {
        return recursive_depth<std::ranges::range_value_t<Range>>() + 1;
    }

    //  recursive_count implementation

    //  recursive_count implementation (the version with unwrap_level)
    template<std::size_t unwrap_level, class T>
    constexpr auto recursive_count(const T& input, const auto& target)
    {
        if constexpr (unwrap_level > 0)
        {
            static_assert(unwrap_level <= recursive_depth<T>(),
                "unwrap level higher than recursion depth of input");
            return std::transform_reduce(std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [&target](auto&& element) {
                return recursive_count<unwrap_level - 1>(element, target);
                });
        }
        else
        {
            return (input == target) ? 1 : 0;
        }
    }

    //  recursive_count implementation (the version without unwrap_level)
    template<std::ranges::input_range Range>
    constexpr auto recursive_count(const Range& input, const auto& target)
    {
        return recursive_count<recursive_depth<Range>()>(input, target);
    }

    //  recursive_count implementation (with execution policy)
    template<class ExPo, std::ranges::input_range Range, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr auto recursive_count(ExPo execution_policy, const Range& input, const T& target)
    {
        return std::count(execution_policy, std::ranges::cbegin(input), std::ranges::cend(input), target);
    }

    //  recursive_count_if implementation
    template<class T, std::invocable<T> Pred>
    constexpr std::size_t recursive_count_if(const T& input, const Pred& predicate)
    {
        return predicate(input) ? 1 : 0;
    }

    template<std::ranges::input_range Range, class Pred>
    requires (!std::invocable<Pred, Range>)
    constexpr auto recursive_count_if(const Range& input, const Pred& predicate)
    {
        return std::transform_reduce(std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [predicate](auto&& element) {
            return recursive_count_if(element, predicate);
            });
    }

    //  recursive_count_if implementation (with execution policy)
    template<class ExPo, class T, std::invocable<T> Pred>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr std::size_t recursive_count_if(ExPo execution_policy, const T& input, const Pred& predicate)
    {
        return predicate(input) ? 1 : 0;
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
    constexpr auto batch_recursive_count_if(const T& input, const Pred1& predicate1)
    {
        std::vector<decltype(recursive_count_if<unwrap_level>(input, predicate1))> output;
        output.push_back(recursive_count_if<unwrap_level>(input, predicate1));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class... Preds>
    constexpr auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Preds&... predicates)
    {
        auto output1 = batch_recursive_count_if<unwrap_level>(input, predicate1);
        auto output2 = batch_recursive_count_if<unwrap_level>(input, predicates...);
        output1.insert(std::ranges::cend(output1), std::ranges::cbegin(output2), std::ranges::cend(output2));
        return output1;
    }

    //  recursive_max implementation
    template<std::totally_ordered T>
    constexpr auto recursive_max(T number)
    {
        return number;
    }

    template<std::ranges::range T>
    constexpr auto recursive_max(const T& numbers)
    {
        auto maxValue = recursive_max(numbers.at(0));
        for (auto& element : numbers)
        {
            maxValue = std::max(maxValue, recursive_max(element));
        }
        return maxValue;
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

    //  recursive_size implementation
    template<class T> requires (!std::ranges::range<T>)
    constexpr auto recursive_size(const T& input)
    {
        return 1;
    }

    template<std::ranges::range Range> requires (!(std::ranges::input_range<std::ranges::range_value_t<Range>>))
    constexpr auto recursive_size(const Range& input)
    {
        return std::ranges::size(input);
    }

    template<std::ranges::range Range> requires (std::ranges::input_range<std::ranges::range_value_t<Range>>)
    constexpr auto recursive_size(const Range& input)
    {
        return std::transform_reduce(std::ranges::begin(input), std::end(input), std::size_t{}, std::plus<std::size_t>(), [](auto& element) {
            return recursive_size(element);
            });
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
            auto result = recursive_reduce(element, ValueType{}, f);
            init = f(init, result);
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

    //  recursive_invoke_result_t implementation
    template<std::size_t, typename, typename>
    struct recursive_invoke_result { };

    template<typename T, typename F>
    struct recursive_invoke_result<0, F, T> { using type = std::invoke_result_t<F, T>; };

    template<std::size_t unwrap_level, typename F, template<typename...> typename Container, typename... Ts>
    requires (std::ranges::input_range<Container<Ts...>> &&
            requires { typename recursive_invoke_result<unwrap_level - 1, F, std::ranges::range_value_t<Container<Ts...>>>::type; })
    struct recursive_invoke_result<unwrap_level, F, Container<Ts...>>
    {
        using type = Container<typename recursive_invoke_result<unwrap_level - 1, F, std::ranges::range_value_t<Container<Ts...>>>::type>;
    };

    template<std::size_t unwrap_level, typename F, typename T>
    using recursive_invoke_result_t = typename recursive_invoke_result<unwrap_level, F, T>::type;

    //  recursive_variadic_invoke_result_t implementation
    template<std::size_t, typename, typename, typename...>
    struct recursive_variadic_invoke_result { };

    template<typename F, class...Ts1, template<class...>class Container1, typename... Ts>
    struct recursive_variadic_invoke_result<1, F, Container1<Ts1...>, Ts...>
    {
        using type = Container1<std::invoke_result_t<F,
            std::ranges::range_value_t<Container1<Ts1...>>,
            std::ranges::range_value_t<Ts>...>>;
    };

    template<std::size_t unwrap_level, typename F, class...Ts1, template<class...>class Container1, typename... Ts>
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

    template<std::size_t unwrap_level, typename F, typename T1, typename... Ts>
    using recursive_variadic_invoke_result_t = typename recursive_variadic_invoke_result<unwrap_level, F, T1, Ts...>::type;

    template<typename OutputIt, typename NAryOperation, typename InputIt, typename... InputIts>
    OutputIt transform(OutputIt d_first, NAryOperation op, InputIt first, InputIt last, InputIts... rest) {
        while (first != last) {
            *d_first++ = op(*first++, (*rest++)...);
        }
        return d_first;
    }

    //  recursive_transform for the multiple parameters cases (the version with unwrap_level)
    template<std::size_t unwrap_level = 1, class F, class Arg1, class... Args>
    constexpr auto recursive_transform(const F& f, const Arg1& arg1, const Args&... args)
    {
        if constexpr (unwrap_level > 0)
        {
            static_assert(unwrap_level <= recursive_depth<Arg1>(),
                "unwrap level higher than recursion depth of input");
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
            return f(arg1, args...);
        }
    }

    //  recursive_transform implementation (the version with unwrap_level, with execution policy)
    template<std::size_t unwrap_level = 1, class ExPo, class T, class F>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr auto recursive_transform(ExPo execution_policy, const F& f, const T& input)
    {
        if constexpr (unwrap_level > 0)
        {
            recursive_invoke_result_t<unwrap_level, F, T> output{};
            std::mutex mutex;

            //  Reference: https://en.cppreference.com/w/cpp/algorithm/for_each
            std::for_each(execution_policy, input.cbegin(), input.cend(),
                [&](auto&& element)
                {
                    auto result = recursive_transform<unwrap_level - 1>(execution_policy, element, f);
                    std::lock_guard lock(mutex);
                    output.emplace_back(std::move(result));
                }
            );
            
            return output;
        }
        else
        {
            return f(input);
        }
    }

    #ifdef USE_BOOST_ITERATOR
    #include <boost/iterator/zip_iterator.hpp>

    //  recursive_transform for the binary operation cases (the version with unwrap_level, with execution policy)
    template<std::size_t unwrap_level = 1, class ExPo, class T1, class T2, class F>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
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


    //  recursive_transform_reduce implementation
    template<class Input, class T, class UnaryOp, class BinaryOp = std::plus<T>>
    constexpr auto recursive_transform_reduce(const Input& input, T init, const UnaryOp& unary_op, const BinaryOp& binop = std::plus<T>())
    {
        return binop(init, unary_op(input));
    }

    template<std::ranges::range Input, class T, class UnaryOp, class BinaryOp = std::plus<T>>
    constexpr auto recursive_transform_reduce(const Input& input, T init, const UnaryOp& unary_op, const BinaryOp& binop = std::plus<T>())
    {
        return std::transform_reduce(std::ranges::begin(input), std::end(input), init, binop, [&](auto& element) {
            return recursive_transform_reduce(element, T{}, unary_op, binop);
            });
    }

    //  With execution policy
    template<class ExPo, class Input, class T, class UnaryOp, class BinaryOp = std::plus<T>>
    //requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr auto recursive_transform_reduce(ExPo execution_policy, const Input& input, T init, const UnaryOp& unary_op, const BinaryOp& binop = std::plus<T>())
    {
        return binop(init, unary_op(input));
    }

    template<class ExPo, std::ranges::range Input, class T, class UnaryOp, class BinaryOp = std::plus<T>>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr auto recursive_transform_reduce(ExPo execution_policy, const Input& input, T init, const UnaryOp& unary_op, const BinaryOp& binop = std::plus<T>())
    {
        return std::transform_reduce(execution_policy, std::ranges::begin(input), std::end(input), init, binop, [&](auto& element) {
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

    template<std::size_t dim, template<class...> class Container = std::vector, class T>
    constexpr auto n_dim_container_generator(T input, std::size_t times)
    {
        if constexpr (dim == 0)
        {
            return input;
        }
        else
        {
            return Container(times, n_dim_container_generator<dim - 1, Container, T>(input, times));
        }
    }
}

#endif