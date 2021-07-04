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

    //  recursive_count implementation
    template<std::ranges::input_range Range, typename T>
    constexpr auto recursive_count(const Range& input, const T& target)
    {
        return std::count(std::ranges::cbegin(input), std::ranges::cend(input), target);
    }

    //  transform_reduce version
    template<std::ranges::input_range Range, typename T>
    requires std::ranges::input_range<std::ranges::range_value_t<Range>>
    constexpr auto recursive_count(const Range& input, const T& target)
    {
        return std::transform_reduce(std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [target](auto&& element) {
            return recursive_count(element, target);
            });
    }

    //  recursive_count implementation (with execution policy)
    template<class ExPo, std::ranges::input_range Range, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr auto recursive_count(ExPo execution_policy, const Range& input, const T& target)
    {
        return std::count(execution_policy, std::ranges::cbegin(input), std::ranges::cend(input), target);
    }

    template<class ExPo, std::ranges::input_range Range, typename T>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>) && (std::ranges::input_range<std::ranges::range_value_t<Range>>)
    constexpr auto recursive_count(ExPo execution_policy, const Range& input, const T& target)
    {
        return std::transform_reduce(execution_policy, std::ranges::cbegin(input), std::ranges::cend(input), std::size_t{}, std::plus<std::size_t>(), [execution_policy, target](auto&& element) {
            return recursive_count(execution_policy, element, target);
            });
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
    auto recursive_count_if(const T& input, const Pred& predicate)
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
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1)
    {
        std::vector<decltype(recursive_count_if<unwrap_level>(input, predicate1))> output;
        output.push_back(recursive_count_if<unwrap_level>(input, predicate1));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate2));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2, class Pred3>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2, const Pred3& predicate3)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1, predicate2);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate3));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2, class Pred3, class Pred4>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2, const Pred3& predicate3, const Pred4& predicate4)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1, predicate2, predicate3);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate4));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2, class Pred3, class Pred4, class Pred5>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2, const Pred3& predicate3, const Pred4& predicate4, const Pred5& predicate5)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1, predicate2, predicate3, predicate4);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate5));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2, class Pred3, class Pred4, class Pred5, class Pred6>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2, const Pred3& predicate3, const Pred4& predicate4, const Pred5& predicate5, const Pred6& predicate6)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1, predicate2, predicate3, predicate4, predicate5);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate6));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2, class Pred3, class Pred4, class Pred5, class Pred6, class Pred7>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2, const Pred3& predicate3, const Pred4& predicate4, const Pred5& predicate5, const Pred6& predicate6, const Pred7& predicate7)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1, predicate2, predicate3, predicate4, predicate5, predicate6);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate7));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2, class Pred3, class Pred4, class Pred5, class Pred6, class Pred7, class Pred8>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2, const Pred3& predicate3, const Pred4& predicate4, const Pred5& predicate5, const Pred6& predicate6, const Pred7& predicate7, const Pred8& predicate8)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1, predicate2, predicate3, predicate4, predicate5, predicate6, predicate7);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate8));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2, class Pred3, class Pred4, class Pred5, class Pred6, class Pred7, class Pred8, class Pred9>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2, const Pred3& predicate3, const Pred4& predicate4, const Pred5& predicate5, const Pred6& predicate6, const Pred7& predicate7, const Pred8& predicate8, const Pred9& predicate9)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1, predicate2, predicate3, predicate4, predicate5, predicate6, predicate7, predicate8);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate9));
        return output;
    }

    template<std::size_t unwrap_level = 1, std::ranges::range T, class Pred1, class Pred2, class Pred3, class Pred4, class Pred5, class Pred6, class Pred7, class Pred8, class Pred9, class Pred10>
    auto batch_recursive_count_if(const T& input, const Pred1& predicate1, const Pred2& predicate2, const Pred3& predicate3, const Pred4& predicate4, const Pred5& predicate5, const Pred6& predicate6, const Pred7& predicate7, const Pred8& predicate8, const Pred9& predicate9, const Pred10& predicate10)
    {
        auto output = batch_recursive_count_if<unwrap_level>(input, predicate1, predicate2, predicate3, predicate4, predicate5, predicate6, predicate7, predicate8, predicate9);
        output.push_back(recursive_count_if<unwrap_level>(input, predicate10));
        return output;
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
    template<std::ranges::input_range Range>
    constexpr auto recursive_print(const Range& input, const int level = 0)
    {
        auto output = input;
        std::cout << std::string(level, ' ') << "Level " << level << ":" << std::endl;
        std::ranges::transform(std::ranges::cbegin(input), std::ranges::cend(input), std::ranges::begin(output),
            [level](auto&& x)
            {
                std::cout << std::string(level, ' ') << x << std::endl;
                return x;
            }
        );
        return output;
    }

    template<std::ranges::input_range Range> requires (std::ranges::input_range<std::ranges::range_value_t<Range>>)
    constexpr auto recursive_print(const Range& input, const int level = 0)
    {
        auto output = input;
        std::cout << std::string(level, ' ') << "Level " << level << ":" << std::endl;
        std::ranges::transform(std::ranges::cbegin(input), std::ranges::cend(input), std::ranges::begin(output),
            [level](auto&& element)
            {
                return recursive_print(element, level + 1);
            }
        );
        return output;
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
    template<typename, typename>
    struct recursive_invoke_result { };

    template<typename T, std::invocable<T> F>
    struct recursive_invoke_result<F, T> { using type = std::invoke_result_t<F, T>; };

    template<typename, typename, typename>
    struct recursive_invoke_result2 { };
    
    template<typename T1, typename T2, std::invocable<T1, T2> F>
    struct recursive_invoke_result2<F, T1, T2> { using type = std::invoke_result_t<F, T1, T2>; };

    template<typename F, template<typename...> typename Container, typename... Ts>
    requires (
        !std::invocable<F, Container<Ts...>>&&
        std::ranges::input_range<Container<Ts...>>&&
        requires { typename recursive_invoke_result<F, std::ranges::range_value_t<Container<Ts...>>>::type; })
        struct recursive_invoke_result<F, Container<Ts...>>
    {
        using type = Container<typename recursive_invoke_result<F, std::ranges::range_value_t<Container<Ts...>>>::type>;
    };

    template<typename F, class...Ts1, class...Ts2, template<class...>class Container1, template<class...>class Container2>
    requires (
        !std::invocable<F, Container1<Ts1...>, Container2<Ts2...>>&&
        std::ranges::input_range<Container1<Ts1...>>&&
        std::ranges::input_range<Container2<Ts2...>>&&
        requires { typename recursive_invoke_result2<F, std::ranges::range_value_t<Container1<Ts1...>>, std::ranges::range_value_t<Container2<Ts2...>>>::type; })
        struct recursive_invoke_result2<F, Container1<Ts1...>, Container2<Ts2...>>
    {
        using type = Container1<typename recursive_invoke_result2<F, std::ranges::range_value_t<Container1<Ts1...>>, std::ranges::range_value_t<Container2<Ts2...>>>::type>;
    };

    template<typename F, typename T>
    using recursive_invoke_result_t = typename recursive_invoke_result<F, T>::type;

    template<typename F, typename T1, typename T2>
    using recursive_invoke_result_t2 = typename recursive_invoke_result2<F, T1, T2>::type;

    //  recursive_transform implementation (the version with unwrap_level)
    template<std::size_t unwrap_level = 1, class T, class F>
    constexpr auto recursive_transform(const T& input, const F& f)
    {
        if constexpr (unwrap_level > 0)
        {
            recursive_invoke_result_t<F, T> output{};
            std::ranges::transform(
                std::ranges::cbegin(input),
                std::ranges::cend(input),
                std::inserter(output, std::ranges::end(output)),
                [&f](auto&& element) { return recursive_transform<unwrap_level - 1>(element, f); }
            );
            return output;
        }
        else
        {
            return f(input);
        }
    }

    //  recursive_transform implementation (the version with unwrap_level, with execution policy)
    template<std::size_t unwrap_level = 1, class ExPo, class T, class F>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr auto recursive_transform(ExPo execution_policy, const T& input, const F& f)
    {
        if constexpr (unwrap_level > 0)
        {
            recursive_invoke_result_t<F, T> output{};
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

    //  recursive_transform for the binary operation cases
    template<std::size_t unwrap_level = 1, class T1, class T2, class F>
    constexpr auto recursive_transform(const T1& input1, const T2& input2, const F& f)
    {
        if constexpr (unwrap_level > 0)
        {
            recursive_invoke_result_t2<F, T1, T2> output{};
            std::transform(
                std::ranges::cbegin(input1),
                std::ranges::cend(input1),
                std::ranges::cbegin(input2),
                std::inserter(output, std::ranges::end(output)),
                [&f](auto&& element1, auto&& element2) { return recursive_transform<unwrap_level - 1>(element1, element2, f); }
            );
            return output;
        }
        else
        {
            return f(input1, input2);
        }
    }

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