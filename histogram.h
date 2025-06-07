#ifndef Histogram_H
#define Histogram_H

#include <chrono>
#include <execution>
#include <iostream>
#include <map>
#ifdef __cpp_lib_print
#include <print>
#endif
#include <utility>
#include <type_traits>
#include <variant>
#include <vector>
#include "image.h"

namespace TinyDIP
{
    template<class ElementT, class CountT = std::size_t>
    class Histogram
    {
    private:
        std::variant<std::vector<CountT>, std::map<ElementT, CountT>> histogram;
    public:
        // Histogram constructor
        // Explicitly initialize based on ElementT type
        Histogram()
        {
            if constexpr ((std::same_as<ElementT, std::uint8_t>) or
                          (std::same_as<ElementT, std::uint16_t>))
            {
                histogram.template emplace<std::vector<CountT>>(std::numeric_limits<ElementT>::max() + 1, 0);
            }
            else
            {
                histogram.template emplace<std::map<ElementT, CountT>>();
            }
        }

        Histogram(const std::map<ElementT, CountT>& input)
        {
            histogram = input;
        }

        Histogram(const std::vector<CountT>& input)
        {
            histogram = input;
        }

        //  Histogram constructor
        Histogram(const Image<ElementT>& input)
        {
            if constexpr ((std::same_as<ElementT, std::uint8_t>) or
                          (std::same_as<ElementT, std::uint16_t>))
            {
                histogram.template emplace<std::vector<CountT>>(std::numeric_limits<ElementT>::max() + 1, 0);
            }
            else
            {
                histogram.template emplace<std::map<ElementT, CountT>>();
            }
            auto image_data = input.getImageData();
            for (std::size_t i = 0; i < image_data.size(); ++i)
            {
                addCount(image_data[i]);
            }
        }

        //  getCount member function implementation
        constexpr auto getCount(const ElementT& input) const
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                return std::get<std::vector<CountT>>(histogram).at(input);
            }
            else
            {
                if (auto search = std::get<std::map<ElementT, CountT>>(histogram).find(input);
                    search != std::get<std::map<ElementT, CountT>>(histogram).end())
                {
                    return search->second;
                }
                else
                {
                    return CountT{ 0 };
                }
            }
        }

        //  getCountSum member function with execution policy
        template<class ExecutionPolicy>
        requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
        constexpr auto getCountSum(ExecutionPolicy&& execution_policy) const
        {
            CountT output{};
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                return std::reduce(
                    std::forward<ExecutionPolicy>(execution_policy),
                    std::ranges::cbegin(get_result),
                    std::ranges::cend(get_result),
                    output);
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                for (const auto& [key, value] : get_result)
                {
                    output += value;
                }
                return output;
            }
        }

        //  getCountSum member function without execution policy
        constexpr auto getCountSum() const
        {
            return getCountSum(std::execution::seq);
        }

        //  addCount member function
        constexpr Histogram& addCount(const ElementT& input)
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                ++get_result[input];
                histogram = get_result;
                return *this;
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                ++get_result[input];
                histogram = get_result;
                return *this;
            }
            
        }

        //  normalize Template Function Implementation with Execution Policy
        template<class ExecutionPolicy, class ProbabilityType = double>
        requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
        constexpr auto normalize(ExecutionPolicy&& execution_policy)
        {
            auto count_sum = static_cast<ProbabilityType>(getCountSum(std::forward<ExecutionPolicy>(execution_policy)));
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                std::vector<ProbabilityType> output(std::numeric_limits<ElementT>::max() + 1);
                const auto& get_result = std::get<std::vector<CountT>>(histogram);
                for (std::size_t i = 0; i < get_result.size(); ++i)
                {
                    output[i] = static_cast<ProbabilityType>(get_result[i]) / count_sum;
                }
                return Histogram<ElementT, ProbabilityType>{output};
            }
            else
            {
                std::map<ElementT, ProbabilityType> output;
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                for (const auto& [key, value] : get_result)
                {
                    output.emplace(key, static_cast<ProbabilityType>(value) / count_sum);
                }
                return Histogram<ElementT, ProbabilityType>{output};
            }
        }

        //  normalize Template Function Implementation without Execution Policy
        template<class ProbabilityType = double>
        constexpr auto normalize()
        {
            return normalize<const std::execution::sequenced_policy, ProbabilityType>(std::move(std::execution::seq));
        }


        constexpr auto size() const
        {
            if constexpr (  std::same_as<ElementT, std::uint8_t> ||
                            std::same_as<ElementT, std::uint16_t>)
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.size();
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.size();
            }
        }

        //  to_probabilities_vector member function with execution policy
        template<class ExecutionPolicy, class FloatingType = double>
        requires((std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>) and
                 (std::floating_point<FloatingType>))
        constexpr std::vector<FloatingType> to_probabilities_vector(ExecutionPolicy&& execution_policy) const
        {
            std::vector<FloatingType> probabilities;
            if constexpr (  std::same_as<ElementT, std::uint8_t> ||
                            std::same_as<ElementT, std::uint16_t>)
            {
                probabilities.resize(std::numeric_limits<ElementT>::max() + 1);
                std::size_t total_count = getCountSum(std::forward<ExecutionPolicy>(execution_policy));
                for (std::size_t i = 0; i < probabilities.size(); ++i)
                {
                    probabilities[i] =
                        static_cast<FloatingType>(getCount(static_cast<ElementT>(i))) /
                        static_cast<FloatingType>(total_count);
                }
            }
            else
            {
                std::vector<std::pair<ElementT, double>> probability_vector(size());
                auto total_count = getCountSum(std::forward<ExecutionPolicy>(execution_policy));
                for (const auto& [key, value] : *this)
                {
                    probability_vector.emplace_back(
                        { key, static_cast<FloatingType>(value) / static_cast<FloatingType>(total_count) });
                }
                std::sort(std::forward<ExecutionPolicy>(execution_policy), probability_vector.begin(), probability_vector.end(),
                    [&](const auto& a, const auto& b) { return a.first < b.first; });
                probabilities.resize(probability_vector.back().first + 1, 0.0);
                for (const auto& pair : probability_vector)
                {
                    probabilities[pair.first] = pair.second;
                }
            }
            return probabilities;
        }

        //  to_probabilities_vector member function without execution policy
        template<class FloatingType = double>
        requires(std::floating_point<FloatingType>)
        constexpr std::vector<FloatingType> to_probabilities_vector() const
        {
            return to_probabilities_vector(std::execution::seq);
        }

        auto cbegin() const 
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.cbegin();
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.cbegin();
            }
        }
        auto cend() const
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.cend();
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.cend();
            }
        }
        auto begin() const
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.cbegin();
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.cbegin();
            }
        }
        auto end() const
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.cend();
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.cend();
            }
        }
        auto begin()
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.begin();
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.begin();
            }
        }
        auto end()
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.end();
            }
            else
            {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.end();
            }
        }

        // += operator to add two Histograms
        Histogram& operator+=(const Histogram& other) const
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                const auto& get_result_other = std::get<const std::vector<CountT>>(other.histogram);
                if (get_result.size() != get_result_other.size())
                {
                    throw std::runtime_error("Size mismatched for vector-based histograms!");
                }
                for (std::size_t i = 0; i < get_result.size(); i++)
                {
                    get_result[i] += get_result_other[i];
                }
                histogram = get_result;
                
            }
            else
            {
                auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                const auto get_result_other = std::get<const std::map<ElementT, CountT>>(other.histogram);
                for (const auto& [key, value] : get_result_other)
                {
                    get_result[key] += value;
                }
                histogram = get_result;
            }
            return *this;
        }

        // operator+ (using operator+=)
        Histogram operator+(const Histogram& other) const
        {
            Histogram result = *this;
            result += other;
            return result;
        }

        // -= operator to subtract two Histograms
        Histogram& operator-=(const Histogram& other) const
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) ||
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                const auto& get_result_other = std::get<const std::vector<CountT>>(other.histogram);
                if (get_result.size() != get_result_other.size())
                {
                    throw std::runtime_error("Size mismatched!");
                }
                std::vector<CountT> result_data = get_result;
                for (std::size_t i = 0; i < result_data.size(); i++)
                {
                    if (result_data[i] >= get_result_other[i])
                    {
                        result_data[i] -= get_result_other[i];
                    }
                    else
                    {
                        result_data[i] = 0;
                    }
                }
                return Histogram<ElementT, CountT>{ result_data };
            }
            else
            {
                auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                const auto& get_result_other = std::get<const std::map<ElementT, CountT>>(other.histogram);
                std::map<ElementT, CountT> result_data = get_result;
                for (const auto& [key, value] : get_result_other)
                {
                    if (result_data.contains(key))
                    {
                        if (result_data.at(key) >= value)
                        {
                            result_data[key] -= value;
                        }
                        else
                        {
                            result_data[key] = 0;
                        }
                    }
                }
                return Histogram<ElementT, CountT>{ result_data };
            }
        }

        // operator- (using operator-=)
        Histogram operator-(const Histogram& other) const
        {
            Histogram result = *this;
            result -= other;
            return result;
        }

        // operator[] to access and modify counts
        CountT& operator[](const ElementT& key)
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) ||
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                if (static_cast<std::size_t>(key) >=
                    get_result.size())
                {
                    std::cerr << "key = " << +key << '\n';
                    throw std::out_of_range("Index out of range");
                }
                return get_result[static_cast<std::size_t>(key)];
            }
            else
            {
                return std::get<std::map<ElementT, CountT>>(histogram)[key];
            }
        }

        // const operator[] Implementation
        const CountT& operator[](const ElementT& key) const
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or
                            (std::same_as<ElementT, std::uint16_t>))
            {
                const auto& get_result = std::get<std::vector<CountT>>(histogram);
                if (static_cast<std::size_t>(key) >= get_result.size())
                {
                    static const CountT zero_count = 0;
                    return zero_count;
                }
                return get_result[static_cast<std::size_t>(key)];
            }
            else
            {
                const auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                if (auto it = get_result.find(key); it != get_result.end())
                {
                    return it->second;
                }
                else
                {
                    static const CountT zero_count = 0;
                    return zero_count;
                }
            }
        }
        
    };
}

#endif