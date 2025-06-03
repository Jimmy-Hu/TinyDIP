#ifndef Histogram_H
#define Histogram_H

#include <chrono>
#include <iostream>
#include <map>
#ifdef __cpp_lib_print
#include <print>
#endif
#include <utility>
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
        Histogram() = default;

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
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                histogram = std::vector<CountT>(std::numeric_limits<ElementT>::max() + 1);
            }
            auto image_data = input.getImageData();
            for (std::size_t i = 0; i < image_data.size(); ++i)
            {
                addCount(image_data[i]);
            }
        }

        //  getCount member function
        constexpr std::size_t getCount(const ElementT& input) const
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
                    return std::size_t{ 0 };
                }
            }
        }

        //  getCountSum member function
        constexpr std::size_t getCountSum() const
        {
            std::size_t output{};
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                for (std::size_t i = 0; i < get_result.size(); ++i)
                {
                    output += get_result[i];
                }
                return output;
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

        template<class ProbabilityType = double>
        constexpr Histogram<ElementT, ProbabilityType> normalize()
        {
            auto count_sum = static_cast<ProbabilityType>(getCountSum());
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) or 
                            (std::same_as<ElementT, std::uint16_t>))
            {
                std::vector<ProbabilityType> output(std::numeric_limits<ElementT>::max() + 1);
                auto get_result = std::get<std::vector<CountT>>(histogram);
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
                std::size_t total_count = std::accumulate(
                    std::get<std::vector<CountT>>(histogram).begin(),
                    std::get<std::vector<CountT>>(histogram).end(), 0ULL);
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
                auto total_count = getCountSum();
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

        // - operator to subtract two Histograms
        Histogram operator-(const Histogram& other) const
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) ||
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                auto get_result_other = std::get<std::vector<CountT>>(other.histogram);
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
            else {
                auto get_result = std::get<std::map<ElementT, CountT>>(histogram);
                auto get_result_other = std::get<std::map<ElementT, CountT>>(other.histogram);
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

        // operator[] to access and modify counts
        CountT& operator[](const ElementT& key)
        {
            if constexpr (  (std::same_as<ElementT, std::uint8_t>) ||
                            (std::same_as<ElementT, std::uint16_t>))
            {
                auto get_result = std::get<std::vector<CountT>>(histogram);
                if (static_cast<std::size_t>(key) >=
                    get_result.size())
                {
                    std::cerr << "key = " << +key << '\n';
                    throw std::out_of_range("Index out of range");
                }
                return get_result[key];
            }
            else
            {
                return std::get<std::map<ElementT, CountT>>(histogram)[key];
            }
        }
        
    };
}

#endif