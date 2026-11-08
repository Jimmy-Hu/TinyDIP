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
    // Concept checking if a multi-channel pixel struct has totally ordered channel elements
    template<class T>
    concept multi_channel_totally_ordered = requires(const T& a)
    {
        a.channels[0];
        std::ranges::size(a.channels);
        requires std::totally_ordered<std::remove_cvref_t<decltype(a.channels[0])>>;
    };

    template<class T>
    concept histogram_element = std::totally_ordered<T> || multi_channel_totally_ordered<T>;

    // Comparator supporting both scalar totally_ordered types and multi-channel pixel types
    template<class ElementT>
    requires histogram_element<ElementT>
    struct HistogramKeyLess
    {
        constexpr bool operator()(const ElementT& a, const ElementT& b) const noexcept
        {
            if constexpr (std::totally_ordered<ElementT>)
            {
                return a < b;
            }
            else if constexpr (multi_channel_totally_ordered<ElementT>)
            {
                return std::ranges::lexicographical_compare(a.channels, b.channels);
            }
        }
    };

    template<class ElementT, class CountT = std::size_t>
    requires(histogram_element<ElementT>)
    class Histogram
    {
    private:
        std::variant<std::vector<CountT>, std::map<ElementT, CountT>> histogram;
        static constexpr bool is_dense = (std::same_as<ElementT, std::uint8_t>) or
                                         (std::same_as<ElementT, std::uint16_t>);
    public:
        static constexpr bool is_dense_v = is_dense;

        // Histogram constructor
        // Explicitly initialize based on ElementT type
        Histogram()
        {
            if constexpr (is_dense)
            {
                histogram.template emplace<std::vector<CountT>>(
                    static_cast<std::size_t>(std::numeric_limits<ElementT>::max()) + 1, CountT{0});
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

        //  Histogram constructor from Image
        Histogram(const Image<ElementT>& input)
        {
            if constexpr (is_dense)
            {
                histogram.template emplace<std::vector<CountT>>(
                    static_cast<std::size_t>(std::numeric_limits<ElementT>::max()) + 1, CountT{0});
            }
            else
            {
                histogram.template emplace<std::map<ElementT, CountT>>();
            }
            const auto& image_data = input.getImageData();
            for (std::size_t i = 0; i < image_data.size(); ++i)
            {
                addCount(image_data[i]);
            }
        }

        //  getCount member function implementation
        constexpr auto getCount(const ElementT& input) const
        {
            if constexpr (is_dense)
            {
                return std::get<std::vector<CountT>>(histogram).at(static_cast<std::size_t>(input));
            }
            else
            {
                const auto& map_ref = std::get<std::map<ElementT, CountT>>(histogram);
                if (auto search = map_ref.find(input); search != map_ref.end())
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
            if constexpr (is_dense)
            {
                const auto& data = std::get<std::vector<CountT>>(histogram);
                return std::reduce(
                    std::forward<ExecutionPolicy>(execution_policy),
                    std::ranges::cbegin(data),
                    std::ranges::cend(data),
                    output);
            }
            else
            {
                const auto& map_ref = std::get<std::map<ElementT, CountT>>(histogram);
                auto values_view = map_ref | std::views::values;
                return std::reduce(
                    std::forward<ExecutionPolicy>(execution_policy),
                    std::ranges::cbegin(values_view),
                    std::ranges::cend(values_view),
                    output);
            }
        }

        //  getCountSum member function without execution policy
        constexpr auto getCountSum() const
        {
            return getCountSum(std::execution::seq);
        }

        //  addCount member function (In-place O(1) update)
        constexpr Histogram& addCount(const ElementT& input)
        {
            if constexpr (is_dense)
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                ++get_result[static_cast<std::size_t>(input)];
                return *this;
            }
            else
            {
                auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                ++get_result[input];
                return *this;
            }
        }

        //  normalize Template Function Implementation with Execution Policy
        template<class ExecutionPolicy, class ProbabilityType = double>
        requires(std::is_execution_policy_v<std::remove_cvref_t<ExecutionPolicy>>)
        constexpr auto normalize(ExecutionPolicy&& execution_policy) const
        {
            auto count_sum = static_cast<ProbabilityType>(getCountSum(std::forward<ExecutionPolicy>(execution_policy)));
            if constexpr (is_dense)
            {
                std::vector<ProbabilityType> output(static_cast<std::size_t>(std::numeric_limits<ElementT>::max()) + 1);
                const auto& get_result = std::get<std::vector<CountT>>(histogram);
                for (std::size_t i = 0; i < get_result.size(); ++i)
                {
                    output[i] = (count_sum != ProbabilityType{0})
                        ? static_cast<ProbabilityType>(get_result[i]) / count_sum
                        : ProbabilityType{0};
                }
                return Histogram<ElementT, ProbabilityType>{output};
            }
            else
            {
                std::map<ElementT, ProbabilityType> output;
                const auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                for (const auto& [key, value] : get_result)
                {
                    output.emplace(key, (count_sum != ProbabilityType{0})
                        ? static_cast<ProbabilityType>(value) / count_sum
                        : ProbabilityType{0});
                }
                return Histogram<ElementT, ProbabilityType>{output};
            }
        }

        //  normalize Template Function Implementation without Execution Policy
        template<class ProbabilityType = double>
        constexpr auto normalize() const
        {
            return normalize<const std::execution::sequenced_policy&, ProbabilityType>(std::execution::seq);
        }

        constexpr auto size() const
        {
            if constexpr (is_dense)
            {
                const auto& get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.size();
            }
            else
            {
                const auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
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
            if constexpr (is_dense)
            {
                probabilities.resize(static_cast<std::size_t>(std::numeric_limits<ElementT>::max()) + 1);
                auto total_count = getCountSum(std::forward<ExecutionPolicy>(execution_policy));
                for (std::size_t i = 0; i < probabilities.size(); ++i)
                {
                    probabilities[i] = (total_count != CountT{0})
                        ? static_cast<FloatingType>(getCount(static_cast<ElementT>(i))) / static_cast<FloatingType>(total_count)
                        : FloatingType{0};
                }
            }
            else
            {
                std::vector<std::pair<ElementT, FloatingType>> probability_vector;
                probability_vector.reserve(size());
                auto total_count = getCountSum(std::forward<ExecutionPolicy>(execution_policy));
                for (const auto& [key, value] : *this)
                {
                    probability_vector.emplace_back(
                        key,
                        (total_count != CountT{0})
                            ? static_cast<FloatingType>(value) / static_cast<FloatingType>(total_count)
                            : FloatingType{0});
                }
                if (!probability_vector.empty())
                {
                    std::sort(std::forward<ExecutionPolicy>(execution_policy), probability_vector.begin(), probability_vector.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; });
                    probabilities.resize( static_cast<std::size_t>(probability_vector.back().first) + 1, FloatingType{0});
                    for (const auto& pair : probability_vector)
                    {
                        probabilities[static_cast<std::size_t>(pair.first)] = pair.second;
                    }
                }
            }
            return probabilities;
        }

        //  to_probabilities_vector member function without execution policy
        template<class FloatingType = double>
        requires(std::floating_point<FloatingType>)
        constexpr std::vector<FloatingType> to_probabilities_vector() const
        {
            return to_probabilities_vector<const std::execution::sequenced_policy&, FloatingType>(std::execution::seq);
        }

        auto cbegin() const
        {
            if constexpr (is_dense)
            {
                const auto& get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.cbegin();
            }
            else
            {
                const auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.cbegin();
            }
        }

        auto cend() const
        {
            if constexpr (is_dense)
            {
                const auto& get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.cend();
            }
            else
            {
                const auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.cend();
            }
        }

        auto begin() const
        {
            if constexpr (is_dense)
            {
                const auto& get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.cbegin();
            }
            else
            {
                const auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.cbegin();
            }
        }

        auto end() const
        {
            if constexpr (is_dense)
            {
                const auto& get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.cend();
            }
            else
            {
                const auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.cend();
            }
        }

        auto begin()
        {
            if constexpr (is_dense)
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.begin();
            }
            else
            {
                auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.begin();
            }
        }

        auto end()
        {
            if constexpr (is_dense)
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                return get_result.end();
            }
            else
            {
                auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                return get_result.end();
            }
        }

        // += operator to add two Histograms
        Histogram& operator+=(const Histogram& other)
        {
            if constexpr (is_dense)
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                const auto& get_result_other = std::get<std::vector<CountT>>(other.histogram);
                if (get_result.size() != get_result_other.size())
                {
                    throw std::runtime_error("Size mismatched for vector-based histograms!");
                }
                for (std::size_t i = 0; i < get_result.size(); ++i)
                {
                    get_result[i] += get_result_other[i];
                }
            }
            else
            {
                auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                const auto& get_result_other = std::get<std::map<ElementT, CountT>>(other.histogram);
                for (const auto& [key, value] : get_result_other)
                {
                    get_result[key] += value;
                }
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
        Histogram& operator-=(const Histogram& other)
        {
            if constexpr (is_dense)
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                const auto& get_result_other = std::get<std::vector<CountT>>(other.histogram);
                if (get_result.size() != get_result_other.size())
                {
                    throw std::runtime_error("Size mismatched!");
                }
                for (std::size_t i = 0; i < get_result.size(); ++i)
                {
                    if (get_result[i] >= get_result_other[i])
                    {
                        get_result[i] -= get_result_other[i];
                    }
                    else
                    {
                        get_result[i] = 0;
                    }
                }
            }
            else
            {
                auto& get_result = std::get<std::map<ElementT, CountT>>(histogram);
                const auto& get_result_other = std::get<std::map<ElementT, CountT>>(other.histogram);
                for (const auto& [key, value] : get_result_other)
                {
                    if (get_result.contains(key))
                    {
                        if (get_result.at(key) >= value)
                        {
                            get_result[key] -= value;
                        }
                        else
                        {
                            get_result[key] = 0;
                        }
                    }
                }
            }
            return *this;
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
            if constexpr (is_dense)
            {
                auto& get_result = std::get<std::vector<CountT>>(histogram);
                if (static_cast<std::size_t>(key) >= get_result.size())
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
            if constexpr (is_dense)
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