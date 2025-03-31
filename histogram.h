#ifndef Histogram_H
#define Histogram_H

#include <chrono>
#include <iostream>
#include <map>
#ifdef __cpp_lib_print
#include <print>
#endif
#include <utility>
#include "image.h"

namespace TinyDIP
{
    template<class ElementT, class CountT = std::size_t>
    class Histogram
    {
    private:
        std::map<ElementT, CountT> histogram;
    public:
        Histogram() = default;

        Histogram(const std::map<ElementT, CountT>& input)
        {
            histogram = input;
        }

        //  Histogram constructor
        Histogram(const Image<ElementT>& input)
        {
            auto image_data = input.getImageData();
            for (std::size_t i = 0; i < image_data.size(); ++i)
            {
                addCount(image_data[i]);
            }
        }

        //  getCount member function
        constexpr std::size_t getCount(const ElementT& input)
        {
            if (auto search = histogram.find(input); search != histogram.end())
            {
                return search->second;
            }
            else
            {
                return std::size_t{ 0 };
            }
        }

        //  getCountSum member function
        constexpr std::size_t getCountSum()
        {
            std::size_t output{};
            for (const auto& [key, value] : histogram)
            {
                output += value;
            }
            return output;
        }

        //  addCount member function
        constexpr Histogram& addCount(const ElementT& input)
        {
            ++histogram[input];
            return *this;
        }

        template<class ProbabilityType = double>
        constexpr Histogram<ElementT, ProbabilityType> normalize()
        {
            std::map<ElementT, ProbabilityType> output;
            auto count_sum = static_cast<ProbabilityType>(getCountSum());
            for (const auto& [key, value] : histogram)
            {
                output.emplace(key, static_cast<ProbabilityType>(value) / count_sum);
            }
            return Histogram<ElementT, ProbabilityType>{output};
        }

        using iterator = typename std::map<ElementT, CountT>::iterator;
        using const_iterator =
            typename std::map<ElementT, CountT>::const_iterator;

        const_iterator cbegin() const { return histogram.cbegin(); }
        const_iterator cend() const { return histogram.cend(); }
        const_iterator begin() const { return histogram.cbegin(); }
        const_iterator end() const { return histogram.cend(); }
        iterator begin() { return histogram.begin(); }
        iterator end() { return histogram.end(); }

        // + operator to add two Histograms
        Histogram operator+(const Histogram& other) const
        {
            Histogram result = *this;
            for (const auto& [key, value] : other.histogram)
            {
                result.histogram[key] += value;
            }
            return result;
        }

        // - operator to subtract two Histograms
        Histogram operator-(const Histogram& other) const
        {
            Histogram result = *this;
            for (const auto& [key, value] : other.histogram) {
                if (result.histogram.contains(key)) {
                    if (result.histogram[key] >= value) {
                        result.histogram[key] -= value;
                    }
                    else {
                        result.histogram[key] = 0;
                    }
                }
            }
            return result;
        }
    };
}

#endif