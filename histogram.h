#ifndef Histogram_H
#define Histogram_H

#include <chrono>
#include <iostream>
#include <map>
#include <print>
#include <utility>

namespace TinyDIP
{
    template<class ElementT>
    class Histogram
    {
    private:
        std::map<ElementT, std::size_t> histogram;
    public:
        Histogram() = default;

        Histogram(const std::map<ElementT, std::size_t>& input)
        {
            histogram = input;
        }

        constexpr std::size_t getCount(const ElementT& input)
        {
            if (histogram.contains(input))
            {
                return histogram.at(input);
            }
            else
            {
                return std::size_t{ 0 };
            }
        }

        constexpr Histogram& addCount(const ElementT& input)
        {
            if (histogram.contains(input))
            {
                ++histogram[input];
            }
            else
            {
                histogram.emplace(input, std::size_t{ 1 });
            }
            return *this;
        }

        using iterator = typename std::map<ElementT, std::size_t>::iterator;
        using const_iterator =
            typename std::map<ElementT, std::size_t>::const_iterator;

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