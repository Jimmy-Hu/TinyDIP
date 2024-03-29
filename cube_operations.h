/* Developed by Jimmy Hu */

#ifndef VolumetricImageOperations_H
#define VolumetricImageOperations_H

#include <concepts>
#include <execution>
#include <fstream>
#include <numbers>
#include <string>
#include "base_types.h"
#include "basic_functions.h"
#include "cube.h"
#include "image.h"

namespace TinyDIP
{
    template<typename ElementT>
    constexpr bool is_width_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        return x.getSizeX() == y.getSizeX();
    }

    template<typename ElementT>
    constexpr bool is_width_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y, const VolumetricImage<ElementT>& z)
    {
        return is_width_same(x, y) && is_width_same(y, z);
    }

    template<typename ElementT>
    constexpr bool is_height_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        return x.getSizeY() == y.getSizeY();
    }
    
    template<typename ElementT>
    constexpr bool is_height_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y, const VolumetricImage<ElementT>& z)
    {
        return is_height_same(x, y) && is_height_same(y, z);
    }

    template<typename ElementT>
    constexpr bool is_depth_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        return x.getSizeZ() == y.getSizeZ();
    }

    template<typename ElementT>
    constexpr bool is_depth_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y, const VolumetricImage<ElementT>& z)
    {
        return is_depth_same(x, y) && is_depth_same(y, z);
    }

    template<typename ElementT>
    constexpr bool is_size_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        return is_width_same(x, y) && is_height_same(x, y) && is_depth_same(x, y);
    }

    template<typename ElementT>
    constexpr bool is_size_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y, const VolumetricImage<ElementT>& z)
    {
        return is_size_same(x, y) && is_size_same(y, z);
    }

    template<typename ElementT>
    constexpr void assert_width_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        assert(is_width_same(x, y));
    }

    template<typename ElementT>
    constexpr void assert_width_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y, const VolumetricImage<ElementT>& z)
    {
        assert(is_width_same(x, y, z));
    }

    template<typename ElementT>
    constexpr void assert_height_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        assert(is_height_same(x, y));
    }

    template<typename ElementT>
    constexpr void assert_height_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y, const VolumetricImage<ElementT>& z)
    {
        assert(is_height_same(x, y, z));
    }

    template<typename ElementT>
    constexpr void assert_depth_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        assert(is_depth_same(x, y));
    }

    template<typename ElementT>
    constexpr void assert_depth_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y, const VolumetricImage<ElementT>& z)
    {
        assert(is_depth_same(x, y));
        assert(is_depth_same(y, z));
    }

    template<typename ElementT>
    constexpr void assert_size_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        assert_width_same(x, y);
        assert_height_same(x, y);
        assert_depth_same(x, y);
    }

    template<typename ElementT>
    constexpr void assert_size_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y, const VolumetricImage<ElementT>& z)
    {
        assert_size_same(x, y);
        assert_size_same(y, z);
    }

    template<typename ElementT>
    constexpr void check_width_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        if (!is_width_same(x, y))
            throw std::runtime_error("Width mismatched!");
    }

    template<typename ElementT>
    constexpr void check_height_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        if (!is_height_same(x, y))
            throw std::runtime_error("Height mismatched!");
    }

    template<typename ElementT>
    constexpr void check_depth_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        if (!is_depth_same(x, y))
            throw std::runtime_error("Depth mismatched!");
    }

    template<typename ElementT>
    constexpr void check_size_same(const VolumetricImage<ElementT>& x, const VolumetricImage<ElementT>& y)
    {
        check_width_same(x, y);
        check_height_same(x, y);
        check_depth_same(x, y);
    }

    //  voxelwiseOperation template function implementation
    template<std::size_t unwrap_level = 1, class... Args>
    constexpr static auto voxelwiseOperation(auto op, const Args&... inputs)
    {
        auto output = VolumetricImage(
            recursive_transform<unwrap_level>(
                [&](auto&& element1, auto&&... elements) 
                    {
                        return op(element1, elements...);
                    },
                inputs.getData()...),
            first_of(inputs...).getWidth(),
            first_of(inputs...).getHeight(),
            first_of(inputs...).getDepth());
        return output;
    }

    template<std::size_t unwrap_level = 1, class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static auto voxelwiseOperation(ExPo execution_policy, auto op, const Image<InputT>& input1)
    {
        auto output = VolumetricImage(
            recursive_transform<unwrap_level>(
                execution_policy,
                [&](auto&& element1) 
                    {
                        return op(element1);
                    },
                (input1.getData())),
            input1.getWidth(),
            input1.getHeight(),
            input1.getDepth());
        return output;
    }

    //  plus template function implementation
    template<class InputT>
    constexpr static VolumetricImage<InputT> plus(const VolumetricImage<InputT>& input1)
    {
        return input1;
    }

    template<class InputT, class... Args>
    constexpr static VolumetricImage<InputT> plus(const VolumetricImage<InputT>& input1, const Args&... inputs)
    {
        return voxelwiseOperation(std::plus<>{}, input1, plus(inputs...));
    }

    template<class InputT, class... Args>
    constexpr static auto plus(const std::vector<VolumetricImage<InputT>>& input1, const Args&... inputs)
    {
        return recursive_transform<1>(
            [](auto&& input1_element, auto&&... inputs_element)
            {
                return plus(input1_element, inputs_element...);
            }, input1, inputs...);
    }

    //  subtract template function implementation
    template<class InputT>
    constexpr static VolumetricImage<InputT> subtract(const VolumetricImage<InputT>& input1, const VolumetricImage<InputT>& input2)
    {
        check_size_same(input1, input2);
        return voxelwiseOperation(std::minus<>{}, input1, input2);
    }

    template<class InputT>
    constexpr static auto subtract(const std::vector<VolumetricImage<InputT>>& input1, const std::vector<VolumetricImage<InputT>>& input2)
    {
        assert(input1.size() == input2.size());
        return recursive_transform<1>(
            [](auto&& input1_element, auto&& input2_element)
            {
                return subtract(input1_element, input2_element);
            }, input1, input2);
    }

    //  multiplies template function implementation
    template<class InputT>
    constexpr static VolumetricImage<InputT> multiplies(const VolumetricImage<InputT>& input1, const VolumetricImage<InputT>& input2)
    {
        return voxelwiseOperation(std::multiplies<>{}, input1, input2);
    }

    template<class InputT, class TimesT>
    requires(std::floating_point<TimesT> || std::integral<TimesT>)
    constexpr static VolumetricImage<InputT> multiplies(const VolumetricImage<InputT>& input1, const TimesT times)
    {
        return multiplies(
            input1,
            VolumetricImage(input1.getWidth(), input1.getHeight(), input1.getDepth(), times)
        );
    }

    template<class InputT, class TimesT>
    requires(std::floating_point<TimesT> || std::integral<TimesT>)
    constexpr static VolumetricImage<InputT> multiplies(const TimesT times, const VolumetricImage<InputT>& input1)
    {
        return multiplies(input1, times);
    }

    template<class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static VolumetricImage<InputT> multiplies(ExPo execution_policy, const VolumetricImage<InputT>& input1, const VolumetricImage<InputT>& input2)
    {
        return voxelwiseOperation(execution_policy, std::multiplies<>{}, input1, input2);
    }

    template<class InputT>
    constexpr static VolumetricImage<InputT> divides(const VolumetricImage<InputT>& input1, const VolumetricImage<InputT>& input2)
    {
        return voxelwiseOperation(std::divides<>{}, input1, input2);
    }

    template<class InputT>
    constexpr static auto divides(const std::vector<VolumetricImage<InputT>>& input1, const std::vector<VolumetricImage<InputT>>& input2)
    {
        assert(input1.size() == input2.size());
        return recursive_transform<1>(
            [](auto&& input1_element, auto&& input2_element)
            {
                return divides(input1_element, input2_element);
            }, input1, input2);
    }

    template<class ExPo, class InputT>
    requires (std::is_execution_policy_v<std::remove_cvref_t<ExPo>>)
    constexpr static VolumetricImage<InputT> divides(ExPo execution_policy, const VolumetricImage<InputT>& input1, const VolumetricImage<InputT>& input2)
    {
        return voxelwiseOperation(execution_policy, std::divides<>{}, input1, input2);
    }

    template<class InputT>
    constexpr static VolumetricImage<InputT> modulus(const VolumetricImage<InputT>& input1, const VolumetricImage<InputT>& input2)
    {
        return voxelwiseOperation(std::modulus<>{}, input1, input2);
    }

    template<class InputT>
    constexpr static VolumetricImage<InputT> negate(const VolumetricImage<InputT>& input1)
    {
        return voxelwiseOperation(std::negate<>{}, input1);
    }
}

#endif
