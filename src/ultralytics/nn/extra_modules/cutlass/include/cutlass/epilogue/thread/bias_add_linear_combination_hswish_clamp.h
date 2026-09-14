/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file
 * include/cutlass/epilogue/thread/bias_add_linear_combination_hswish_clamp.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/epilogue.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/numeric_array_converter_policy.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"


namespace cutlass {
namespace epilogue {
namespace thread {


template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_ = ElementOutput_,
          typename ElementBias_ = ElementOutput_,

          typename ElementCompute_ = ElementOutput_,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest_integer,
          typename Policy = NumericArrayConverterPolicy<
                  ElementOutput_, Count, ElementAccumulator_, ElementBias_,
                  ElementCompute_, Round>>
class BiasAddLinearCombinationHSwishClamp {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementBias = ElementBias_;
    using ElementCompute = ElementCompute_;

    static int const kCount = Count;

    static EpilogueType const kType =
            EpilogueType::kBiasAddLinearCombinationHSwishClamp;

    using FragmentOutput = Array<ElementOutput, kCount>;
    using FragmentAccumulator = Array<ElementAccumulator, kCount>;
    using FragmentBias = Array<ElementBias, kCount>;
    using ComputeFragment = Array<ElementCompute, kCount>;
    using ComputeFragmentBias = Array<ElementCompute, kCount>;
    using SourceConverter = typename Policy::SourceConverter;
    using AccumulatorConverter = typename Policy::AccumulatorConverter;
    using BiasConverter = typename Policy::BiasConverter;
    using OutputConverter = typename Policy::OutputConverter;

    static FloatRoundStyle const kRound = Round;

    struct Params {
        ElementCompute alpha;
        ElementCompute beta;
        ElementCompute gamma;
        ElementCompute delta;
        ElementCompute theta;
        ElementCompute scale;
        ElementCompute const* alpha_ptr;
        ElementCompute const* beta_ptr;
        ElementCompute const* gamma_ptr;
        ElementCompute const* delta_ptr;
        ElementCompute const* theta_ptr;
        ElementCompute const* scale_ptr;


        CUTLASS_HOST_DEVICE
        Params()
                : alpha(ElementCompute(1)),
                  beta(ElementCompute(1)),
                  gamma(ElementCompute(0)),
                  delta(ElementCompute(0)),
                  theta(ElementCompute(0)),
                  scale(ElementCompute(1)),
                  alpha_ptr(nullptr),
                  beta_ptr(nullptr),
                  gamma_ptr(nullptr),
                  delta_ptr(nullptr),
                  theta_ptr(nullptr),
                  scale_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute alpha, ElementCompute beta, ElementCompute gamma,
               ElementCompute scale, ElementCompute delta = ElementCompute(0),
               ElementCompute theta = ElementCompute(0))
                : alpha(alpha),
                  beta(beta),
                  gamma(gamma),
                  delta(delta),
                  theta(theta),
                  scale(scale),
                  alpha_ptr(nullptr),
                  beta_ptr(nullptr),
                  gamma_ptr(nullptr),
                  delta_ptr(nullptr),
                  theta_ptr(nullptr),
                  scale_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr,
               ElementCompute const* gamma_ptr, ElementCompute const* scale_ptr,
               ElementCompute const* delta_ptr = nullptr,
               ElementCompute const* theta_ptr = nullptr)
                : alpha(0),
                  beta(0),
                  gamma(0),
                  delta(0),
                  theta(0),
                  scale(0),
                  alpha_ptr(alpha_ptr),
                  beta_ptr(beta_ptr),
                  gamma_ptr(gamma_ptr),
                  delta_ptr(delta_ptr),
                  theta_ptr(theta_ptr),
                  scale_ptr(scale_ptr) {}
    };

private:

    ElementCompute alpha_;
    ElementCompute beta_;
    ElementCompute gamma_;
    ElementCompute delta_;
    ElementCompute theta_;
    ElementCompute scale_;
    ElementCompute inv_scale_;

public:
    CUTLASS_HOST_DEVICE
    BiasAddLinearCombinationHSwishClamp(Params const& params) {
        alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
        beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
        gamma_ = (params.gamma_ptr ? *params.gamma_ptr : params.gamma);
        delta_ = (params.delta_ptr ? *params.delta_ptr : params.delta);
        theta_ = (params.theta_ptr ? *params.theta_ptr : params.theta);
        scale_ = (params.scale_ptr ? *params.scale_ptr : params.scale);
        inv_scale_ = ElementCompute(1.f / scale_);
    }

    CUTLASS_HOST_DEVICE
    bool is_bias_needed() const { return beta_ != ElementCompute(0); }

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return gamma_ != ElementCompute(0); }

    CUTLASS_HOST_DEVICE
    FragmentOutput apply_add_bias_source(FragmentAccumulator const& accumulator,
                                         FragmentBias const& bias,
                                         FragmentOutput const& source) const {
        SourceConverter source_converter;
        AccumulatorConverter accumulator_converter;
        BiasConverter bias_converter;

        ComputeFragment converted_source = source_converter(source);
        ComputeFragment converted_accumulator =
                accumulator_converter(accumulator);
        ComputeFragmentBias converted_bias = bias_converter(bias);


        ComputeFragment intermediate;

        multiplies<ComputeFragment> mul_add_source;
        multiply_add<ComputeFragment> mul_add_accumulator;
        multiply_add<ComputeFragmentBias> mul_add_bias;
        plus<ComputeFragment> plus_delta;
        plus<ComputeFragment> plus_theta;
        HSwish<ComputeFragment> hswish;

        minimum<ComputeFragment> min_accumulator;
        maximum<ComputeFragment> max_accumulator;

        intermediate =
                mul_add_source(gamma_, converted_source);
        intermediate =
                mul_add_accumulator(alpha_, converted_accumulator,
                                    intermediate);
        intermediate = mul_add_bias(beta_, converted_bias,
                                    intermediate);
        intermediate = plus_delta(delta_, intermediate);

        intermediate =
                hswish(scale_, inv_scale_, intermediate);
        intermediate = plus_theta(theta_, intermediate);

        ElementCompute const kClampMax =
                ElementCompute(platform::numeric_limits<ElementOutput>::max());

        ElementCompute const kClampMin = ElementCompute(
                platform::numeric_limits<ElementOutput>::lowest());

        intermediate = max_accumulator(intermediate, kClampMin);
        intermediate = min_accumulator(intermediate, kClampMax);

        OutputConverter destination_converter;

        return destination_converter(intermediate);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput apply_add_bias(FragmentAccumulator const& accumulator,
                                  FragmentBias const& bias) const {
        AccumulatorConverter accumulator_converter;
        BiasConverter bias_converter;

        ComputeFragment converted_accumulator =
                accumulator_converter(accumulator);
        ComputeFragmentBias converted_bias = bias_converter(bias);


        ComputeFragment intermediate;

        multiplies<ComputeFragment> mul_accumulator;
        multiply_add<ComputeFragmentBias> mul_add_bias;
        plus<ComputeFragment> plus_delta;
        plus<ComputeFragment> plus_theta;
        HSwish<ComputeFragment> hswish;

        minimum<ComputeFragment> min_accumulator;
        maximum<ComputeFragment> max_accumulator;

        intermediate = mul_accumulator(
                alpha_, converted_accumulator);
        intermediate = mul_add_bias(beta_, converted_bias,
                                    intermediate);
        intermediate = plus_delta(delta_, intermediate);

        intermediate =
                hswish(scale_, inv_scale_, intermediate);
        intermediate = plus_theta(theta_, intermediate);

        ElementCompute const kClampMax =
                ElementCompute(platform::numeric_limits<ElementOutput>::max());

        ElementCompute const kClampMin = ElementCompute(
                platform::numeric_limits<ElementOutput>::lowest());

        intermediate = max_accumulator(intermediate, kClampMin);
        intermediate = min_accumulator(intermediate, kClampMax);

        OutputConverter destination_converter;

        return destination_converter(intermediate);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput apply_add_source(FragmentAccumulator const& accumulator,
                                    FragmentOutput const& source) const {
        SourceConverter source_converter;
        AccumulatorConverter accumulator_converter;

        ComputeFragment converted_source = source_converter(source);
        ComputeFragment converted_accumulator =
                accumulator_converter(accumulator);


        ComputeFragment intermediate;

        multiplies<ComputeFragment> mul_add_source;
        multiply_add<ComputeFragment> mul_add_accumulator;
        plus<ComputeFragment> plus_delta;
        plus<ComputeFragment> plus_theta;
        HSwish<ComputeFragment> hswish;

        minimum<ComputeFragment> min_accumulator;
        maximum<ComputeFragment> max_accumulator;

        intermediate =
                mul_add_source(gamma_, converted_source);
        intermediate =
                mul_add_accumulator(alpha_, converted_accumulator,
                                    intermediate);
        intermediate = plus_delta(delta_, intermediate);

        intermediate =
                hswish(scale_, inv_scale_, intermediate);
        intermediate = plus_theta(theta_, intermediate);

        ElementCompute const kClampMax =
                ElementCompute(platform::numeric_limits<ElementOutput>::max());

        ElementCompute const kClampMin = ElementCompute(
                platform::numeric_limits<ElementOutput>::lowest());

        intermediate = max_accumulator(intermediate, kClampMin);
        intermediate = min_accumulator(intermediate, kClampMax);

        OutputConverter destination_converter;

        return destination_converter(intermediate);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput apply(FragmentAccumulator const& accumulator) const {
        AccumulatorConverter accumulator_converter;

        ComputeFragment converted_accumulator =
                accumulator_converter(accumulator);

        ComputeFragment intermediate;

        multiplies<ComputeFragment> mul_add_source;
        plus<ComputeFragment> plus_delta;
        plus<ComputeFragment> plus_theta;
        HSwish<ComputeFragment> hswish;

        minimum<ComputeFragment> min_accumulator;
        maximum<ComputeFragment> max_accumulator;

        intermediate = mul_add_source(alpha_,
                                      converted_accumulator);
        intermediate = plus_delta(delta_, intermediate);

        intermediate =
                hswish(scale_, inv_scale_, intermediate);
        intermediate = plus_theta(theta_, intermediate);

        ElementCompute const kClampMax =
                ElementCompute(platform::numeric_limits<ElementOutput>::max());

        ElementCompute const kClampMin = ElementCompute(
                platform::numeric_limits<ElementOutput>::lowest());

        intermediate = max_accumulator(intermediate, kClampMin);
        intermediate = min_accumulator(intermediate, kClampMax);

        OutputConverter destination_converter;

        return destination_converter(intermediate);
    }
};


template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementBias_, typename ElementCompute_,
          FloatRoundStyle Round>
using FastBiasAddLinearCombinationHSwishClamp =
        BiasAddLinearCombinationHSwishClamp<
                ElementOutput_, Count, ElementAccumulator_, ElementBias_,
                ElementCompute_, Round,
                FastNumericArrayConverterPolicy<
                        ElementOutput_, Count, ElementAccumulator_,
                        ElementBias_, ElementCompute_, Round>>;

}
}
}
