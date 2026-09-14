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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/complex.h"
#include "cutlass/array_planar_complex.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"


namespace cutlass {
namespace epilogue {
namespace thread {


template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_ =
                  ElementOutput_,
          typename ElementCompute_ =
                  ElementOutput_,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class LinearCombinationPlanarComplex {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;

    static int const kCount = Count;

    using FragmentOutput = ArrayPlanarComplex<ElementOutput, kCount>;
    using FragmentAccumulator = ArrayPlanarComplex<ElementAccumulator, kCount>;
    using ComputeFragment = ArrayPlanarComplex<ElementCompute, kCount>;

    static FloatRoundStyle const kRound = Round;

    struct Params {
        complex<ElementCompute> alpha;
        complex<ElementCompute> beta;
        complex<ElementCompute> const*
                alpha_ptr;
        complex<ElementCompute> const*
                beta_ptr;


        CUTLASS_HOST_DEVICE
        Params()
                : alpha(ElementCompute(1)),
                  beta(ElementCompute(0)),
                  alpha_ptr(nullptr),
                  beta_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(complex<ElementCompute> alpha, complex<ElementCompute> beta)
                : alpha(alpha),
                  beta(beta),
                  alpha_ptr(nullptr),
                  beta_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(complex<ElementCompute> const* alpha_ptr,
               complex<ElementCompute> const* beta_ptr)
                : alpha(complex<ElementCompute>()),
                  beta(complex<ElementCompute>()),
                  alpha_ptr(alpha_ptr),
                  beta_ptr(beta_ptr) {}
    };

private:

    complex<ElementCompute> alpha_;
    complex<ElementCompute> beta_;

public:
    CUTLASS_HOST_DEVICE
    LinearCombinationPlanarComplex(Params const& params) {
        alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
        beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    }

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const {
        return beta_.real() != ElementCompute(0) ||
               beta_.imag() != ElementCompute(0);
    }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int k_partition_count) {
        if (k_partition) {
            beta_ = ElementCompute(1);
        }
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator,
                              FragmentOutput const& source) const {
        NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
                source_converter;
        NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
                accumulator_converter;

        ComputeFragment converted_source(source_converter(source.real),
                                         source_converter(source.imag));

        ComputeFragment converted_accumulator(
                accumulator_converter(accumulator.real),
                accumulator_converter(accumulator.imag));

        ComputeFragment intermediate;

        multiplies<Array<ElementCompute, kCount> > mul_op;
        multiply_add<Array<ElementCompute, kCount> > mul_add_op;

        intermediate.real = mul_op(beta_.real(), converted_source.real);
        intermediate.imag = mul_op(beta_.real(), converted_source.imag);

        intermediate.real = mul_add_op(-beta_.imag(), converted_source.imag,
                                       intermediate.real);
        intermediate.imag = mul_add_op(beta_.imag(), converted_source.real,
                                       intermediate.imag);

        intermediate.real = mul_add_op(
                alpha_.real(), converted_accumulator.real, intermediate.real);
        intermediate.imag = mul_add_op(
                alpha_.real(), converted_accumulator.imag, intermediate.imag);

        intermediate.real = mul_add_op(
                -alpha_.imag(), converted_accumulator.imag, intermediate.real);
        intermediate.imag = mul_add_op(
                alpha_.imag(), converted_accumulator.real, intermediate.imag);

        NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
                destination_converter;

        return FragmentOutput(destination_converter(intermediate.real),
                              destination_converter(intermediate.imag));
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
        NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
                accumulator_converter;

        ComputeFragment converted_accumulator(
                accumulator_converter(accumulator.real),
                accumulator_converter(accumulator.imag));

        ComputeFragment intermediate;

        multiplies<Array<ElementCompute, kCount> > mul_op;
        multiply_add<Array<ElementCompute, kCount> > mul_add_op;

        intermediate.real =
                mul_add_op(alpha_.real(), converted_accumulator.real);
        intermediate.imag =
                mul_add_op(alpha_.real(), converted_accumulator.imag);

        intermediate.real = mul_add_op(
                -alpha_.imag(), converted_accumulator.imag, intermediate.real);
        intermediate.imag = mul_add_op(
                alpha_.imag(), converted_accumulator.real, intermediate.imag);

        NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
                destination_converter;

        return FragmentOutput(destination_converter(intermediate.real),
                              destination_converter(intermediate.imag));
    }
};


}
}
}

