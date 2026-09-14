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

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/epilogue/warp/tensor_op_policy.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape,
          typename OperatorShape,
          typename OperatorElementC,
          typename OperatorFragmentC,
          typename Layout
          >
class FragmentIteratorGaussianComplexTensorOp;


template <typename WarpShape_,
          typename OperatorShape_,
          typename OperatorElementC_,
          typename OperatorFragmentC_
          >
class FragmentIteratorGaussianComplexTensorOp<
        WarpShape_, OperatorShape_, OperatorElementC_, OperatorFragmentC_,
        layout::RowMajor> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using OperatorElementC = OperatorElementC_;
    using OperatorFragmentC = OperatorFragmentC_;
    using Layout = layout::RowMajor;

    using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Fragment =
            Array<complex<OperatorElementC>,
                  Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

    static int const kElementsAccumulatorPerPart =
            OperatorFragmentC::kElements * Policy::OperatorCount::kRow *
            Policy::OperatorCount::kColumn;

    static int const kPart1Index = kElementsAccumulatorPerPart * 0;

    static int const kPart2Index = kElementsAccumulatorPerPart * 1;

    static int const kPart3Index = kElementsAccumulatorPerPart * 2;

    using AccumulatorTile =
            Array<OperatorElementC, kElementsAccumulatorPerPart * 3>;

    using OutputAccumulatorTile =
            Array<complex<OperatorElementC>, kElementsAccumulatorPerPart>;

    static int const kIterations = Policy::kIterations;

private:
    using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;

    using FragmentAccessType =
            Array<complex<OperatorElementC>, Policy::kElementsPerAccess>;

private:

    AccessType const* accumulators_;

    int index_;

public:
    CUTLASS_HOST_DEVICE
    FragmentIteratorGaussianComplexTensorOp(AccumulatorTile const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0) {}

    CUTLASS_HOST_DEVICE
    FragmentIteratorGaussianComplexTensorOp& operator++() {
        ++index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    FragmentIteratorGaussianComplexTensorOp& operator--() {
        --index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, int index_offset = 0) const {
        int index = index_ + index_offset;

        FragmentAccessType* frag_ptr =
                reinterpret_cast<FragmentAccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            int accumulator_access_offset =
                    index + n * Policy::kAccumulatorColumnStride /
                                    Policy::kElementsPerAccess;

            auto const& part1_accum_array =
                    accumulators_[accumulator_access_offset + kPart1Index];
            auto const& part2_accum_array =
                    accumulators_[accumulator_access_offset +
                                  kPart2Index / Policy::kElementsPerAccess];
            auto const& part3_accum_array =
                    accumulators_[accumulator_access_offset +
                                  kPart3Index / Policy::kElementsPerAccess];

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Policy::kElementsPerAccess; ++i) {
                frag_ptr[n][i].real() =
                        part1_accum_array[i] - part3_accum_array[i];
                frag_ptr[n][i].imag() =
                        part1_accum_array[i] + part2_accum_array[i];
            }
        }
    }
};


}
}
}

