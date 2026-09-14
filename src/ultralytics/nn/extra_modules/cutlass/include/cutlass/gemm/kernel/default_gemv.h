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

#include "cutlass/gemm/threadblock/gemv.h"
#include "cutlass/gemm/threadblock/default_gemv_core.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace cutlass {
namespace gemm {
namespace kernel {


template <
        typename ThreadBlockShape_,
        typename ThreadShape_,
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementCD_,
        typename LayoutCD_,
        typename ElementAccumulator_ = ElementCD_>
struct DefaultGemv {
    using ThreadBlockShape = ThreadBlockShape_;

    using ThreadShape = ThreadShape_;

    using ElementA = ElementA_;

    using LayoutA = LayoutA_;

    using ElementB = ElementB_;

    using LayoutB = LayoutB_;

    using ElementAccumulator = ElementAccumulator_;

    using LayoutAccumulator = LayoutCD_;

    using ElementCD = ElementCD_;

    using LayoutCD = LayoutCD_;

    using Core = typename cutlass::gemm::threadblock::DefaultGemvCore<
            ThreadBlockShape, ThreadShape, ElementA, LayoutA, ElementB, LayoutB,
            ElementAccumulator, LayoutAccumulator>;

    using ThreadBlockGemv = cutlass::gemm::threadblock::Gemv<Core>;

    using IteratorA = typename ThreadBlockGemv::IteratorA;

    using IteratorB = typename ThreadBlockGemv::IteratorB;

    using IteratorPolicyCD = typename platform::conditional<
            platform::is_same<LayoutCD, layout::RowMajor>::value,
            cutlass::transform::PitchLinearTilePolicyStripminedThreadContiguous<
                    layout::PitchLinearShape<ThreadBlockShape::kN,
                                             ThreadBlockShape::kM>,
                    Core::kThreadsPerN, ThreadShape::kN>,
            cutlass::transform::PitchLinearTilePolicyStripminedThreadStrided<
                    layout::PitchLinearShape<ThreadBlockShape::kM,
                                             ThreadBlockShape::kN>,
                    Core::kThreadsPerN, ThreadShape::kM>>::type;

    using IteratorCD = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<ThreadBlockShape::kM, ThreadBlockShape::kN>,
            ElementCD, LayoutCD, 0, IteratorPolicyCD>;

    using FragmentCD = typename IteratorCD::Fragment;

    using ThreadBlockSwizzle = cutlass::gemm::threadblock::
            GemvBatchedStridedThreadblockDefaultSwizzle;
};


template <
        typename ThreadBlockShape_,
        typename ThreadShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementCD_,
        typename ElementAccumulator_>
struct DefaultGemv<ThreadBlockShape_, ThreadShape_, ElementA_,
                   cutlass::layout::RowMajor, ElementB_,
                   cutlass::layout::RowMajor, ElementCD_,
                   cutlass::layout::RowMajor, ElementAccumulator_> {
    using ThreadBlockShape = ThreadBlockShape_;

    using ThreadShape = ThreadShape_;

    using ElementA = ElementA_;

    using LayoutA = cutlass::layout::RowMajor;

    using ElementB = ElementB_;

    using LayoutB = cutlass::layout::RowMajor;

    using ElementAccumulator = ElementAccumulator_;

    using LayoutAccumulator = cutlass::layout::RowMajor;

    using ElementCD = ElementCD_;

    using LayoutCD = cutlass::layout::RowMajor;

    using Core = typename cutlass::gemm::threadblock::DefaultGemvCore<
            ThreadBlockShape, ThreadShape, ElementA, LayoutA, ElementB, LayoutB,
            ElementAccumulator, LayoutAccumulator>;

    using ThreadBlockGemv = cutlass::gemm::threadblock::GemvBatchedReduction<Core>;

    using IteratorA = typename ThreadBlockGemv::IteratorA;

    using IteratorB = typename ThreadBlockGemv::IteratorB;

    using IteratorPolicyCD =
            cutlass::transform::PitchLinearTilePolicyStripminedThreadContiguous<
                    layout::PitchLinearShape<ThreadBlockShape::kN,
                                             ThreadBlockShape::kM>,
                    Core::kThreadsPerN, ThreadShape::kN>;

    using IteratorCD = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<ThreadBlockShape::kM, ThreadBlockShape::kN>,
            ElementCD, LayoutCD, 0, IteratorPolicyCD>;

    using FragmentCD = typename IteratorCD::Fragment;

    using ThreadBlockSwizzle = cutlass::gemm::threadblock::
            GemvBatchedStridedThreadblockReductionSwizzle;
};



}
}
}
