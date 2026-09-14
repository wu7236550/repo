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
#include "cutlass/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include "cutlass/wmma_array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_wmma.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename Shape_,
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_,
        typename Policy_,
        int PartitionsK_ = 1,
        typename Enable = bool>
class MmaTensorOpWmma {
public:
    using Shape = Shape_;

    using ElementA = ElementA_;

    using LayoutA = LayoutA_;

    using ElementB = ElementB_;

    using LayoutB = LayoutB_;

    using ElementC = ElementC_;

    using LayoutC = LayoutC_;

    using Policy = Policy_;

    using InstructionShape = typename Policy::Operator::Shape;

    using ArchTag = typename Policy::Operator::ArchTag;

    static ComplexTransform const kTransformA = ComplexTransform::kNone;

    static ComplexTransform const kTransformB = ComplexTransform::kNone;

    using OperatorClass = arch::OpClassWmmaTensorOp;

    static int const kThreadCount = 32;

    static int const kPartitionsK = PartitionsK_;

public:
    using IteratorA = MmaTensorOpWmmaMultiplicandTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
            Policy::OpDelta::kRow, kThreadCount, Policy>;

    using FragmentA = typename IteratorA::Fragment;

    using IteratorB = MmaTensorOpWmmaMultiplicandTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
            Policy::OpDelta::kRow, kThreadCount, Policy>;

    using FragmentB = typename IteratorB::Fragment;

    using IteratorC = MmaTensorOpWmmaAccumulatorTileIterator<
            MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
            typename Policy::OpDelta, Policy>;

    using FragmentC = typename IteratorC::Fragment;

private:
    static_assert(!(Shape::kM % Policy::Operator::Shape::kM) &&
                          !(Shape::kN % Policy::Operator::Shape::kN),
                  "Shape of warp-level Wmma must be divisible by operator "
                  "shape (wmma native size)");

    using WmmaIterations = MatrixShape<Shape::kM / Policy::Operator::Shape::kM,
                                       Shape::kN / Policy::Operator::Shape::kN>;

public:
    typename Policy::Operator wmma;

public:

    CUTLASS_DEVICE
    MmaTensorOpWmma() {}

    CUTLASS_DEVICE
    void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B,
                    FragmentC const& C) const {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < WmmaIterations::kColumn; ++n) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < WmmaIterations::kRow; ++m) {
                wmma(D[m * WmmaIterations::kColumn + n], A[m], B[n],
                     C[m * WmmaIterations::kColumn + n]);
            }
        }
    }
};


}
}
}

#endif
