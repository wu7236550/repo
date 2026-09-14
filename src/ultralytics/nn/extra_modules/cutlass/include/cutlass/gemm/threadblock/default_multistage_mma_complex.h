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

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementAccumulator_,
        typename LayoutC_,
        typename OperatorClass_,
        typename ArchTag_,
        typename ThreadblockShape_,
        typename WarpShape_,
        typename InstructionShape_,
        int Stages,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone,
        typename Operator = arch::OpMultiplyAddComplex,
        bool AccumulatorsInRowMajor = false>
struct DefaultMultistageMmaComplex;


template <
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator>
struct DefaultMultistageMmaComplex<
        ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
        layout::RowMajor, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
        InstructionShape, Stages, TransformA, TransformB, Operator> {
    using MmaCore = typename cutlass::gemm::threadblock::
            DefaultMultistageMmaComplexCore<
                    ThreadblockShape, WarpShape, InstructionShape, ElementA,
                    LayoutA, ElementB, LayoutB, ElementAccumulator,
                    layout::RowMajor, OperatorClass, Stages, TransformA,
                    TransformB, Operator>;

    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA =
            cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
    using IteratorA =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB =
            cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
    using IteratorB =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kK,
                                         ThreadblockShape::kN>,
                    ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
            MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
            typename MmaCore::MmaPolicy, Stages>;
};


}
}
}

