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
#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma_core_sparse_sm80.h"
#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/gemm/threadblock/default_mma_core_wmma.h"
#endif


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename ElementA_,
        typename LayoutA_,
        int kAlignmentA,
        typename ElementB_,
        typename LayoutB_,
        int kAlignmentB,
        typename ElementAccumulator_,
        typename LayoutC_,
        typename OperatorClass_,
        typename ArchTag_,
        typename ThreadblockShape_,
        typename WarpShape_,
        typename InstructionShape_,
        int Stages,
        typename Operator,
        bool AccumulatorsInRowMajor = false>
struct DefaultSparseMma;


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementAccumulator,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        int Stages,
        typename Operator>
struct DefaultSparseMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                        kAlignmentB, ElementAccumulator, layout::RowMajor,
                        arch::OpClassTensorOp, ArchTag, ThreadblockShape,
                        WarpShape, InstructionShape, Stages, Operator, false> {
    static cutlass::arch::CacheOperation::Kind const CacheOpA =
            ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
                    ? cutlass::arch::CacheOperation::Global
                    : cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB =
            ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
                    ? cutlass::arch::CacheOperation::Global
                    : cutlass::arch::CacheOperation::Always;

    using MmaCore = typename cutlass::gemm::threadblock::DefaultSparseMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            arch::OpClassTensorOp, Stages, Operator, false, CacheOpA, CacheOpB>;

    static int const kSparse = MmaCore::kSparse;

    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK / kSparse>,
                    ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kK,
                                         ThreadblockShape::kN>,
                    ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

    using ElementE = typename MmaCore::ElementE;
    using LayoutE = typename MmaCore::GmemLayoutE;
    using ThreadMapE = typename MmaCore::IteratorThreadMapE;
    using AccessTypeE =
            cutlass::Array<ElementE, 128 / sizeof_bits<ElementE>::value>;
    using IteratorE =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK / kSparse /
                                                 MmaCore::kElementsPerElementE>,
                    ElementE, LayoutE, 1, ThreadMapE, AccessTypeE>;

    using ThreadblockMma = cutlass::gemm::threadblock::SparseMmaMultistage<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
            MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor, IteratorE,
            typename MmaCore::SmemIteratorE, MmaCore::kCacheOpE,
            typename MmaCore::MmaPolicy, Stages>;
};


}
}
}

