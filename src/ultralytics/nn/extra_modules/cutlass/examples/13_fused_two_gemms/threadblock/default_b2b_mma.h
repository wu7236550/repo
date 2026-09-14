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

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"

#include "threadblock/b2b_mma_pipelined.h"
#include "threadblock/b2b_mma_multistage.h"


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
        typename ThreadblockShape0_,
        typename ThreadblockShape1_,
        typename WarpShape0_,
        typename WarpShape1_,
        typename InstructionShape_,
        int Stages,
        typename Operator,
        typename EpilogueOutputOp,
        bool AccumulatorsInRowMajor = false>
struct DefaultB2bMma;

template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape0,
        typename ThreadblockShape1,
        typename WarpShape0,
        typename WarpShape1,
        typename InstructionShape,
        typename Operator,
        typename EpilogueOutputOp>
struct DefaultB2bMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                     kAlignmentB, ElementAccumulator, layout::RowMajor,
                     OperatorClass, ArchTag, ThreadblockShape0,
                     ThreadblockShape1, WarpShape0, WarpShape1,
                     InstructionShape, 2, Operator, EpilogueOutputOp, false> {
    using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape0, WarpShape0, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            OperatorClass, 2, Operator>;
    using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape1, WarpShape1, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            OperatorClass, 2, Operator>;

    using IteratorA0 = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore0::Shape::kM, MmaCore0::Shape::kK>,
            ElementA, LayoutA, 1, typename MmaCore0::IteratorThreadMapA,
            kAlignmentA>;

    using IteratorB0 = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore0::Shape::kK, MmaCore0::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore0::IteratorThreadMapB,
            kAlignmentB>;

    using AccumulatorLayout = cutlass::layout::ColumnMajor;
    using FragmentIteratorA1 = cutlass::gemm::warp::MmaTensorOpFragmentIterator<
            cutlass::MatrixShape<MmaCore1::WarpShape::kM,
                                 MmaCore1::InstructionShape::kK>,
            cutlass::MatrixShape<MmaCore0::WarpShape::kM,
                                 MmaCore0::WarpShape::kN>,
            MmaCore1::Shape::kK,
            ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape,
            EpilogueOutputOp, true>;

    using IteratorB1 = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore1::Shape::kK, MmaCore1::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore1::IteratorThreadMapB>;

    using ThreadblockB2bMma = cutlass::gemm::threadblock::B2bMmaPipelined<
            typename MmaCore0::Shape, IteratorA0,
            typename MmaCore0::SmemIteratorA, IteratorB0,
            typename MmaCore0::SmemIteratorB, typename MmaCore1::Shape,
            FragmentIteratorA1, IteratorB1, typename MmaCore1::SmemIteratorB,
            ElementAccumulator, layout::RowMajor, EpilogueOutputOp,
            typename MmaCore0::MmaPolicy, typename MmaCore1::MmaPolicy>;
};

template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ThreadblockShape0,
        typename ThreadblockShape1,
        typename WarpShape0,
        typename WarpShape1,
        typename InstructionShape,
        typename Operator,
        typename EpilogueOutputOp,
        int InterleavedK>
struct DefaultB2bMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                     kAlignmentB, ElementAccumulator,
                     layout::ColumnMajorInterleaved<InterleavedK>,
                     OperatorClass, arch::Sm75, ThreadblockShape0,
                     ThreadblockShape1, WarpShape0, WarpShape1,
                     InstructionShape, 2, Operator, EpilogueOutputOp, true> {
    using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape0, WarpShape0, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator,
            layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, 2,
            Operator, true>;
    using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape1, WarpShape1, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator,
            layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, 2,
            Operator, true>;

    static_assert(kAlignmentA == 128 / sizeof_bits<ElementA>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentB == 128 / sizeof_bits<ElementB>::value,
                  "Alignment must match thread data map's vector length");

    using IteratorA0 = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore0::Shape::kM, MmaCore0::Shape::kK>,
            ElementA, LayoutA, 1, typename MmaCore0::IteratorThreadMapA>;

    using IteratorB0 = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore0::Shape::kK, MmaCore0::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore0::IteratorThreadMapB>;

    using AccumulatorLayout =
            cutlass::layout::RowMajor;
    using FragmentIteratorA1 = cutlass::gemm::warp::MmaTensorOpFragmentIterator<
            cutlass::MatrixShape<MmaCore1::WarpShape::kM,
                                 MmaCore1::InstructionShape::kK>,
            cutlass::MatrixShape<MmaCore0::WarpShape::kM,
                                 MmaCore0::WarpShape::kN>,
            MmaCore1::Shape::kK,
            ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape,
            EpilogueOutputOp,
            true >;

    using IteratorB1 = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore1::Shape::kK, MmaCore1::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore1::IteratorThreadMapB>;

    using ThreadblockB2bMma = cutlass::gemm::threadblock::B2bMmaPipelined<
            typename MmaCore0::Shape, IteratorA0,
            typename MmaCore0::SmemIteratorA, IteratorB0,
            typename MmaCore0::SmemIteratorB, typename MmaCore1::Shape,
            FragmentIteratorA1, IteratorB1, typename MmaCore1::SmemIteratorB,
            ElementAccumulator, layout::ColumnMajorInterleaved<InterleavedK>,
            EpilogueOutputOp, typename MmaCore0::MmaPolicy,
            typename MmaCore1::MmaPolicy>;
};


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape0,
        typename ThreadblockShape1,
        typename WarpShape0,
        typename WarpShape1,
        typename InstructionShape,
        int Stages,
        typename Operator,
        typename EpilogueOutputOp,
        int InterleavedK>
struct DefaultB2bMma<
        ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
        ElementAccumulator, layout::ColumnMajorInterleaved<InterleavedK>,
        OperatorClass, ArchTag, ThreadblockShape0, ThreadblockShape1,
        WarpShape0, WarpShape1, InstructionShape, Stages, Operator,
        EpilogueOutputOp, true> {
    using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape0, WarpShape0, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator,
            layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, Stages,
            Operator, true>;
    using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape1, WarpShape1, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator,
            layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, Stages,
            Operator, true>;

    using ThreadMapA0 = typename MmaCore0::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA0 =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape0::kM,
                                         ThreadblockShape0::kK>,
                    ElementA, LayoutA, 1, ThreadMapA0, AccessTypeA>;

    using ThreadMapB0 = typename MmaCore0::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
    using IteratorB0 =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape1::kK,
                                         ThreadblockShape1::kN>,
                    ElementB, LayoutB, 0, ThreadMapB0, AccessTypeB>;

    using AccumulatorLayout =
            cutlass::layout::RowMajor;
    using FragmentIteratorA1 = cutlass::gemm::warp::MmaTensorOpFragmentIterator<
            cutlass::MatrixShape<MmaCore1::WarpShape::kM,
                                 MmaCore1::InstructionShape::kK>,
            cutlass::MatrixShape<MmaCore0::WarpShape::kM,
                                 MmaCore0::WarpShape::kN>,
            MmaCore1::Shape::kK,
            ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape,
            EpilogueOutputOp,
            true >;

    using ThreadMapB1 = typename MmaCore1::IteratorThreadMapB;
    using IteratorB1 =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape1::kK,
                                         ThreadblockShape1::kN>,
                    ElementB, LayoutB, 0, ThreadMapB1, AccessTypeB>;

    using ThreadblockB2bMma = cutlass::gemm::threadblock::B2bMmaMultistage<
            typename MmaCore0::Shape, IteratorA0,
            typename MmaCore0::SmemIteratorA, MmaCore0::kCacheOpA, IteratorB0,
            typename MmaCore0::SmemIteratorB, MmaCore0::kCacheOpB,
            typename MmaCore1::Shape, FragmentIteratorA1, IteratorB1,
            typename MmaCore1::SmemIteratorB, MmaCore1::kCacheOpB,
            ElementAccumulator, layout::ColumnMajorInterleaved<InterleavedK>,
            EpilogueOutputOp, typename MmaCore0::MmaPolicy,
            typename MmaCore1::MmaPolicy, Stages>;
};


}
}
}

