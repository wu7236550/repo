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
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

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
struct DefaultMma;


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
        typename Operator>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            arch::OpClassSimt, 2, Operator>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
            kAlignmentA>;

    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB,
            kAlignmentB>;

    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
            layout::RowMajor, typename MmaCore::MmaPolicy>;
};


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
        typename Operator>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            arch::OpClassTensorOp, 2, Operator>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
            kAlignmentA>;

    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB,
            kAlignmentB>;

    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
            layout::RowMajor, typename MmaCore::MmaPolicy>;
};

template <
        typename LayoutA,
        int kAlignmentA,
        typename LayoutB,
        int kAlignmentB,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename Operator>
struct DefaultMma<float, LayoutA, kAlignmentA, float, LayoutB, kAlignmentB,
                  float, layout::RowMajor, arch::OpClassTensorOp, ArchTag,
                  ThreadblockShape, WarpShape, InstructionShape, 2, Operator,
                  false> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, float, LayoutA,
            float, LayoutB, float, layout::RowMajor, arch::OpClassTensorOp, 2,
            arch::OpMultiplyAddFastF16>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, float,
            LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, float,
            LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, float, layout::RowMajor,
            typename MmaCore::MmaPolicy>;
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
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename Operator,
        int InterleavedK>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator,
                  layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
                  ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2,
                  Operator, true> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator,
            layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, 2,
            Operator, true>;

    static_assert(kAlignmentA == 128 / sizeof_bits<ElementA>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentB == 128 / sizeof_bits<ElementB>::value,
                  "Alignment must match thread data map's vector length");

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB>;

    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
            layout::ColumnMajorInterleaved<InterleavedK>,
            typename MmaCore::MmaPolicy>;
};


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
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            arch::OpClassSimt, Stages, Operator>;

    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
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
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false> {
    static cutlass::arch::CacheOperation::Kind const CacheOpA =
            ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
                    ? cutlass::arch::CacheOperation::Global
                    : cutlass::arch::CacheOperation::Always;

    static cutlass::arch::CacheOperation::Kind const CacheOpB =
            ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
                    ? cutlass::arch::CacheOperation::Global
                    : cutlass::arch::CacheOperation::Always;

    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            arch::OpClassTensorOp, Stages, Operator, false, CacheOpA, CacheOpB>;

    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
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
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        int Stages,
        typename Operator,
        int InterleavedK>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator,
                  layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
                  ArchTag, ThreadblockShape, WarpShape, InstructionShape,
                  Stages, Operator, true> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator,
            layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, Stages,
            Operator, true>;

    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
    using IteratorA =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
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


template <
        typename LayoutA,
        int kAlignmentA,
        typename LayoutB,
        int kAlignmentB,
        typename ElementAccumulator,
        typename ArchTag,
        typename ThreadblockShape,
        typename Operator,
        typename WarpShape>
struct DefaultMma<int8_t, LayoutA, kAlignmentA, int8_t, LayoutB, kAlignmentB,
                  ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
                  ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 4>, 2,
                  Operator, false> {
    using InstructionShape = GemmShape<1, 1, 4>;
    using ElementA = int8_t;
    using ElementB = int8_t;
    using OperatorClass = arch::OpClassSimt;

    static const bool transposeA =
            cutlass::platform::is_same<LayoutA, layout::ColumnMajor>::value;
    static const bool transposeB =
            cutlass::platform::is_same<LayoutB, layout::RowMajor>::value;

    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            OperatorClass, 2, Operator>;

    using IteratorA =
            cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
                    cutlass::MatrixShape<MmaCore::Shape::kM,
                                         MmaCore::Shape::kK>,
                    ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
                    transposeA>;

    using IteratorB =
            cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB,
                    transposeB>;

    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
            layout::RowMajor, typename MmaCore::MmaPolicy>;
};


#if defined(CUTLASS_ARCH_WMMA_ENABLED)
template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementAccumulator,
        typename LayoutC,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename Operator>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape,
                  WarpShape, InstructionShape, 2, Operator> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, LayoutC,
            arch::OpClassWmmaTensorOp, 2, Operator>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
            kAlignmentA>;

    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB,
            kAlignmentB>;

    using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
            LayoutC, typename MmaCore::MmaPolicy>;
};


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementAccumulator,
        typename LayoutC,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename Operator>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape,
                  WarpShape, InstructionShape, 1, Operator> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, LayoutC,
            arch::OpClassWmmaTensorOp, 1, Operator>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
            kAlignmentA>;

    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB,
            kAlignmentB>;

    using ThreadblockMma = cutlass::gemm::threadblock::MmaSingleStage<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
            LayoutC, typename MmaCore::MmaPolicy>;
};

#endif

}
}
}

