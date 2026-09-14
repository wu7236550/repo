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
#include "cutlass/cutlass.h"

#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/default_mma_sparse_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

#include "cutlass/gemm/threadblock/default_mma_core.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/gemm/threadblock/mma_sparse_multistage.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape,
        typename WarpShape,
        typename InstructionShape,
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename LayoutC,
        typename OperatorClass,
        int Stages,
        typename Operator = typename platform::conditional<
                (platform::is_same<OperatorClass,
                                   cutlass::arch::OpClassTensorOp>::value) &&
                        (platform::is_same<ElementA, int8_t>::value ||
                         platform::is_same<ElementA, int4b_t>::value ||
                         platform::is_same<ElementA, uint8_t>::value ||
                         platform::is_same<ElementA, uint4b_t>::value),
                cutlass::arch::OpMultiplyAddSaturate,
                cutlass::arch::OpMultiplyAdd>::type,
        bool AccumulatorsInRowMajor = false
        ,
        cutlass::arch::CacheOperation::Kind CacheOpA =
                cutlass::arch::CacheOperation::Global,
        cutlass::arch::CacheOperation::Kind CacheOpB =
                cutlass::arch::CacheOperation::Global>
struct DefaultSparseMmaCore;


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                            layout::ColumnMajor, ElementB_, layout::RowMajor,
                            ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                            Operator_, false, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

    static int const kSparse = 2;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    using Operator = Operator_;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
            sizeof_bits<ElementA>::value, int(128 / sizeof(ElementA))>;

    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
            sizeof_bits<ElementB>::value, int(128 / sizeof(ElementB))>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK / kSparse>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA,
            1, IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    static cutlass::arch::CacheOperation::Kind const kCacheOpE =
            cutlass::arch::CacheOperation::Global;

    static int const kInterleavedE = MmaTensorOp::kInterleaved;
    static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
    static int const kMaxID2 = MmaTensorOp::kMaxID2;
    static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

    using ElementE = typename MmaTensorOp::ElementE;
    using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

    using SmemLayoutE = typename MmaTensorOp::LayoutE;

    static int const kElementsPerAccessE =
            kAccessSizeInBits / sizeof_bits<ElementE>::value;

    static int const kThreadsE =
            (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                     (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
             kThreads)
                    ? kThreads
                    : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                       (kAccessSizeInBits / sizeof_bits<ElementE>::value));

    using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<
                    Shape::kM * kInterleavedE,
                    Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
            kThreadsE, kElementsPerAccessE>;

    using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM * kInterleavedE,
                        Shape::kK / kSparse / kElementsPerElementE /
                                kInterleavedE>,
            ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

    using MmaPolicy =
            SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                            MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                            layout::RowMajor, ElementB_, layout::ColumnMajor,
                            ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                            Operator_, false, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

    static int const kSparse = 2;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    using Operator = Operator_;

    static int const kWarpThreadArrangementContiguousA =
            Shape::kK / kSparse /
            (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;

    static int const kCrosswiseB =
            (Shape::kK > (1024 / sizeof_bits<ElementB>::value))
                    ? (1024 / sizeof_bits<ElementB>::value)
                    : Shape::kK;

    static int const kWarpThreadArrangementContiguousB =
            kCrosswiseB / (kAccessSizeInBits / sizeof_bits<ElementB>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementA>::value, Shape::kK / kSparse>;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementB>::value, kCrosswiseB>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK / kSparse, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA,
            0, IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    static cutlass::arch::CacheOperation::Kind const kCacheOpE =
            cutlass::arch::CacheOperation::Global;

    static int const kInterleavedE = MmaTensorOp::kInterleaved;
    static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
    static int const kMaxID2 = MmaTensorOp::kMaxID2;
    static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

    using ElementE = typename MmaTensorOp::ElementE;
    using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

    using SmemLayoutE = typename MmaTensorOp::LayoutE;

    static int const kElementsPerAccessE =
            kAccessSizeInBits / sizeof_bits<ElementE>::value;

    static int const kThreadsE =
            (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                     (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
             kThreads)
                    ? kThreads
                    : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                       (kAccessSizeInBits / sizeof_bits<ElementE>::value));

    using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<
                    Shape::kM * kInterleavedE,
                    Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
            kThreadsE, kElementsPerAccessE>;

    using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM * kInterleavedE,
                        Shape::kK / kSparse / kElementsPerElementE /
                                kInterleavedE>,
            ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

    using MmaPolicy =
            SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                            MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                            layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                            ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                            Operator_, false, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;

    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::ColumnMajor;

    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

    static int const kSparse = 2;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    using Operator = Operator_;

    static int const kCrosswiseB =
            (Shape::kK > (1024 / sizeof_bits<ElementB>::value))
                    ? (1024 / sizeof_bits<ElementB>::value)
                    : Shape::kK;

    static int const kWarpThreadArrangementContiguousB =
            kCrosswiseB / (kAccessSizeInBits / sizeof_bits<ElementB>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
            sizeof_bits<ElementA>::value, int(128 / sizeof(ElementA))>;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementB>::value, kCrosswiseB>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK / kSparse>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA,
            1, IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    static cutlass::arch::CacheOperation::Kind const kCacheOpE =
            cutlass::arch::CacheOperation::Global;

    static int const kInterleavedE = MmaTensorOp::kInterleaved;
    static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
    static int const kMaxID2 = MmaTensorOp::kMaxID2;
    static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

    using ElementE = typename MmaTensorOp::ElementE;
    using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

    using SmemLayoutE = typename MmaTensorOp::LayoutE;

    static int const kElementsPerAccessE =
            kAccessSizeInBits / sizeof_bits<ElementE>::value;

    static int const kThreadsE =
            (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                     (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
             kThreads)
                    ? kThreads
                    : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                       (kAccessSizeInBits / sizeof_bits<ElementE>::value));

    using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<
                    Shape::kM * kInterleavedE,
                    Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
            kThreadsE, kElementsPerAccessE>;

    using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM * kInterleavedE,
                        Shape::kK / kSparse / kElementsPerElementE /
                                kInterleavedE>,
            ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

    using MmaPolicy =
            SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                            MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                            layout::RowMajor, ElementB_, layout::RowMajor,
                            ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                            Operator_, false, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

    static int const kSparse = 2;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    using Operator = Operator_;

    static int const kWarpThreadArrangementContiguousA =
            Shape::kK / kSparse /
            (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementA>::value, Shape::kK / kSparse>;

    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
            sizeof_bits<ElementB>::value, int(128 / sizeof(ElementB))>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK / kSparse, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA,
            0, IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    static cutlass::arch::CacheOperation::Kind const kCacheOpE =
            cutlass::arch::CacheOperation::Global;

    static int const kInterleavedE = MmaTensorOp::kInterleaved;
    static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
    static int const kMaxID2 = MmaTensorOp::kMaxID2;
    static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

    using ElementE = typename MmaTensorOp::ElementE;
    using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

    using SmemLayoutE = typename MmaTensorOp::LayoutE;

    static int const kElementsPerAccessE =
            kAccessSizeInBits / sizeof_bits<ElementE>::value;

    static int const kThreadsE =
            (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                     (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
             kThreads)
                    ? kThreads
                    : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
                       (kAccessSizeInBits / sizeof_bits<ElementE>::value));

    using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<
                    Shape::kM * kInterleavedE,
                    Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
            kThreadsE, kElementsPerAccessE>;

    using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM * kInterleavedE,
                        Shape::kK / kSparse / kElementsPerElementE /
                                kInterleavedE>,
            ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

    using MmaPolicy =
            SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                            MatrixShape<0, 0>, WarpCount::kK>;
};


}
}
}
