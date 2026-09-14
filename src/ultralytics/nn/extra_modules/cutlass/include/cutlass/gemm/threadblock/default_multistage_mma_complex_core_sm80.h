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
#include "cutlass/gemm/warp/default_mma_complex_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<8, 8, 4>, complex<double>,
        layout::ColumnMajor, complex<double>, layout::RowMajor, complex<double>,
        LayoutC_, arch::OpClassTensorOp, Stages, TransformA, TransformB,
        Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = complex<double>;
    using LayoutA = layout::ColumnMajor;
    using ElementB = complex<double>;
    using LayoutB = layout::RowMajor;
    using ElementC = complex<double>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous128b;

    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous128b;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, kTransformA, kTransformB,
            Operator>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<8, 8, 4>, complex<double>,
        layout::ColumnMajor, complex<double>, layout::ColumnMajor,
        complex<double>, LayoutC_, arch::OpClassTensorOp, Stages, TransformA,
        TransformB, Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = complex<double>;
    using LayoutA = layout::ColumnMajor;
    using ElementB = complex<double>;
    using LayoutB = layout::ColumnMajor;
    using ElementC = complex<double>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    using Operator = Operator_;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous128b;
    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise128x4;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, kTransformA, kTransformB,
            Operator>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<8, 8, 4>, complex<double>,
        layout::RowMajor, complex<double>, layout::ColumnMajor, complex<double>,
        LayoutC_, arch::OpClassTensorOp, Stages, TransformA, TransformB,
        Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = complex<double>;
    using LayoutA = layout::RowMajor;
    using ElementB = complex<double>;
    using LayoutB = layout::ColumnMajor;
    using ElementC = complex<double>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise128x4;
    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise128x4;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, kTransformA, kTransformB,
            Operator>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<8, 8, 4>, complex<double>,
        layout::RowMajor, complex<double>, layout::RowMajor, complex<double>,
        LayoutC_, arch::OpClassTensorOp, Stages, TransformA, TransformB,
        Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = complex<double>;
    using LayoutA = layout::RowMajor;
    using ElementB = complex<double>;
    using LayoutB = layout::RowMajor;
    using ElementC = complex<double>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise128x4;
    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous128b;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, kTransformA, kTransformB,
            Operator>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<16, 8, 8>, complex<float>,
        layout::ColumnMajor, complex<float>, layout::ColumnMajor,
        complex<float>, LayoutC_, arch::OpClassTensorOp, Stages, TransformA,
        TransformB, Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<16, 8, 8>;
    using ElementA = complex<float>;
    using LayoutA = layout::ColumnMajor;
    using ElementB = complex<float>;
    using LayoutB = layout::ColumnMajor;
    using ElementC = complex<float>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 64;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous64b;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicand64bCrosswise;


    using IteratorThreadMapA = transform::PitchLinearWarpStripedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            layout::PitchLinearShape<16, 2>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<16, 2>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, kTransformA, kTransformB,
            Operator>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<16, 8, 8>, complex<float>,
        layout::ColumnMajor, complex<float>, layout::RowMajor, complex<float>,
        LayoutC_, arch::OpClassTensorOp, Stages, TransformA, TransformB,
        Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<16, 8, 8>;
    using ElementA = complex<float>;
    using LayoutA = layout::ColumnMajor;
    using ElementB = complex<float>;
    using LayoutB = layout::RowMajor;
    using ElementC = complex<float>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 64;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous64b;

    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous64b;


    using IteratorThreadMapA = transform::PitchLinearWarpStripedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            layout::PitchLinearShape<16, 2>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpStripedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<16, 2>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, kTransformA, kTransformB,
            Operator>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<16, 8, 8>, complex<float>,
        layout::RowMajor, complex<float>, layout::ColumnMajor, complex<float>,
        LayoutC_, arch::OpClassTensorOp, Stages, TransformA, TransformB,
        Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<16, 8, 8>;
    using ElementA = complex<float>;
    using LayoutA = layout::RowMajor;
    using ElementB = complex<float>;
    using LayoutB = layout::ColumnMajor;
    using ElementC = complex<float>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 64;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicand64bCrosswise;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicand64bCrosswise;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<16, 2>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<16, 2>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, kTransformA, kTransformB,
            Operator>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<16, 8, 8>, complex<float>,
        layout::RowMajor, complex<float>, layout::RowMajor, complex<float>,
        LayoutC_, arch::OpClassTensorOp, Stages, TransformA, TransformB,
        Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<16, 8, 8>;
    using ElementA = complex<float>;
    using LayoutA = layout::RowMajor;
    using ElementB = complex<float>;
    using LayoutB = layout::RowMajor;
    using ElementC = complex<float>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 64;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicand64bCrosswise;

    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous64b;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<16, 2>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpStripedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<16, 2>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, kTransformA, kTransformB,
            Operator>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_, typename RealA, typename RealB, typename RealC,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<1, 1, 1>, complex<RealA>,
        layout::ColumnMajor, complex<RealB>, layout::ColumnMajor,
        complex<RealC>, LayoutC_, arch::OpClassSimt, Stages, TransformA,
        TransformB, Operator_, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = complex<RealA>;
    using LayoutA = layout::ColumnMajor;
    using ElementB = complex<RealB>;
    using LayoutB = layout::ColumnMajor;
    using ElementC = complex<RealC>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = sizeof_bits<ElementA>::value;

    static int const kElementsPerAccess = 1;


    using SmemLayoutA = layout::ColumnMajor;

    using SmemLayoutB = layout::RowMajor;


    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            kElementsPerAccess>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            kElementsPerAccess>;

    using SmemThreadMapB =
            transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            SmemThreadMapB>;


    static const int WarpNumThreadsM =
            4;
    static const int WarpNumThreadsN = 8;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::RowMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy
            >;

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                                MatrixShape<0, Shape::kK / 32>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_, typename RealA, typename RealB, typename RealC,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<1, 1, 1>, complex<RealA>,
        layout::ColumnMajor, complex<RealB>, layout::RowMajor, complex<RealC>,
        LayoutC_, arch::OpClassSimt, Stages, TransformA, TransformB, Operator_,
        CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = complex<RealA>;
    using LayoutA = layout::ColumnMajor;
    using ElementB = complex<RealB>;
    using LayoutB = layout::RowMajor;
    using ElementC = complex<RealC>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = sizeof_bits<ElementA>::value;

    static int const kElementsPerAccess = 1;


    using SmemLayoutA = layout::ColumnMajor;

    using SmemLayoutB = layout::RowMajor;


    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            kElementsPerAccess>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            kElementsPerAccess>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            IteratorThreadMapB>;


    static const int WarpNumThreadsM =
            4;
    static const int WarpNumThreadsN = 8;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::RowMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy
            >;

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                                MatrixShape<0, 0>,
                                WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_, typename RealA, typename RealB, typename RealC,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<1, 1, 1>, complex<RealA>,
        layout::RowMajor, complex<RealB>, layout::ColumnMajor, complex<RealC>,
        LayoutC_, arch::OpClassSimt, Stages, TransformA, TransformB, Operator_,
        CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = complex<RealA>;
    using LayoutA = layout::RowMajor;
    using ElementB = complex<RealB>;
    using LayoutB = layout::ColumnMajor;
    using ElementC = complex<RealC>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = sizeof_bits<ElementA>::value;

    static int const kElementsPerAccess = 1;


    using SmemLayoutA = layout::ColumnMajor;

    using SmemLayoutB = layout::RowMajor;


    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            kElementsPerAccess>;

    using SmemThreadMapA =
            transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            SmemThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            kElementsPerAccess>;

    using SmemThreadMapB =
            transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            SmemThreadMapB>;


    static const int WarpNumThreadsM =
            4;
    static const int WarpNumThreadsN = 8;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::RowMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy
            >;

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<Shape::kK / 32, 0>,
                                MatrixShape<0, Shape::kK / 32>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_, typename RealA, typename RealB, typename RealC,
        typename LayoutC_,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
        Shape_, WarpShape_, GemmShape<1, 1, 1>, complex<RealA>,
        layout::RowMajor, complex<RealB>, layout::RowMajor, complex<RealC>,
        LayoutC_, arch::OpClassSimt, Stages, TransformA, TransformB, Operator_,
        CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = complex<RealA>;
    using LayoutA = layout::RowMajor;
    using ElementB = complex<RealB>;
    using LayoutB = layout::RowMajor;
    using ElementC = complex<RealC>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;
    using Operator = Operator_;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static_assert(WarpCount::kCount > 1,
                  "This specialization requires at least two warps.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = sizeof_bits<ElementA>::value;

    static int const kElementsPerAccess = 1;


    using SmemLayoutA = layout::ColumnMajor;

    using SmemLayoutB = layout::RowMajor;


    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            kElementsPerAccess>;

    using SmemThreadMapA =
            transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            SmemThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            kElementsPerAccess>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            IteratorThreadMapB>;


    static const int WarpNumThreadsM =
            4;
    static const int WarpNumThreadsN = 8;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::RowMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy
            >;

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<Shape::kK / 32, 0>,
                                MatrixShape<0, 0>,
                                WarpCount::kK>;
};


}
}
}

