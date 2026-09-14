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
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h"

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
        typename InstructionShape_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::ColumnMajor, double, layout::ColumnMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                      CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = double;
    using LayoutA = layout::ColumnMajor;
    using ElementB = double;
    using LayoutB = layout::ColumnMajor;
    using ElementC = double;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
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

    using Operator = Operator_;


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


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::ColumnMajor, double, layout::RowMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                      CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = double;
    using LayoutA = layout::ColumnMajor;
    using ElementB = double;
    using LayoutB = layout::RowMajor;
    using ElementC = double;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
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

    using Operator = Operator_;


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


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::RowMajor, double, layout::ColumnMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                      CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = double;
    using LayoutA = layout::RowMajor;
    using ElementB = double;
    using LayoutB = layout::ColumnMajor;
    using ElementC = double;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
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

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 64;

    using Operator = Operator_;


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


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::RowMajor, double, layout::RowMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                      CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = double;
    using LayoutA = layout::RowMajor;
    using ElementB = double;
    using LayoutB = layout::RowMajor;
    using ElementC = double;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
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

    using Operator = Operator_;


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


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutA_,
        typename LayoutB_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB,
        ComplexTransform TransformA_,
        ComplexTransform TransformB_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<16, 8, 8>, complex<float>,
                      LayoutA_, complex<float>, LayoutB_, complex<float>,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                      CacheOpA, CacheOpB, TransformA_, TransformB_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<16, 8, 8>;
    using ElementA = complex<float>;
    using LayoutA = LayoutA_;
    using ElementB = complex<float>;
    using LayoutB = LayoutB_;
    using ElementC = complex<float>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;
    static const ComplexTransform TransformA = TransformA_;
    static const ComplexTransform TransformB = TransformB_;

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

    using Operator = Operator_;

    static_assert(
            platform::is_same<Operator, arch::OpMultiplyAddComplex>::value ||
                    platform::is_same<
                            Operator,
                            arch::OpMultiplyAddGaussianComplex>::value,
            "The operator tag must indicate complex multiplication.");


    using MmaComplexCore = DefaultMultistageMmaComplexCore<
            Shape, WarpShape, InstructionShape, ElementA, LayoutA, ElementB,
            LayoutB, ElementC, LayoutC, arch::OpClassTensorOp, kStages,
            TransformA, TransformB, Operator, kCacheOpA, kCacheOpB>;


    using SmemLayoutA = typename MmaComplexCore::SmemLayoutA;

    using SmemLayoutB = typename MmaComplexCore::SmemLayoutB;


    using IteratorThreadMapA = typename MmaComplexCore::IteratorThreadMapA;

    using SmemIteratorA = typename MmaComplexCore::SmemIteratorA;

    using IteratorThreadMapB = typename MmaComplexCore::IteratorThreadMapB;

    using SmemIteratorB = typename MmaComplexCore::SmemIteratorB;


    using MmaTensorOp = typename MmaComplexCore::MmaTensorOp;

    using MmaPolicy = typename MmaComplexCore::MmaPolicy;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename LayoutA_,
        typename LayoutB_,
        typename LayoutC_,
        int Stages,
        typename Operator_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB,
        ComplexTransform TransformA_,
        ComplexTransform TransformB_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<8, 8, 4>, complex<double>,
                      LayoutA_, complex<double>, LayoutB_, complex<double>,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                      CacheOpA, CacheOpB, TransformA_, TransformB_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = complex<double>;
    using LayoutA = LayoutA_;
    using ElementB = complex<double>;
    using LayoutB = LayoutB_;
    using ElementC = complex<double>;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA =
            cutlass::arch::CacheOperation::Always;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB =
            cutlass::arch::CacheOperation::Always;
    static const ComplexTransform TransformA = TransformA_;
    static const ComplexTransform TransformB = TransformB_;

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

    using Operator = Operator_;

    static_assert(
            platform::is_same<Operator, arch::OpMultiplyAddComplex>::value ||
                    platform::is_same<
                            Operator,
                            arch::OpMultiplyAddGaussianComplex>::value,
            "The operator tag must indicate complex multiplication.");


    using MmaComplexCore = DefaultMultistageMmaComplexCore<
            Shape, WarpShape, InstructionShape, ElementA, LayoutA, ElementB,
            LayoutB, ElementC, LayoutC, arch::OpClassTensorOp, kStages,
            TransformA, TransformB, Operator, kCacheOpA, kCacheOpB>;


    using SmemLayoutA = typename MmaComplexCore::SmemLayoutA;

    using SmemLayoutB = typename MmaComplexCore::SmemLayoutB;


    using IteratorThreadMapA = typename MmaComplexCore::IteratorThreadMapA;

    using SmemIteratorA = typename MmaComplexCore::SmemIteratorA;

    using IteratorThreadMapB = typename MmaComplexCore::IteratorThreadMapB;

    using SmemIteratorB = typename MmaComplexCore::SmemIteratorB;


    using MmaTensorOp = typename MmaComplexCore::MmaTensorOp;

    using MmaPolicy = typename MmaComplexCore::MmaPolicy;
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
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


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
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
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;

    static int const kWarpThreadArrangementContiguousB =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementA>::value, Shape::kK>;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementB>::value, Shape::kK>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
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

    static int const kWarpThreadArrangementContiguousB =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
            sizeof_bits<ElementA>::value, int(128 / sizeof(ElementA))>;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementB>::value, Shape::kK>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                      CacheOpA, CacheOpB> {
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
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementA>::value, Shape::kK>;

    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
            sizeof_bits<ElementB>::value, int(128 / sizeof(ElementB))>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
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
        bool AccumulatorsInRowMajor,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        cutlass::arch::CacheOperation::Kind CacheOpB,
        int InterleavedK>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajorInterleaved<InterleavedK>, ElementB_,
                      layout::RowMajorInterleaved<InterleavedK>, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      AccumulatorsInRowMajor, CacheOpA, CacheOpB> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    static int const kStages = Stages;
    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
    static int const kInterleavedK = InterleavedK;

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

    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementA>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementA>::value, kInterleavedK>;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementB>::value, kInterleavedK>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapA = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapA,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            SmemThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapB = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapB,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            SmemThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK,
            AccumulatorsInRowMajor>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
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

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    using Operator = Operator_;

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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::RowMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
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

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    using Operator = Operator_;

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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
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

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    using Operator = Operator_;

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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, Stages, Operator_, false,
                      CacheOpA, CacheOpB> {
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

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    using Operator = Operator_;

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
                                MatrixShape<0, 0>, WarpCount::kK>;
};


}
}
}
