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
#include "cutlass/array.h"
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::RowMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2,
                      Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

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

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
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
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2,
                      Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

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

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
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
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

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

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
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
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2,
                      Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

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

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
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
        typename LayoutC_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, float,
                      layout::ColumnMajor, float, layout::RowMajor, float,
                      LayoutC_, arch::OpClassTensorOp, 2,
                      arch::OpMultiplyAddFastF16> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = float;
    using LayoutA = layout::ColumnMajor;
    using ElementB = float;
    using LayoutB = layout::RowMajor;
    using ElementC = float;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 256;

    using Operator = arch::OpMultiplyAdd;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
            sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>;

    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
            sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, half_t, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, half_t, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, half_t, SmemLayoutA, half_t,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutC_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, float,
                      layout::RowMajor, float, layout::ColumnMajor, float,
                      LayoutC_, arch::OpClassTensorOp, 2,
                      arch::OpMultiplyAddFastF16> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = float;
    using LayoutA = layout::RowMajor;
    using ElementB = float;
    using LayoutB = layout::ColumnMajor;
    using ElementC = float;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 256;

    using Operator = arch::OpMultiplyAdd;

    static int const kWarpThreadArrangementContiguousA =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;

    static int const kWarpThreadArrangementContiguousB =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<half_t>::value, Shape::kK>;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<half_t>::value, Shape::kK>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, half_t, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, half_t, SmemLayoutB, 1,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, half_t, SmemLayoutA, half_t,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutC_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, float,
                      layout::RowMajor, float, layout::RowMajor, float,
                      LayoutC_, arch::OpClassTensorOp, 2,
                      arch::OpMultiplyAddFastF16> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = float;
    using LayoutA = layout::RowMajor;
    using ElementB = float;
    using LayoutB = layout::RowMajor;
    using ElementC = float;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 256;

    using Operator = arch::OpMultiplyAdd;

    static int const kWarpThreadArrangementContiguousA =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;


    using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<half_t>::value, Shape::kK>;

    using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
            sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, half_t, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, half_t, SmemLayoutB, 0,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, half_t, SmemLayoutA, half_t,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutC_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, float,
                      layout::ColumnMajor, float, layout::ColumnMajor, float,
                      LayoutC_, arch::OpClassTensorOp, 2,
                      arch::OpMultiplyAddFastF16> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = float;
    using LayoutA = layout::ColumnMajor;
    using ElementB = float;
    using LayoutB = layout::ColumnMajor;
    using ElementC = float;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 256;

    using Operator = arch::OpMultiplyAdd;

    static int const kWarpThreadArrangementContiguousB =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;


    using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
            sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>;

    using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<half_t>::value, Shape::kK>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            layout::PitchLinearShape<8, 4>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, half_t, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, half_t, SmemLayoutB, 1,
            IteratorThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, half_t, SmemLayoutA, half_t,
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
        typename Operator_,
        bool AccumulatorsInRowMajor,
        int InterleavedK>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajorInterleaved<InterleavedK>, ElementB_,
                      layout::RowMajorInterleaved<InterleavedK>, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, 2, Operator_,
                      AccumulatorsInRowMajor> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;
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

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
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

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            SmemThreadMapB>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB,
            SmemLayoutB, ElementC, LayoutC, Operator, WarpCount::kK,
            AccumulatorsInRowMajor>::Type;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


}
}
}
