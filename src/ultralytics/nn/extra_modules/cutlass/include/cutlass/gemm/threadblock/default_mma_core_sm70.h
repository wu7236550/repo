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

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h"

#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape_,
        typename WarpShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<8, 8, 4>, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::RowMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2,
                      Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

    using Operator = Operator_;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;


    using SmemLayoutA = layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<
            sizeof_bits<ElementA>::value>;

    using SmemLayoutB = layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
            sizeof_bits<ElementB>::value>;


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


    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32,
                               ElementA, LayoutA, ElementB, LayoutB, ElementC,
                               cutlass::layout::RowMajor,
                               cutlass::arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
            WarpShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB, ElementC,
            LayoutC, Policy>;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<8, 8, 4>, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2,
                      Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

    using Operator = Operator_;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;


    using SmemLayoutA = layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementA>::value, Shape::kK>;

    using SmemLayoutB = layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementB>::value, Shape::kK>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<4, 8>,
            kAccessSizeInBits / sizeof_bits<ElementA>::value>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<4, 8>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            IteratorThreadMapB>;


    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32,
                               ElementA, LayoutA, ElementB, LayoutB, ElementC,
                               cutlass::layout::RowMajor,
                               cutlass::arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
            WarpShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB, ElementC,
            LayoutC, Policy>;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<8, 8, 4>, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

    using Operator = Operator_;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;


    using SmemLayoutA = layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementA>::value, Shape::kK>;

    using SmemLayoutB = layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
            sizeof_bits<ElementB>::value>;


    using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<4, 8>,
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


    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32,
                               ElementA, LayoutA, ElementB, LayoutB, ElementC,
                               cutlass::layout::RowMajor,
                               cutlass::arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
            WarpShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB, ElementC,
            LayoutC, Policy>;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<8, 8, 4>, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2,
                      Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<8, 8, 4>;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassTensorOp;

    using Operator = Operator_;

    using WarpCount =
            GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
                      Shape::kK / WarpShape::kK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;


    using SmemLayoutA = layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<
            sizeof_bits<ElementA>::value>;

    using SmemLayoutB = layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
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
            layout::PitchLinearShape<4, 8>,
            kAccessSizeInBits / sizeof_bits<ElementB>::value>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
            IteratorThreadMapB>;


    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32,
                               ElementA, LayoutA, ElementB, LayoutB, ElementC,
                               cutlass::layout::RowMajor,
                               cutlass::arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
            WarpShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB, ElementC,
            LayoutC, Policy>;

    using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

}
}
}
