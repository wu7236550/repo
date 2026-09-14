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

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/default_epilogue_complex_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"


namespace cutlass {
namespace gemm {
namespace kernel {


template <
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator,
        bool SplitKSerial>
struct DefaultGemmComplex;


template <
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator,
        bool SplitKSerial>
struct DefaultGemmComplex<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, layout::RowMajor,
        ElementAccumulator, arch::OpClassSimt, arch::Sm50, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, TransformA, TransformB, Operator, SplitKSerial> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
            ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
            arch::OpClassSimt, Stages, Operator, false,
            cutlass::arch::CacheOperation::Global,
            cutlass::arch::CacheOperation::Global, TransformA, TransformB>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
            ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
            ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB>;

    using Mma = cutlass::gemm::threadblock::MmaPipelined<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
            layout::RowMajor, typename MmaCore::MmaPolicy>;

    using Epilogue =
            typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
                    ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
                    EpilogueOutputOp::kCount>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator,
        bool SplitKSerial>
struct DefaultGemmComplex<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, layout::RowMajor,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm80, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, TransformA, TransformB, Operator, SplitKSerial> {
    using Mma =
            typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
                    ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
                    layout::RowMajor, arch::OpClassTensorOp, arch::Sm80,
                    ThreadblockShape, WarpShape, InstructionShape, Stages,
                    TransformA, TransformB, Operator>::ThreadblockMma;

    using Epilogue = typename cutlass::epilogue::threadblock::
            DefaultEpilogueComplexTensorOp<
                    ThreadblockShape, typename Mma::Operator, 1,
                    EpilogueOutputOp, EpilogueOutputOp::kCount,
                    Operator>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Operator,
        bool SplitKSerial>
struct DefaultGemmComplex<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, layout::RowMajor,
        ElementAccumulator, arch::OpClassSimt, arch::Sm80, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, TransformA, TransformB, Operator, SplitKSerial> {
    using Mma =
            typename cutlass::gemm::threadblock::DefaultMultistageMmaComplex<
                    ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
                    layout::RowMajor, arch::OpClassSimt, arch::Sm80,
                    ThreadblockShape, WarpShape, InstructionShape, Stages,
                    TransformA, TransformB, Operator>::ThreadblockMma;

    using Epilogue =
            typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
                    ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
                    EpilogueOutputOp::kCount>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


}
}
}

