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

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/kernel/gemm_planar_complex.h"
#include "cutlass/gemm/kernel/gemm_planar_complex_array.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"

#include "cutlass/epilogue/threadblock/default_epilogue_planar_complex.h"
#include "cutlass/gemm/threadblock/default_mma_planar_complex_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_planar_complex_multistage.h"


namespace cutlass {
namespace gemm {
namespace kernel {


template <
        typename ElementA,
        typename LayoutA,
        ComplexTransform TransformA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        ComplexTransform TransformB,
        int kAlignmentB,
        typename ElementC,
        typename LayoutC,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename Operator,
        typename Enable = void>
struct DefaultGemmPlanarComplexUniversal;


template <
        typename ElementA,
        typename LayoutA,
        ComplexTransform TransformA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        ComplexTransform TransformB,
        int kAlignmentB,
        typename ElementC,
        typename LayoutC,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename Operator>
struct DefaultGemmPlanarComplexUniversal<
        ElementA, LayoutA, TransformA, kAlignmentA, ElementB, LayoutB,
        TransformB, kAlignmentB, ElementC, LayoutC, ElementAccumulator,
        OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator,
        typename std::enable_if<(Stages <= 2)>::type> {
    using Mma = typename gemm::threadblock::DefaultMmaPlanarComplexPipelined<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, LayoutC, OperatorClass, ArchTag,
            ThreadblockShape, WarpShape, InstructionShape, Stages, TransformA,
            TransformB, Operator>::ThreadblockMma;

    using Epilogue =
            typename epilogue::threadblock::DefaultEpiloguePlanarComplex<
                    ThreadblockShape, typename Mma::Policy::Operator,
                    OperatorClass, ArchTag,
                    ThreadblockShape::kK / WarpShape::kK, EpilogueOutputOp,
                    EpilogueOutputOp::kCount>::Epilogue;

    using GemmKernel =
            kernel::GemmPlanarComplex<Mma, Epilogue, ThreadblockSwizzle>;

    using GemmArrayKernel =
            kernel::GemmPlanarComplexArray<Mma, Epilogue, ThreadblockSwizzle>;
};


template <
        typename ElementA,
        typename LayoutA,
        ComplexTransform TransformA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        ComplexTransform TransformB,
        int kAlignmentB,
        typename ElementC,
        typename LayoutC,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename Operator>
struct DefaultGemmPlanarComplexUniversal<
        ElementA, LayoutA, TransformA, kAlignmentA, ElementB, LayoutB,
        TransformB, kAlignmentB, ElementC, LayoutC, ElementAccumulator,
        OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator,
        typename std::enable_if<(Stages > 2)>::type> {
    using Mma = typename gemm::threadblock::DefaultMmaPlanarComplexMultistage<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, LayoutC, OperatorClass, ArchTag,
            ThreadblockShape, WarpShape, InstructionShape, Stages, TransformA,
            TransformB, Operator>::ThreadblockMma;

    using Epilogue =
            typename epilogue::threadblock::DefaultEpiloguePlanarComplex<
                    ThreadblockShape, typename Mma::Policy::Operator,
                    OperatorClass, ArchTag,
                    ThreadblockShape::kK / WarpShape::kK, EpilogueOutputOp,
                    EpilogueOutputOp::kCount>::Epilogue;

    using GemmKernel =
            kernel::GemmPlanarComplex<Mma, Epilogue, ThreadblockSwizzle>;

    using GemmArrayKernel =
            kernel::GemmPlanarComplexArray<Mma, Epilogue, ThreadblockSwizzle>;
};


}
}
}

