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

#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"


namespace cutlass {
namespace gemm {
namespace kernel {


template <
        typename ElementA_,
        typename LayoutA_,
        ComplexTransform TransformA,
        int kAlignmentA,
        typename ElementB_,
        typename LayoutB_,
        ComplexTransform TransformB,
        int kAlignmentB,
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
        typename Operator,
        typename Enable = void>
struct DefaultGemmUniversal;


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
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
struct DefaultGemmUniversal<
        ElementA, LayoutA,
        ComplexTransform::kNone,
        kAlignmentA, ElementB, LayoutB,
        ComplexTransform::kNone,
        kAlignmentB, ElementC, LayoutC, ElementAccumulator, OperatorClass,
        ArchTag, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator,
        typename std::enable_if<
                !cutlass::is_complex<ElementAccumulator>::value>::type> {
    using DefaultGemmKernel = typename kernel::DefaultGemm<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
            ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
            ThreadblockSwizzle, Stages, true, Operator, false>::GemmKernel;

    using GemmKernel =
            kernel::GemmUniversal<typename DefaultGemmKernel::Mma,
                                  typename DefaultGemmKernel::Epilogue,
                                  ThreadblockSwizzle>;
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
struct DefaultGemmUniversal<
        ElementA, LayoutA, TransformA, kAlignmentA, ElementB, LayoutB,
        TransformB, kAlignmentB, ElementC, LayoutC, ElementAccumulator,
        OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator,
        typename std::enable_if<
                cutlass::is_complex<ElementAccumulator>::value>::type> {
    using DefaultGemmKernel = typename kernel::DefaultGemmComplex<
            ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
            WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
            Stages, TransformA, TransformB, Operator, false>::GemmKernel;

    using GemmKernel =
            kernel::GemmUniversal<typename DefaultGemmKernel::Mma,
                                  typename DefaultGemmKernel::Epilogue,
                                  ThreadblockSwizzle>;
};


}
}
}

