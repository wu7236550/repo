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
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "kernel/b2b_gemm.h"
#include "threadblock/default_b2b_mma.h"


namespace cutlass {
namespace gemm {
namespace kernel {


template <
        typename ElementA_,
        typename LayoutA_,
        int kAlignmentA,
        typename ElementB_,
        typename LayoutB_,
        int kAlignmentB,
        typename ElementC_,
        typename LayoutC_,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape0,
        typename ThreadblockShape1,
        typename WarpShape0,
        typename WarpShape1,
        typename InstructionShape,
        typename EpilogueOutputOp0,
        typename EpilogueOutputOp1,
        typename ThreadblockSwizzle,
        int Stages,
        bool SplitKSerial,
        typename Operator,
        bool IsBetaZero = false>
struct DefaultB2bGemm;


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementC,
        typename ElementAccumulator,
        typename ThreadblockShape0,
        typename ThreadblockShape1,
        typename WarpShape0,
        typename WarpShape1,
        typename InstructionShape,
        typename EpilogueOutputOp0,
        typename EpilogueOutputOp1,
        typename ThreadblockSwizzle,
        bool SplitKSerial,
        typename Operator>
struct DefaultB2bGemm<
        ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
        ElementC, layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
        arch::Sm75, ThreadblockShape0, ThreadblockShape1, WarpShape0,
        WarpShape1, InstructionShape, EpilogueOutputOp0, EpilogueOutputOp1,
        ThreadblockSwizzle, 2, SplitKSerial, Operator> {
    using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
            arch::Sm75, ThreadblockShape0, ThreadblockShape1, WarpShape0,
            WarpShape1, InstructionShape, 2, Operator,
            EpilogueOutputOp0>::ThreadblockB2bMma;

    static const int kPartitionsK1 = ThreadblockShape1::kK / WarpShape1::kK;

    using Epilogue =
            typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
                    ThreadblockShape1, typename B2bMma::Operator1,
                    kPartitionsK1, EpilogueOutputOp1,
                    EpilogueOutputOp1::kCount>::Epilogue;

    using B2bGemmKernel =
            kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

template <
        typename ElementA,
        int kAlignmentA,
        typename ElementB,
        int kAlignmentB,
        typename ElementC,
        typename ThreadblockShape0,
        typename ThreadblockShape1,
        typename WarpShape0,
        typename WarpShape1,
        typename InstructionShape,
        typename EpilogueOutputOp0,
        typename EpilogueOutputOp1,
        typename ThreadblockSwizzle,
        int Stages,
        int InterleavedK,
        bool SplitKSerial,
        typename Operator,
        bool IsBetaZero>
struct DefaultB2bGemm<
        ElementA, layout::ColumnMajorInterleaved<InterleavedK>, kAlignmentA,
        ElementB, layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
        ElementC, layout::ColumnMajorInterleaved<InterleavedK>, int32_t,
        arch::OpClassTensorOp, arch::Sm80, ThreadblockShape0, ThreadblockShape1,
        WarpShape0, WarpShape1, InstructionShape, EpilogueOutputOp0,
        EpilogueOutputOp1, ThreadblockSwizzle, Stages, SplitKSerial, Operator,
        IsBetaZero> {
    using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
    using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
    using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

    using ElementAccumulator = int32_t;

    using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
            ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
            InstructionShape, Stages, Operator, EpilogueOutputOp0,
            true>::ThreadblockB2bMma;

    static const int kPartitionsK1 = ThreadblockShape1::kK / WarpShape1::kK;

    using Epilogue = typename cutlass::epilogue::threadblock::
            DefaultInterleavedEpilogueTensorOp<
                    ThreadblockShape1, typename B2bMma::Operator1,
                    kPartitionsK1, EpilogueOutputOp1,
                    64 / sizeof_bits<ElementC>::value, InterleavedK,
                    IsBetaZero>::Epilogue;

    using B2bGemmKernel =
            kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename ElementA,
        int kAlignmentA,
        typename ElementB,
        int kAlignmentB,
        typename ElementC,
        typename ThreadblockShape0,
        typename ThreadblockShape1,
        typename WarpShape0,
        typename WarpShape1,
        typename InstructionShape,
        typename EpilogueOutputOp0,
        typename EpilogueOutputOp1,
        typename ThreadblockSwizzle,
        int InterleavedK,
        bool SplitKSerial,
        typename Operator,
        bool IsBetaZero>
struct DefaultB2bGemm<
        ElementA, layout::ColumnMajorInterleaved<InterleavedK>, kAlignmentA,
        ElementB, layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
        ElementC, layout::ColumnMajorInterleaved<InterleavedK>, int32_t,
        arch::OpClassTensorOp, arch::Sm75, ThreadblockShape0, ThreadblockShape1,
        WarpShape0, WarpShape1, InstructionShape, EpilogueOutputOp0,
        EpilogueOutputOp1, ThreadblockSwizzle, 2, SplitKSerial, Operator,
        IsBetaZero> {
    using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
    using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
    using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

    using ElementAccumulator = int32_t;

    using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm75,
            ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
            InstructionShape, 2, Operator, EpilogueOutputOp0,
            true>::ThreadblockB2bMma;

    static const int kPartitionsK1 = ThreadblockShape1::kK / WarpShape1::kK;

    using Epilogue = typename cutlass::epilogue::threadblock::
            DefaultInterleavedEpilogueTensorOp<
                    ThreadblockShape1, typename B2bMma::Operator1,
                    kPartitionsK1, EpilogueOutputOp1,
                    64 / sizeof_bits<ElementC>::value, InterleavedK,
                    IsBetaZero>::Epilogue;

    using B2bGemmKernel =
            kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};



}
}
}
