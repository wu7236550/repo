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
#include "cutlass/arch/wmma.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif


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
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        bool SplitKSerial,
        typename Operator,
        bool IsBetaZero = false>
struct DefaultGemm;


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementC,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        bool SplitKSerial,
        typename Operator>
struct DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                   kAlignmentB, ElementC, layout::RowMajor, ElementAccumulator,
                   arch::OpClassTensorOp, arch::Sm80, ThreadblockShape,
                   WarpShape, InstructionShape, EpilogueOutputOp,
                   ThreadblockSwizzle, Stages, SplitKSerial, Operator> {
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
            arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, Stages,
            Operator>::ThreadblockMma;

    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    using Epilogue =
            typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
                    ThreadblockShape, typename Mma::Operator, kPartitionsK,
                    EpilogueOutputOp, EpilogueOutputOp::kCount>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementC,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        bool SplitKSerial,
        typename Operator>
struct DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                   kAlignmentB, ElementC, layout::RowMajor, ElementAccumulator,
                   arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
                   WarpShape, InstructionShape, EpilogueOutputOp,
                   ThreadblockSwizzle, 2, SplitKSerial, Operator> {
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
            arch::Sm75, ThreadblockShape, WarpShape, InstructionShape, 2,
            Operator>::ThreadblockMma;

    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    using Epilogue =
            typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
                    ThreadblockShape, typename Mma::Operator, kPartitionsK,
                    EpilogueOutputOp, EpilogueOutputOp::kCount>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename ElementA,
        int kAlignmentA,
        typename ElementB,
        int kAlignmentB,
        typename ElementC,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        int InterleavedK,
        bool SplitKSerial,
        typename Operator,
        bool IsBetaZero>
struct DefaultGemm<
        ElementA, layout::ColumnMajorInterleaved<InterleavedK>, kAlignmentA,
        ElementB, layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
        ElementC, layout::ColumnMajorInterleaved<InterleavedK>, int32_t,
        arch::OpClassTensorOp, arch::Sm80, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages,
        SplitKSerial, Operator, IsBetaZero> {
    using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
    using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
    using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

    using ElementAccumulator = int32_t;

    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
            ThreadblockShape, WarpShape, InstructionShape, Stages, Operator,
            true>::ThreadblockMma;

    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    using Epilogue = typename cutlass::epilogue::threadblock::
            DefaultInterleavedEpilogueTensorOp<
                    ThreadblockShape, typename Mma::Operator, kPartitionsK,
                    EpilogueOutputOp, 64 / sizeof_bits<ElementC>::value,
                    InterleavedK, IsBetaZero>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename ElementA,
        int kAlignmentA,
        typename ElementB,
        int kAlignmentB,
        typename ElementC,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int InterleavedK,
        bool SplitKSerial,
        typename Operator,
        bool IsBetaZero>
struct DefaultGemm<ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
                   kAlignmentA, ElementB,
                   layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
                   ElementC, layout::ColumnMajorInterleaved<InterleavedK>,
                   int32_t, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
                   WarpShape, InstructionShape, EpilogueOutputOp,
                   ThreadblockSwizzle, 2, SplitKSerial, Operator, IsBetaZero> {
    using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
    using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
    using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

    using ElementAccumulator = int32_t;

    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm75,
            ThreadblockShape, WarpShape, InstructionShape, 2, Operator,
            true>::ThreadblockMma;

    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    using Epilogue = typename cutlass::epilogue::threadblock::
            DefaultInterleavedEpilogueTensorOp<
                    ThreadblockShape, typename Mma::Operator, kPartitionsK,
                    EpilogueOutputOp, 64 / sizeof_bits<ElementC>::value,
                    InterleavedK, IsBetaZero>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementC,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        bool SplitKSerial,
        typename Operator>
struct DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                   kAlignmentB, ElementC, layout::RowMajor, ElementAccumulator,
                   arch::OpClassTensorOp, arch::Sm70, ThreadblockShape,
                   WarpShape, GemmShape<8, 8, 4>, EpilogueOutputOp,
                   ThreadblockSwizzle, 2, SplitKSerial, Operator> {
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
            arch::Sm70, ThreadblockShape, WarpShape, GemmShape<8, 8, 4>, 2,
            Operator>::ThreadblockMma;

    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    using Epilogue = typename cutlass::epilogue::threadblock::
            DefaultEpilogueVoltaTensorOp<
                    ThreadblockShape, typename Mma::Operator, kPartitionsK,
                    EpilogueOutputOp, EpilogueOutputOp::kCount>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementC,
        typename ElementAccumulator,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        bool SplitKSerial,
        typename Operator>
struct DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                   kAlignmentB, ElementC, layout::RowMajor, ElementAccumulator,
                   arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                   GemmShape<1, 1, 1>, EpilogueOutputOp, ThreadblockSwizzle, 2,
                   SplitKSerial, Operator> {
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, layout::RowMajor, arch::OpClassSimt, arch::Sm50,
            ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, 2,
            Operator>::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1,
                  "simt epilogue must operate on scalars");

    using Epilogue =
            typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
                    ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename ElementA,
        typename LayoutA,
        int kAlignmentA,
        typename ElementB,
        typename LayoutB,
        int kAlignmentB,
        typename ElementC,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        bool SplitKSerial,
        typename Operator>
struct DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                   kAlignmentB, ElementC, layout::RowMajor, ElementAccumulator,
                   arch::OpClassSimt, arch::Sm80, ThreadblockShape, WarpShape,
                   GemmShape<1, 1, 1>, EpilogueOutputOp, ThreadblockSwizzle,
                   Stages, SplitKSerial, Operator> {
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, layout::RowMajor, arch::OpClassSimt, arch::Sm80,
            ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, Stages,
            Operator>::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1,
                  "simt epilogue must operate on scalars");

    using Epilogue =
            typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
                    ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};


template <
        typename LayoutA,
        int kAlignmentA,
        typename LayoutB,
        int kAlignmentB,
        typename LayoutC,
        typename ElementC,
        typename ArchTag,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        bool SplitKSerial,
        typename Operator>
struct DefaultGemm<int8_t, LayoutA, kAlignmentA, int8_t, LayoutB, kAlignmentB,
                   ElementC, LayoutC, ElementAccumulator, arch::OpClassSimt,
                   ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 4>,
                   EpilogueOutputOp, ThreadblockSwizzle, 2, SplitKSerial,
                   Operator, false> {
    using InstructionShape = GemmShape<1, 1, 4>;
    using ElementA = int8_t;
    using ElementB = int8_t;

    using OperatorClass = arch::OpClassSimt;
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, LayoutC, arch::OpClassSimt, arch::Sm50,
            ThreadblockShape, WarpShape, InstructionShape, 2, Operator,
            false>::ThreadblockMma;

    static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
    static_assert(kEpilogueElementsPerAccess == 1,
                  "simt epilogue must operate on scalars");

    using Epilogue =
            typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
                    ThreadblockShape, typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
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
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        bool SplitKSerial,
        typename Operator>
struct DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                   kAlignmentB, ElementC, LayoutC, ElementAccumulator,
                   arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape,
                   WarpShape, InstructionShape, EpilogueOutputOp,
                   ThreadblockSwizzle, Stages, SplitKSerial, Operator> {
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementAccumulator, LayoutC, arch::OpClassWmmaTensorOp, ArchTag,
            ThreadblockShape, WarpShape, InstructionShape, Stages,
            Operator>::ThreadblockMma;

    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

    using Epilogue = typename cutlass::epilogue::threadblock::
            DefaultEpilogueWmmaTensorOp<
                    ThreadblockShape, typename Mma::Operator, kPartitionsK,
                    EpilogueOutputOp, EpilogueOutputOp::kCount>::Epilogue;

    using GemmKernel =
            kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};
#endif


}
}
}
