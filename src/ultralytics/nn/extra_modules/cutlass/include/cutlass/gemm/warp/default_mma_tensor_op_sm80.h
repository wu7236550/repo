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
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename WarpShape_,
        typename LayoutA,
        typename LayoutB,
        typename LayoutC,
        int PartitionsK,
        bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<WarpShape_, GemmShape<16, 8, 8>, float, LayoutA,
                          float, LayoutB, float, LayoutC,
                          arch::OpMultiplyAddFastBF16, PartitionsK,
                          AccumulatorsInRowMajor> {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<GemmShape<16, 8, 8>, 32, bfloat16_t,
                               cutlass::layout::RowMajor, bfloat16_t,
                               cutlass::layout::ColumnMajor, float,
                               cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaTensorOp<
            WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC, Policy,
            PartitionsK, AccumulatorsInRowMajor>;
};


template <
        typename WarpShape_,
        typename LayoutA,
        typename LayoutB,
        typename LayoutC,
        int PartitionsK,
        bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<WarpShape_, GemmShape<16, 8, 8>, float, LayoutA,
                          float, LayoutB, float, LayoutC,
                          arch::OpMultiplyAddFastF16, PartitionsK,
                          AccumulatorsInRowMajor> {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<GemmShape<16, 8, 8>, 32, half_t,
                               cutlass::layout::RowMajor, half_t,
                               cutlass::layout::ColumnMajor, float,
                               cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaTensorOp<
            WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC, Policy,
            PartitionsK, AccumulatorsInRowMajor>;
};


template <
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutA,
        typename LayoutB,
        typename LayoutC,
        int PartitionsK,
        bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp<WarpShape_, InstructionShape_, float, LayoutA, float,
                          LayoutB, float, LayoutC, arch::OpMultiplyAdd,
                          PartitionsK, AccumulatorsInRowMajor> {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<InstructionShape_, 32, tfloat32_t,
                               cutlass::layout::RowMajor, tfloat32_t,
                               cutlass::layout::ColumnMajor, float,
                               cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaTensorOp<
            WarpShape_, float, LayoutA, float, LayoutB, float, LayoutC, Policy,
            PartitionsK, AccumulatorsInRowMajor>;
};


}
}
}


#include "cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h"

