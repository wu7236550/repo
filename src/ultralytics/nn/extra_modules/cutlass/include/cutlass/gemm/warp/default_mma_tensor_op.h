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
#include "cutlass/gemm/warp/mma_tensor_op.h"

namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename WarpShape_,
        typename InstructionShape_,
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_ = arch::OpMultiplyAdd,
        int PartitionsK = 1,
        bool AccumulatorsInRowMajor = false>
struct DefaultMmaTensorOp;


template <
        typename WarpShape_,
        typename InstructionShape_,
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename LayoutC,
        typename Operator_,
        int PartitionsK,
        bool AccumulatorsInRowMajor>
struct DefaultMmaTensorOp {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<InstructionShape_, 32, ElementA,
                               cutlass::layout::RowMajor, ElementB,
                               cutlass::layout::ColumnMajor, ElementC,
                               cutlass::layout::RowMajor, Operator_>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaTensorOp<
            WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            Policy, PartitionsK, AccumulatorsInRowMajor>;
};


}
}
}


#include "default_mma_tensor_op_sm80.h"

