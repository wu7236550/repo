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

#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/mma_singlestage.h"
#include "cutlass/arch/cache_operation.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape,
        typename WarpShape,
        typename InstructionShape,
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename LayoutC,
        typename OperatorClass,
        int Stages = 2,
        typename Operator = typename platform::conditional<
                (platform::is_same<OperatorClass,
                                   cutlass::arch::OpClassTensorOp>::value) &&
                        (platform::is_same<ElementA, int8_t>::value ||
                         platform::is_same<ElementA, int4b_t>::value ||
                         platform::is_same<ElementA, uint8_t>::value ||
                         platform::is_same<ElementA, uint4b_t>::value),
                cutlass::arch::OpMultiplyAddSaturate,
                cutlass::arch::OpMultiplyAdd>::type,
        bool AccumulatorsInRowMajor = false
        ,
        cutlass::arch::CacheOperation::Kind CacheOpA =
                cutlass::arch::CacheOperation::Global,
        cutlass::arch::CacheOperation::Kind CacheOpB =
                cutlass::arch::CacheOperation::Global,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone,
        bool IsComplex = false
        >
struct DefaultMmaCore;


}
}
}
