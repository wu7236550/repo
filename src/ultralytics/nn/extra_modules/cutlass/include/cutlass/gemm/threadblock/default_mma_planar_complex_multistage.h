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

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/mma_planar_complex_multistage.h"

#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename ElementA_,
        typename LayoutA_,
        int kAlignmentA,
        typename ElementB_,
        typename LayoutB_,
        int kAlignmentB,
        typename ElementAccumulator_,
        typename LayoutC_,
        typename OperatorClass_,
        typename ArchTag_,
        typename ThreadblockShape_,
        typename WarpShape_,
        typename InstructionShape_,
        int Stages,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone,
        typename Operator = arch::OpMultiplyAdd>
struct DefaultMmaPlanarComplexMultistage {
    using RealMmaMultistage = typename DefaultMma<
            ElementA_, LayoutA_, kAlignmentA, ElementB_, LayoutB_, kAlignmentB,
            ElementAccumulator_, LayoutC_, OperatorClass_, ArchTag_,
            ThreadblockShape_, WarpShape_, InstructionShape_, Stages,
            Operator>::ThreadblockMma;

    using ThreadblockMma = MmaPlanarComplexMultistage<
            ThreadblockShape_, typename RealMmaMultistage::IteratorA,
            typename RealMmaMultistage::SmemIteratorA,
            cutlass::arch::CacheOperation::Global,
            typename RealMmaMultistage::IteratorB,
            typename RealMmaMultistage::SmemIteratorB,
            cutlass::arch::CacheOperation::Global, ElementAccumulator_,
            LayoutC_, typename RealMmaMultistage::Policy, Stages, TransformA,
            TransformB>;
};


}
}
}

