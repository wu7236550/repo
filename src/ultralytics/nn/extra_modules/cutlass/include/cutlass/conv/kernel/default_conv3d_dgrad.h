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
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv3d_dgrad_output_gradient_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv3d_dgrad_filter_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_tile_iterator.h"


namespace cutlass {
namespace conv {
namespace kernel {

template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementAccumulator, typename OperatorClass, typename ArchTag,
          typename ThreadblockShape, typename WarpShape,
          typename InstructionShape, typename EpilogueOutputOp,
          typename ThreadblockSwizzle, int Stages, typename MathOperatorTag,
          conv::IteratorAlgorithm IteratorAlgorithm =
                  IteratorAlgorithm::kAnalytic,
          conv::StrideSupport StrideSupport = StrideSupport::kStrided>
struct DefaultConv3dDgrad;

template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementAccumulator, typename OperatorClass, typename ArchTag,
          typename ThreadblockShape, typename WarpShape,
          typename InstructionShape, typename EpilogueOutputOp,
          typename ThreadblockSwizzle, int Stages, typename MathOperatorTag>
struct DefaultConv3dDgrad<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                          LayoutC, ElementAccumulator, OperatorClass, ArchTag,
                          ThreadblockShape, WarpShape, InstructionShape,
                          EpilogueOutputOp, ThreadblockSwizzle, Stages,
                          MathOperatorTag, IteratorAlgorithm::kAnalytic,
                          StrideSupport::kStrided> {
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementA,
            layout::RowMajor, ElementB, layout::RowMajor, ElementAccumulator,
            layout::RowMajor, OperatorClass, Stages, MathOperatorTag>;

    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using IteratorA = cutlass::conv::threadblock::
            Conv3dDgradOutputGradientTileAccessIteratorAnalytic<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementA, ThreadMapA, StrideSupport::kStrided>;

    using SmemIteratorA = typename MmaCore::SmemIteratorA;

    using ThreadMapB = typename MmaCore::IteratorThreadMapB;
    using IteratorB = cutlass::conv::threadblock::
            Conv3dDgradFilterTileAccessIteratorAnalytic<
                    cutlass::MatrixShape<ThreadblockShape::kK,
                                         ThreadblockShape::kN>,
                    ElementB, ThreadMapB>;

    using SmemIteratorB = typename MmaCore::SmemIteratorB;

    using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
    using MmaPolicy = typename MmaCore::MmaPolicy;

    using Mma = threadblock::ImplicitGemmMultistage<
            ThreadblockShape, IteratorA, SmemIteratorA,
            arch::CacheOperation::Always, IteratorB, SmemIteratorB,
            arch::CacheOperation::Global, MmaPolicy, Stages>;

    using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
            ThreadblockShape, WarpMmaTensorOp, 1, EpilogueOutputOp,
            EpilogueOutputOp::kCount>::Epilogue;

    using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad,
            Conv3dProblemSize>;
};


}
}
}

