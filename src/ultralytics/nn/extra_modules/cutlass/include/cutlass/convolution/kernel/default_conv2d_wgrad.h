/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
modification, are permitted
 * provided that the following conditions are met:
namespace conv {
 *     * Redistributions of source code must retain the above copyright notice,
this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
notice, this list of
 *       conditions and the following disclaimer in the documentation and/or
other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
contributors may be used
 *       to endorse or promote products derived from this software without
specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA
CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/**
 * \file include/cutlass/convolution/kernel/default_conv2d_wgrad.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/convolution/kernel/implicit_batched_gemm_dwconv2d_wgrad.h"

#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_simt.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm70.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm75.h"

#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn.h"

#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/dwconv2d_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/dwconv2d_direct_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/dwconv2d_direct_epilogue_volta_tensor_op.h"

#include "cutlass/epilogue/thread/bias_add_linear_combination_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_relu_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_hswish_clamp.h"
#include "cutlass/epilogue/thread/linear_combination.h"


namespace cutlass {
namespace conv {
namespace kernel {


template <
        typename ElementSrc,
        typename LayoutSrc,
        typename ElementDiff,
        typename LayoutDiff,
        typename ElementGrad,
        typename LayoutGrad,
        typename ElementAccumulator,
        typename OperatorClass,
        typename ArchTag,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename MathOperatorTag,
        int AlignmentSrc,
        int AlignmentDiff,
        ImplicitGemmMode GemmMode = ImplicitGemmMode::GEMM_NT,
        ConvType ConvolutionType = ConvType::kConvolution>
struct DefaultConvolution2dWgrad;


template <
        typename ElementGrad,
        typename LayoutGrad,
        typename ArchTag,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename MathOperatorTag,
        int kAlignmentSrc,
        int kAlignmentDiff>
struct DefaultConvolution2dWgrad<
        float, layout::TensorNCHW, float, layout::TensorNCHW, ElementGrad,
        LayoutGrad, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 1>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentDiff, ImplicitGemmMode::GEMM_NT,
        ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<1, 1, 1>;
    using ElementSrc = float;
    using ElementDiff = float;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutDiff = layout::TensorNCHW;
    using OperatorClass = arch::OpClassSimt;
    static const int kStages = Stages;
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_NT;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementDiff, LayoutDiff, kAlignmentDiff,
            ElementAccumulator, LayoutGrad, OperatorClass, Stages,
            MathOperatorTag, true, kGemmMode>;

    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess, 1>;

    using IteratorDiff = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementDiff, LayoutDiff, typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess, 1>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorDiff,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutGrad, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorDiff,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutGrad, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    using Epilogue =
            cutlass::epilogue::threadblock::Dwconv2dWgradDirectEpilogueSimt<
                    ThreadblockShape, typename Mma::Operator, ElementGrad,
                    LayoutGrad, EpilogueOutputOp>;

    using Kernel = cutlass::conv::kernel::
            ImplicitBatchedGemmDepthwiseConvolution2dWgrad<Mma, Epilogue,
                                                           ThreadblockSwizzle>;
};


template <
        typename ElementSrc,
        typename ElementDiff,
        typename ElementGrad,
        typename LayoutGrad,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename MathOperatorTag,
        int kAlignmentSrc,
        int kAlignmentDiff>
struct DefaultConvolution2dWgrad<
        ElementSrc, layout::TensorNCHW, ElementDiff, layout::TensorNCHW,
        ElementGrad, LayoutGrad, ElementAccumulator, arch::OpClassTensorOp,
        arch::Sm70, ThreadblockShape, WarpShape, gemm::GemmShape<8, 8, 4>,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, MathOperatorTag,
        kAlignmentSrc, kAlignmentDiff, ImplicitGemmMode::GEMM_NT,
        ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<8, 8, 4>;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutDiff = layout::TensorNCHW;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_NT;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementDiff, LayoutDiff, kAlignmentDiff,
            ElementAccumulator, LayoutGrad, OperatorClass, Stages,
            MathOperatorTag, true, kGemmMode>;

    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            kAlignmentSrc, 1>;

    using IteratorDiff = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementDiff, LayoutDiff, typename MmaCore::IteratorThreadMapFilter,
            kAlignmentDiff, 1>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorDiff,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutGrad, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorDiff,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutGrad, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    using Epilogue = cutlass::epilogue::threadblock::
            Dwconv2dWgradDirectEpilogueVoltaTensorOp<
                    ThreadblockShape, typename Mma::Operator, ElementGrad,
                    LayoutGrad, EpilogueOutputOp>;

    using Kernel = cutlass::conv::kernel::
            ImplicitBatchedGemmDepthwiseConvolution2dWgrad<Mma, Epilogue,
                                                           ThreadblockSwizzle>;
};


}
}
}

