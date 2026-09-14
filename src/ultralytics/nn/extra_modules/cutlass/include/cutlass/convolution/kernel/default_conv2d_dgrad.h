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
 * \file include/cutlass/convolution/kernel/default_conv2d_dgrad.h
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

#include "cutlass/convolution/kernel/implicit_gemm_nt_precomp_convolution.h"
#include "cutlass/convolution/kernel/implicit_gemm_tn_precomp_convolution.h"
#include "cutlass/convolution/kernel/implicit_batched_gemm_tn_dwconv2d.h"

#include "cutlass/convolution/threadblock/conv2d_tile_iterator_nt.h"
#include "cutlass/convolution/threadblock/conv2d_tile_iterator_nt_src_dgrad_precomp.h"
#include "cutlass/convolution/threadblock/conv2d_tile_iterator_tn_dgrad_nhwc_precomp.h"
#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn_filter_fprop_precomp.h"
#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn_filter_dgrad_precomp.h"
#include "cutlass/convolution/threadblock/dwconv2d_tile_iterator_tn.h"

#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_simt.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm70.h"
#include "cutlass/convolution/threadblock/implicit_mma_core_sm75.h"

#include "cutlass/convolution/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/convolution_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/convolution_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/dwconv2d_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/dwconv2d_epilogue_volta_tensor_op.h"

#include "cutlass/epilogue/thread/bias_add_linear_combination_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_relu_clamp.h"
#include "cutlass/epilogue/thread/bias_add_linear_combination_hswish_clamp.h"
#include "cutlass/epilogue/thread/linear_combination.h"


namespace cutlass {
namespace conv {
namespace kernel {


template <typename ElementSrc,
          typename LayoutSrc,
          typename ElementFilter,
          typename LayoutFilter,
          typename ElementDst,
          typename LayoutDst,
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
          int AlignmentFilter,
          SpecialOptimizeDesc SpecialOpt = SpecialOptimizeDesc::NONE,
          ImplicitGemmMode GemmMode = ImplicitGemmMode::GEMM_NT,
          ConvType ConvolutionType = ConvType::kConvolution>
struct DefaultConvolution2dDgrad;



template <typename LayoutDst,
          typename ElementDst,
          typename ArchTag,
          typename ElementAccumulator,
          typename ThreadblockShape,
          typename WarpShape,
          typename EpilogueOutputOp,
          typename ThreadblockSwizzle,
          int Stages,
          typename MathOperatorTag,
          int kAlignmentSrc,
          int kAlignmentFilter,
          SpecialOptimizeDesc SpecialOpt>
struct DefaultConvolution2dDgrad<
        int8_t, layout::TensorNCxHWx<4>, int8_t, layout::TensorKxRSCx<4>,
        ElementDst, LayoutDst, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 4>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentFilter, SpecialOpt> {
    using InstructionShape = gemm::GemmShape<1, 1, 4>;
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<4>;
    using LayoutFilter = layout::TensorKxRSCx<4>;
    static const int kStages = Stages;

    using OperatorClass = arch::OpClassSimt;
    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            Operator, true>;

    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    SpecialOpt>;

    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, 4,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2NHW_Col2C>>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaNtPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaNtPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    static int const kEpilogueElementsPerAccess =
            cutlass::platform::is_same<LayoutDst, layout::TensorNCHW>::value
                    ? 1
                    : 4;

    using Epilogue =
            typename cutlass::epilogue::threadblock::ConvolutionEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};



template <
        typename LayoutDst,
        typename ElementDst,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Interleaved,
        typename MathOperatorTag,
        int kAlignmentSrc,
        int kAlignmentFilter, SpecialOptimizeDesc SpecialOpt>
struct DefaultConvolution2dDgrad<
        int8_t, layout::TensorNCxHWx<Interleaved>, int8_t,
        layout::TensorKxRSCx<Interleaved>, ElementDst, LayoutDst,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, 2,
        MathOperatorTag, kAlignmentSrc, kAlignmentFilter, SpecialOpt> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorKxRSCx<Interleaved>;

    static int const kInterleavedK = Interleaved;

    using OperatorClass = arch::OpClassTensorOp;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    using MmaCore = cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            MathOperatorTag, true>;

    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    SpecialOpt>;

    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, kInterleavedK,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2NHW_Col2C>>;

    using Mma = cutlass::conv::threadblock::MmaNtPrecompPipelined<
            typename MmaCore::Shape, IteratorSrc,
            typename MmaCore::SmemIteratorSrc, IteratorFilter,
            typename MmaCore::SmemIteratorFilter, ElementAccumulator, LayoutDst,
            typename MmaCore::MmaPolicy>;

    static int const kEpilogueElementsPerAccess =
            64 / sizeof_bits<ElementDst>::value;

    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};



template <
        typename ElementDst,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Interleaved,
        typename MathOperatorTag,
        int kAlignmentSrc,
        int kAlignmentFilter, SpecialOptimizeDesc SpecialOpt>
struct DefaultConvolution2dDgrad<
        int8_t, layout::TensorNCxHWx<Interleaved>, int8_t,
        layout::TensorKxRSCx<Interleaved>, ElementDst, layout::TensorNCxHWx<4>,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, 2,
        MathOperatorTag, kAlignmentSrc, kAlignmentFilter, SpecialOpt> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<Interleaved>;
    using LayoutFilter = layout::TensorKxRSCx<Interleaved>;
    using LayoutDst = layout::TensorNCxHWx<4>;

    using OperatorClass = arch::OpClassTensorOp;

    static int const kInterleavedK = Interleaved;

    static_assert(kAlignmentSrc == 128 / sizeof_bits<ElementSrc>::value,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter == 128 / sizeof_bits<ElementFilter>::value,
                  "Alignment must match thread data map's vector length");

    using MmaCore = cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass, 2,
            MathOperatorTag, true>;

    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc,
                    MmaCore::IteratorThreadMapSrc::kElementsPerAccess,
                    cutlass::conv::threadblock::TileMap<
                            LayoutSrc, cutlass::conv::threadblock::TileMapType::
                                               kRow2C_Col2NHW>,
                    SpecialOpt>;

    using IteratorFilter = cutlass::conv::threadblock::Conv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kM>,
            ElementFilter, LayoutFilter, kInterleavedK,
            typename MmaCore::IteratorThreadMapFilter,
            MmaCore::IteratorThreadMapFilter::kElementsPerAccess,
            cutlass::conv::threadblock::TileMap<
                    LayoutFilter,
                    cutlass::conv::threadblock::TileMapType::kRow2NHW_Col2C>>;

    using Mma = cutlass::conv::threadblock::MmaNtPrecompPipelined<
            typename MmaCore::Shape, IteratorSrc,
            typename MmaCore::SmemIteratorSrc, IteratorFilter,
            typename MmaCore::SmemIteratorFilter, ElementAccumulator, LayoutDst,
            typename MmaCore::MmaPolicy>;

    static int const kEpilogueElementsPerAccess =
            32 / sizeof_bits<ElementDst>::value;

    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        kEpilogueElementsPerAccess>::Epilogue;

    using Kernel = cutlass::conv::kernel::ImplicitGemmNtPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};

template <
        typename ElementDst,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename InstructionShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename MathOperatorTag,
        int kAlignmentSrc,
        int kAlignmentFilter, SpecialOptimizeDesc SpecialOpt>
struct DefaultConvolution2dDgrad<
        int8_t, layout::TensorNHWC, int8_t,
        layout::TensorCKxRSx<kAlignmentFilter>, ElementDst, layout::TensorNHWC,
        ElementAccumulator, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
        WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, MathOperatorTag, kAlignmentSrc, kAlignmentFilter, SpecialOpt,
        ImplicitGemmMode::GEMM_TN> {
    using ElementSrc = int8_t;
    using ElementFilter = int8_t;
    using LayoutSrc = layout::TensorNHWC;
    using LayoutFilter = layout::TensorCKxRSx<kAlignmentFilter>;
    using LayoutDst = layout::TensorNHWC;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;

    static_assert(kAlignmentSrc == kAlignmentFilter,
                  "kAlignmentSrc and kAlignmentFilter must be the same");

    static_assert(kAlignmentSrc % 4 == 0,
                  "Alignment must match thread data map's vector length");

    static_assert(kAlignmentFilter % 4 == 0,
                  "Alignment must match thread data map's vector length");

    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, false, ImplicitGemmMode::GEMM_TN>;

    using IteratorSrc =
            cutlass::conv::threadblock::Conv2dTileSrcIteratorDgradPrecompNHWC<
                    cutlass::MatrixShape<ThreadblockShape::kM,
                                         ThreadblockShape::kK>,
                    ElementSrc, LayoutSrc,
                    typename MmaCore::IteratorThreadMapSrc, kAlignmentSrc,
                    SpecialOpt>;

    using IteratorFilter =
            cutlass::conv::threadblock::Conv2dTileFilterIteratorDgradCKxRSx<
                    cutlass::MatrixShape<ThreadblockShape::kK,
                                         ThreadblockShape::kN>,
                    ElementFilter, LayoutFilter,
                    typename MmaCore::IteratorThreadMapFilter,
                    kAlignmentFilter>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    using Epilogue = typename cutlass::epilogue::threadblock::
            ConvolutionEpilogueTensorOp<ThreadblockShape, LayoutDst, LayoutDst,
                                        typename Mma::Operator,
                                        EpilogueOutputOp,
                                        EpilogueOutputOp::kCount>::Epilogue;

    using Kernel = cutlass::conv::kernel::ImplicitGemmTnPrecompConvolution<
            Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};


template <
        typename ElementDst,
        typename LayoutDst,
        typename ArchTag,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename MathOperatorTag,
        int kAlignmentSrc,
        int kAlignmentFilter>
struct DefaultConvolution2dDgrad<
        float, layout::TensorNCHW, float, layout::TensorNCHW, ElementDst,
        LayoutDst, ElementAccumulator, arch::OpClassSimt, ArchTag,
        ThreadblockShape, WarpShape, gemm::GemmShape<1, 1, 1>, EpilogueOutputOp,
        ThreadblockSwizzle, Stages, MathOperatorTag, kAlignmentSrc,
        kAlignmentFilter, SpecialOptimizeDesc::NONE, ImplicitGemmMode::GEMM_TN,
        ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<1, 1, 1>;
    using ElementSrc = float;
    using ElementFilter = float;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutFilter = layout::TensorNCHW;
    using OperatorClass = arch::OpClassSimt;
    static const int kStages = Stages;
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_TN;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true, kGemmMode>;

    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            MmaCore::IteratorThreadMapSrc::kElementsPerAccess>;

    using IteratorFilter =
            cutlass::conv::threadblock::Dwconv2dTileFilterIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementFilter, LayoutFilter,
                    typename MmaCore::IteratorThreadMapFilter,
                    MmaCore::IteratorThreadMapFilter::kElementsPerAccess>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    static int const kEpilogueElementsPerAccess = 1;

    using Epilogue =
            typename cutlass::epilogue::threadblock::Dwconv2dEpilogueSimt<
                    ThreadblockShape, LayoutDst, LayoutDst,
                    typename Mma::Operator, EpilogueOutputOp,
                    kEpilogueElementsPerAccess>::Epilogue;

    using Kernel =
            cutlass::conv::kernel::ImplicitBatchedGemmTnDepthwiseConvolution<
                    Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};


template <
        typename ElementSrc,
        typename ElementFilter,
        typename ElementDst,
        typename LayoutDst,
        typename ElementAccumulator,
        typename ThreadblockShape,
        typename WarpShape,
        typename EpilogueOutputOp,
        typename ThreadblockSwizzle,
        int Stages,
        typename MathOperatorTag,
        int kAlignmentSrc,
        int kAlignmentFilter>
struct DefaultConvolution2dDgrad<
        ElementSrc, layout::TensorNCHW, ElementFilter, layout::TensorNCHW,
        ElementDst, LayoutDst, ElementAccumulator, arch::OpClassTensorOp,
        arch::Sm70, ThreadblockShape, WarpShape, gemm::GemmShape<8, 8, 4>,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, MathOperatorTag,
        kAlignmentSrc, kAlignmentFilter, SpecialOptimizeDesc::NONE,
        ImplicitGemmMode::GEMM_TN, ConvType::kDepthwiseConvolution> {
    using InstructionShape = gemm::GemmShape<8, 8, 4>;
    using LayoutSrc = layout::TensorNCHW;
    using LayoutFilter = layout::TensorNCHW;
    using OperatorClass = arch::OpClassTensorOp;
    static const int kStages = Stages;
    static const ImplicitGemmMode kGemmMode = ImplicitGemmMode::GEMM_TN;
    static const ConvType kConvolutionType = ConvType::kDepthwiseConvolution;

    using MmaCore = typename cutlass::conv::threadblock::DefaultMmaCore<
            ThreadblockShape, WarpShape, InstructionShape, ElementSrc,
            LayoutSrc, kAlignmentSrc, ElementFilter, LayoutFilter,
            kAlignmentFilter, ElementAccumulator, LayoutDst, OperatorClass,
            Stages, MathOperatorTag, true, kGemmMode>;

    using IteratorSrc = cutlass::conv::threadblock::Dwconv2dTileIterator<
            cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
            ElementSrc, LayoutSrc, typename MmaCore::IteratorThreadMapSrc,
            kAlignmentSrc>;

    using IteratorFilter =
            cutlass::conv::threadblock::Dwconv2dTileFilterIteratorDgradPrecomp<
                    cutlass::MatrixShape<MmaCore::Shape::kK,
                                         MmaCore::Shape::kN>,
                    ElementFilter, LayoutFilter,
                    typename MmaCore::IteratorThreadMapFilter,
                    kAlignmentFilter>;

    using MmaPipelineSingleStage =
            cutlass::conv::threadblock::MmaTnPrecompSingleStage<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using MmaPipelineTwoStages =
            cutlass::conv::threadblock::MmaTnPrecompPipelined<
                    typename MmaCore::Shape, IteratorSrc,
                    typename MmaCore::SmemIteratorSrc, IteratorFilter,
                    typename MmaCore::SmemIteratorFilter, ElementAccumulator,
                    LayoutDst, typename MmaCore::MmaPolicy>;

    using Mma = typename cutlass::platform::conditional<
            (kStages == 1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

    using Epilogue = typename cutlass::epilogue::threadblock::
            Dwconv2dEpilogueVoltaTensorOp<ThreadblockShape, LayoutDst,
                                          LayoutDst, typename Mma::Operator,
                                          EpilogueOutputOp,
                                          EpilogueOutputOp::kCount>::Epilogue;

    using Kernel =
            cutlass::conv::kernel::ImplicitBatchedGemmTnDepthwiseConvolution<
                    Mma, Epilogue, ThreadblockSwizzle, conv::Operator::kDgrad>;
};


}
}
}

