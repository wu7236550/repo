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

/**
 * \file include/cutlass/convolution/threadblock/implicit_mma_core_sm75.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"

#include "cutlass/convolution/threadblock/implicit_mma_core.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/layout/tensor.h"


namespace cutlass {
namespace conv {
namespace threadblock {

template <typename Shape_, int Threads, typename WarpThreadArrangement_,
          int ElementsPerAccess = 1>
struct PitchLinearWarpRakedThreadMapOpt {
    using TensorCoord = layout::PitchLinearCoord;

    using Shape = Shape_;

    static int const kThreads = Threads;

    static int const kElementsPerAccess = ElementsPerAccess;

    using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

    struct Detail {
        using WarpThreadArrangement = WarpThreadArrangement_;

        static int const kWarpSize = WarpThreadArrangement::kCount;

        static int const kWarpCount = kThreads / kWarpSize;

        static_assert(!(Shape::kContiguous % kElementsPerAccess),
                      "Shape must be divisible by vector length.");

        using ShapeInAccesses = layout::PitchLinearShape<
                Shape::kContiguous / kElementsPerAccess, Shape::kStrided>;

        using WarpAccessIterations = layout::PitchLinearShape<
                ShapeInAccesses::kContiguous /
                        WarpThreadArrangement::kContiguous,
                ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided>;

        static int const kWarpsContiguous =
                (WarpAccessIterations::kContiguous >= kWarpCount
                         ? kWarpCount
                         : WarpAccessIterations::kContiguous);

        static int const kWarpsStrided =
                (kWarpCount > WarpAccessIterations::kContiguous
                         ? kWarpCount / kWarpsContiguous
                         : 1);

        using WarpArrangement =
                layout::PitchLinearShape<kWarpsContiguous, kWarpsStrided>;
    };

    using Iterations =
            layout::PitchLinearShape<Detail::WarpAccessIterations::kContiguous /
                                             Detail::kWarpsContiguous,
                                     Detail::WarpAccessIterations::kStrided /
                                             Detail::kWarpsStrided>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    using Delta = layout::PitchLinearShape<
            Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
            Detail::WarpThreadArrangement::kStrided>;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        int warp_id = (thread_id / Detail::kWarpSize);
        int lane_id = (thread_id % Detail::kWarpSize);


        layout::PitchLinearCoord warp_footprint{
                Detail::WarpThreadArrangement::kContiguous *
                        Iterations::kContiguous,
                Detail::WarpThreadArrangement::kStrided * Iterations::kStrided};

        layout::PitchLinearCoord warp_offset{
                (warp_id % Detail::kWarpsContiguous),
                (warp_id / Detail::kWarpsContiguous)};

        layout::PitchLinearCoord thread_offset_in_warp{
                lane_id % Detail::WarpThreadArrangement::kContiguous,
                lane_id / Detail::WarpThreadArrangement::kContiguous};

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
                warp_footprint * warp_offset + thread_offset_in_warp;

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
                thread_offset_in_threadblock_tile_vec.contiguous() *
                        kElementsPerAccess,
                thread_offset_in_threadblock_tile_vec.strided()};

        return thread_offset_in_threadblock_tile_base;
    }
};


template <
        typename Shape_,
        typename WarpShape_,
        int kAlignmentSrc,
        typename LayoutFilter_,
        int kAlignmentFilter,
        typename ElementDst_,
        typename LayoutDst_,
        int Stages,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 16>, int8_t,
                      layout::TensorNCxHWx<32>, kAlignmentSrc, int8_t,
                      LayoutFilter_, kAlignmentFilter, ElementDst_, LayoutDst_,
                      arch::OpClassTensorOp, Stages, Operator_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 16>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<32>;
    using ElementFilter = int8_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 32;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    using Operator = Operator_;

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;


    using SmemLayoutSrc = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;


    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 1,
            SmemThreadMapSrc>;

    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter,
            0, SmemThreadMapFilter>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,
            InstructionShape,
            ElementFilter,
            SmemLayoutFilter,
            ElementSrc,
            SmemLayoutSrc,
            ElementDst,
            layout::RowMajor,
            Operator,
            PartitionsK,
            AccumulatorsInRowMajor>::Type;

    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        int kAlignmentSrc,
        typename LayoutFilter_,
        int kAlignmentFilter,
        typename ElementDst_,
        typename LayoutDst_,
        int Stages,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 16>, int8_t,
                      layout::TensorNCxHWx<32>, kAlignmentSrc, int8_t,
                      LayoutFilter_, kAlignmentFilter, ElementDst_, LayoutDst_,
                      arch::OpClassTensorOp, Stages, Operator_, true,
                      ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 16>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNCxHWx<32>;
    using ElementFilter = int8_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 32;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    using Operator = Operator_;

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;

    using SmemLayoutSrc = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;


    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            SmemThreadMapSrc>;

    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            1, SmemThreadMapFilter>;

    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,
            InstructionShape,
            ElementSrc,
            SmemLayoutSrc,
            ElementFilter,
            SmemLayoutFilter,
            ElementDst,
            layout::RowMajor,
            Operator,
            PartitionsK,
            AccumulatorsInRowMajor>::Type;

    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        bool Signed,
        typename Shape_,
        typename WarpShape_,
        int kAlignmentSrc,
        typename LayoutFilter_,
        int kAlignmentFilter,
        typename ElementDst_,
        typename LayoutDst_,
        int Stages,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 32>,
                      integer_subbyte<4, Signed>, layout::TensorNCxHWx<64>,
                      kAlignmentSrc, int4b_t, LayoutFilter_, kAlignmentFilter,
                      ElementDst_, LayoutDst_, arch::OpClassTensorOp, Stages,
                      Operator_, true> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 32>;
    using ElementSrc = integer_subbyte<4, Signed>;
    using LayoutSrc = layout::TensorNCxHWx<64>;
    using ElementFilter = int4b_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 64;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    using Operator = Operator_;

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;


    using SmemLayoutSrc = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;


    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementSrc, SmemLayoutSrc, 1,
            SmemThreadMapSrc>;

    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementFilter, SmemLayoutFilter,
            0, SmemThreadMapFilter>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,
            InstructionShape,
            ElementFilter,
            SmemLayoutFilter,
            ElementSrc,
            SmemLayoutSrc,
            ElementDst,
            layout::RowMajor,
            Operator,
            PartitionsK,
            AccumulatorsInRowMajor>::Type;

    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        bool Signed,
        typename Shape_,
        typename WarpShape_,
        int kAlignmentSrc,
        typename LayoutFilter_,
        int kAlignmentFilter,
        typename ElementDst_,
        typename LayoutDst_,
        int Stages,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 32>,
                      integer_subbyte<4, Signed>, layout::TensorNCxHWx<64>,
                      kAlignmentSrc, int4b_t, LayoutFilter_, kAlignmentFilter,
                      ElementDst_, LayoutDst_, arch::OpClassTensorOp, Stages,
                      Operator_, true, ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 32>;
    using ElementSrc = integer_subbyte<4, Signed>;
    using LayoutSrc = layout::TensorNCxHWx<64>;
    using ElementFilter = int4b_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static int const kInterleavedK = 64;
    static bool const AccumulatorsInRowMajor = true;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    using Operator = Operator_;

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value;

    static int const kWarpThreadArrangementContiguous =
            kInterleavedK / kElementsPerAccess;

    static int const kWarpThreadArrangementStrided =
            kWarpSize / kWarpThreadArrangementContiguous;

    using SmemLayoutSrc = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, kInterleavedK>;

    using SmemLayoutFilter = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, kInterleavedK>;


    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kM * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapSrc = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapSrc,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            SmemThreadMapSrc>;

    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kN * kInterleavedK,
                                     Shape::kK / kInterleavedK>,
            kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

    using SmemThreadMapFilter = transform::TransposePitchLinearThreadMap<
            IteratorThreadMapFilter,
            layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                                     kWarpThreadArrangementStrided>>;

    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            1, SmemThreadMapFilter>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape,
            InstructionShape,
            ElementSrc,
            SmemLayoutSrc,
            ElementFilter,
            SmemLayoutFilter,
            ElementDst,
            layout::RowMajor,
            Operator,
            PartitionsK,
            AccumulatorsInRowMajor>::Type;

    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        bool Signed,
        typename Shape_,
        typename WarpShape_,
        int kAlignmentSrc,
        typename LayoutFilter_,
        int kAlignmentFilter,
        typename ElementDst_,
        typename LayoutDst_,
        int Stages,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 32>,
                      integer_subbyte<4, Signed>, layout::TensorNHWC,
                      kAlignmentSrc, int4b_t, LayoutFilter_, kAlignmentFilter,
                      ElementDst_, LayoutDst_, arch::OpClassTensorOp, Stages,
                      Operator_, false, ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 32>;
    using ElementSrc = integer_subbyte<4, Signed>;
    using LayoutSrc = layout::TensorNHWC;
    using ElementFilter = int4b_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    using Operator = Operator_;

    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    static int const kWarpThreadArrangementContiguousA =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementSrc>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;

    static int const kWarpThreadArrangementContiguousB =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementSrc>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;


    using SmemLayoutSrc = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, Shape::kK>;

    using SmemLayoutFilter = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, Shape::kK>;


    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value>;

    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            IteratorThreadMapSrc>;

    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementFilter>::value>;

    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            1, IteratorThreadMapFilter>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementSrc, SmemLayoutSrc,
            ElementFilter, SmemLayoutFilter, ElementDst, layout::RowMajor,
            Operator, WarpCount::kK>::Type;

    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        int kAlignmentSrc,
        typename LayoutFilter_,
        int kAlignmentFilter,
        typename ElementDst_,
        typename LayoutDst_,
        int Stages,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, gemm::GemmShape<8, 8, 16>, int8_t,
                      layout::TensorNHWC, kAlignmentSrc, int8_t, LayoutFilter_,
                      kAlignmentFilter, ElementDst_, LayoutDst_,
                      arch::OpClassTensorOp, Stages, Operator_, false,
                      ImplicitGemmMode::GEMM_TN> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = gemm::GemmShape<8, 8, 16>;
    using ElementSrc = int8_t;
    using LayoutSrc = layout::TensorNHWC;
    using ElementFilter = int8_t;
    using LayoutFilter = LayoutFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using OperatorClass = arch::OpClassTensorOp;
    static int const PartitionsK = Shape::kK / WarpShape::kK;
    static_assert(PartitionsK == 1,
                  "Split K algorithm for convolution operator is disabled");

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, PartitionsK>;

    using Operator = Operator_;

    static int const kWarpSize =
            gemm::warp::WarpSize<arch::OpClassTensorOp>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kAccessSizeInBits = 128;

    static int const kWarpThreadArrangementContiguousA =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementSrc>::value);

    static int const kWarpThreadArrangementStridedA =
            kWarpSize / kWarpThreadArrangementContiguousA;

    static int const kWarpThreadArrangementContiguousB =
            Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementSrc>::value);

    static int const kWarpThreadArrangementStridedB =
            kWarpSize / kWarpThreadArrangementContiguousB;


    using SmemLayoutSrc = layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementSrc>::value, Shape::kK>;

    using SmemLayoutFilter = layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<ElementFilter>::value, Shape::kK>;


    using IteratorThreadMapSrc = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                     kWarpThreadArrangementStridedA>,
            kAccessSizeInBits / sizeof_bits<ElementSrc>::value>;

    using SmemIteratorSrc = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementSrc, SmemLayoutSrc, 0,
            IteratorThreadMapSrc>;

    using IteratorThreadMapFilter = PitchLinearWarpRakedThreadMapOpt<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                                     kWarpThreadArrangementStridedB>,
            kAccessSizeInBits / sizeof_bits<ElementFilter>::value>;

    using SmemIteratorFilter = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementFilter, SmemLayoutFilter,
            1, IteratorThreadMapFilter>;


    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, ElementSrc, SmemLayoutSrc,
            ElementFilter, SmemLayoutFilter, ElementDst, layout::RowMajor,
            Operator, WarpCount::kK>::Type;

    using MmaPolicy =
            gemm::threadblock::MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                         MatrixShape<0, 0>, WarpCount::kK>;
};


}
}
}
