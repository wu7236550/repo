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
#include "cutlass/fast_math.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"


namespace cutlass {
namespace gemm {
namespace threadblock {

namespace detail {

template <typename WarpShape>
constexpr int simt_get_warp_threads_m() {
    return (WarpShape::kM > WarpShape::kN) ? 8 : 4;
}

constexpr int simt_transpose_padding(int threads, int crosswise,
                                     int size_in_bits) {
    return (size_in_bits >= 32 ? threads / crosswise / (size_in_bits / 32)
                               : threads / crosswise * (32 / size_in_bits));
}

}


template <
        typename Shape_,
        typename WarpShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::RowMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using Operator = Operator_;

    using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kElementsPerAccess = 1;


    using SmemLayoutA = layout::ColumnMajor;
    using SmemLayoutB = layout::RowMajor;


    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            kElementsPerAccess>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            kElementsPerAccess>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::RowMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy
            >;

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using Operator = Operator_;

    using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kElementsPerAccess = 1;


    using SmemLayoutA = layout::ColumnMajor;
    using SmemLayoutB = layout::RowMajor;


    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            kElementsPerAccess>;

    using SmemThreadMapA =
            transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            SmemThreadMapA
            >;

    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            kElementsPerAccess>;

    using SmemThreadMapB =
            transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            SmemThreadMapB
            >;


    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

    static int const kPaddingM = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
    static int const kPaddingN = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::RowMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy
            >;

    using MmaPolicy =
            MmaPolicy<MmaWarpSimt,
                      MatrixShape<kPaddingM, 0>,
                      MatrixShape<0, kPaddingN>,
                      WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = ElementA_;
    using LayoutA = layout::RowMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using Operator = Operator_;

    using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kElementsPerAccess = 1;


    using SmemLayoutA = layout::ColumnMajor;
    using SmemLayoutB = layout::RowMajor;


    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
            kElementsPerAccess>;

    using SmemThreadMapA =
            transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            SmemThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
            kElementsPerAccess>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            IteratorThreadMapB>;


    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

    static int const kPaddingM = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);

    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::RowMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy
            >;

    using MmaPolicy =
            MmaPolicy<MmaWarpSimt,
                      MatrixShape<kPaddingM, 0>,
                      MatrixShape<0, 0>, WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename ElementA_,
        typename ElementB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 1>;
    using ElementA = ElementA_;
    using LayoutA = layout::ColumnMajor;
    using ElementB = ElementB_;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using Operator = Operator_;

    using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;

    static int const kElementsPerAccess = 1;


    using SmemLayoutA = layout::ColumnMajor;
    using SmemLayoutB = layout::RowMajor;


    using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
            kElementsPerAccess>;

    using SmemIteratorA = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
            IteratorThreadMapA>;

    using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
            layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
            kElementsPerAccess>;

    using SmemThreadMapB =
            transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

    using SmemIteratorB = transform::threadblock::RegularTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
            SmemThreadMapB>;


    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
    static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

    static int const kPaddingN = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 1>;
    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::RowMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy
            >;

    using MmaPolicy =
            MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                      MatrixShape<0, kPaddingN>,
                      WarpCount::kK>;
};


template <
        typename Shape_,
        typename WarpShape_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 4>, int8_t,
                      layout::ColumnMajor, int8_t, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 4>;
    using ElementA = int8_t;
    using LayoutA = layout::ColumnMajor;
    using ElementB = int8_t;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using Operator = Operator_;

    using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;


    using SmemLayoutA = layout::ColumnMajorInterleaved<4>;
    using SmemLayoutB = layout::RowMajorInterleaved<4>;


    using IteratorThreadMapA =
            transform::PitchLinear2DThreadTileStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
                    layout::PitchLinearShape<4, 4> >;

    using SmemIteratorA =
            transform::threadblock::RegularTileIterator2dThreadTile<
                    MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
                    IteratorThreadMapA>;

    using IteratorThreadMapB =
            transform::PitchLinear2DThreadTileStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
                    layout::PitchLinearShape<4, 4> >;

    using SmemIteratorB =
            transform::threadblock::RegularTileIterator2dThreadTile<
                    MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
                    IteratorThreadMapB>;


    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(4, ThreadTileM);
    static const int LaneN = cutlass::const_min(4, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 4>;

    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::ColumnMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy,
            PartitionsK
            >;

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 4>, int8_t,
                      layout::RowMajor, int8_t, layout::ColumnMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 4>;
    using ElementA = int8_t;
    using LayoutA = layout::RowMajor;
    using ElementB = int8_t;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using Operator = Operator_;

    using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;


    using SmemLayoutA = layout::ColumnMajorInterleaved<4>;
    using SmemLayoutB = layout::RowMajorInterleaved<4>;


    using IteratorThreadMapA =
            transform::PitchLinear2DThreadTileStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
                    layout::PitchLinearShape<4, 4> >;

    using SmemThreadMapA = transform::TransposePitchLinearThreadMap2DThreadTile<
            IteratorThreadMapA>;

    using SmemIteratorA =
            transform::threadblock::RegularTileIterator2dThreadTile<
                    MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
                    SmemThreadMapA>;

    using IteratorThreadMapB =
            transform::PitchLinear2DThreadTileStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
                    layout::PitchLinearShape<4, 4> >;

    using SmemThreadMapB = transform::TransposePitchLinearThreadMap2DThreadTile<
            IteratorThreadMapB>;

    using SmemIteratorB =
            transform::threadblock::RegularTileIterator2dThreadTile<
                    MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
                    SmemThreadMapB>;


    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(4, ThreadTileM);
    static const int LaneN = cutlass::const_min(4, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 4>;

    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::ColumnMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy,
            PartitionsK
            >;

    static int const kPaddingM = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
    static int const kPaddingN = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<kPaddingM, 0>,
                                MatrixShape<0, kPaddingN>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 4>, int8_t,
                      layout::RowMajor, int8_t, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 4>;
    using ElementA = int8_t;
    using LayoutA = layout::RowMajor;
    using ElementB = int8_t;
    using LayoutB = layout::RowMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using Operator = Operator_;

    using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;


    using SmemLayoutA = layout::ColumnMajorInterleaved<4>;
    using SmemLayoutB = layout::RowMajorInterleaved<4>;


    using IteratorThreadMapA =
            transform::PitchLinear2DThreadTileStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
                    layout::PitchLinearShape<4, 4> >;

    using SmemThreadMapA = transform::TransposePitchLinearThreadMap2DThreadTile<
            IteratorThreadMapA>;

    using SmemIteratorA =
            transform::threadblock::RegularTileIterator2dThreadTile<
                    MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
                    SmemThreadMapA>;

    using IteratorThreadMapB =
            transform::PitchLinear2DThreadTileStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
                    layout::PitchLinearShape<4, 4> >;

    using SmemIteratorB =
            transform::threadblock::RegularTileIterator2dThreadTile<
                    MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
                    IteratorThreadMapB>;


    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(4, ThreadTileM);
    static const int LaneN = cutlass::const_min(4, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 4>;

    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::ColumnMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy,
            PartitionsK
            >;

    static int const kPaddingM = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
    static int const kPaddingN = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<kPaddingM, 0>,
                                MatrixShape<0, 0>, WarpCount::kK>;
};

template <
        typename Shape_,
        typename WarpShape_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 4>, int8_t,
                      layout::ColumnMajor, int8_t, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, 2, Operator_> {
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    using InstructionShape = GemmShape<1, 1, 4>;
    using ElementA = int8_t;
    using LayoutA = layout::ColumnMajor;
    using ElementB = int8_t;
    using LayoutB = layout::ColumnMajor;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using OperatorClass = arch::OpClassSimt;
    static int const PartitionsK = Shape::kK / WarpShape::kK;

    using Operator = Operator_;

    using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                                Shape::kN / WarpShape::kN, PartitionsK>;

    static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
                  "Threadblock-scoped GEMM should be divisible by warp-scoped "
                  "GEMM size.");

    static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

    static int const kThreads = WarpCount::kCount * kWarpSize;


    using SmemLayoutA = layout::ColumnMajorInterleaved<4>;
    using SmemLayoutB = layout::RowMajorInterleaved<4>;


    using IteratorThreadMapA =
            transform::PitchLinear2DThreadTileStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
                    layout::PitchLinearShape<4, 4> >;

    using SmemIteratorA =
            transform::threadblock::RegularTileIterator2dThreadTile<
                    MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
                    IteratorThreadMapA>;

    using IteratorThreadMapB =
            transform::PitchLinear2DThreadTileStripminedThreadMap<
                    layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
                    layout::PitchLinearShape<4, 4> >;

    using SmemThreadMapB = transform::TransposePitchLinearThreadMap2DThreadTile<
            IteratorThreadMapB>;

    using SmemIteratorB =
            transform::threadblock::RegularTileIterator2dThreadTile<
                    MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
                    SmemThreadMapB>;


    static const int WarpNumThreadsM =
            detail::simt_get_warp_threads_m<WarpShape>();
    static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
    static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
    static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
    static_assert(!(WarpShape::kM % WarpNumThreadsM) &&
                          !(WarpShape::kN % WarpNumThreadsN),
                  "WarpShape must be divisible by ThreadTile shape.");
    static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
    static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
    static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
    static const int LaneM = cutlass::const_min(4, ThreadTileM);
    static const int LaneN = cutlass::const_min(4, ThreadTileN);
    using LaneMmaShape = cutlass::gemm::GemmShape<LaneM, LaneN, 4>;

    using Policy = cutlass::gemm::warp::MmaSimtPolicy<
            cutlass::MatrixShape<WarpNumThreadsM,
                                 WarpNumThreadsN>,
            cutlass::layout::ColumnMajorInterleaved<LaneLayout>,
            LaneMmaShape>;

    using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
            WarpShape,
            ElementA,
            SmemLayoutA,
            ElementB,
            SmemLayoutB,
            ElementC,
            LayoutC,
            Policy,
            PartitionsK
            >;

    static int const kPaddingM = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);
    static int const kPaddingN = detail::simt_transpose_padding(
            kWarpSize, Shape::kK, sizeof_bits<ElementB>::value);

    using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<0, 0>,
                                MatrixShape<0, kPaddingN>, WarpCount::kK>;
};

}
}
}
