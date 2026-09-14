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
#include "cutlass/array.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/threadblock/dwconv2d_predicated_tile_iterator.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <typename Shape_,
          typename Operator_,
          typename Element_,
          typename Layout_,
          typename OutputOp_
          >
class Dwconv2dWgradDirectEpilogueSimt {
public:
    using Shape = Shape_;
    using Operator = Operator_;
    using Layout = Layout_;
    using Element = Element_;

    using WarpCount = gemm::GemmShape<Shape::kM / Operator::Shape::kM,
                                      Shape::kN / Operator::Shape::kN,
                                      Shape::kK / Operator::Shape::kK>;

    static_assert(WarpCount::kK == 1,
                  "Depthwise convolution direct epilogue cannot be used with "
                  "when the threadblock "
                  "tile is partitioned along the K dimension.");

    using AccumulatorTile = typename Operator::FragmentC;

    using OutputOp = OutputOp_;

    using TensorRef = TensorRef<Element, Layout>;

    using Policy = typename Operator::Policy;
    using WarpShape = typename Policy::WarpShape;
    using LaneLayout = typename Policy::LaneLayout;
    using LaneMmaShape = typename Policy::LaneMmaShape;

    using LogicalLayout = cutlass::layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using TensorCoord = typename Layout::TensorCoord;

    struct Detail {
        static CUTLASS_DEVICE LogicalCoord get_lane_offset(int lane_idx) {
            auto lane_layout = Policy::get_lane_layout();
            LogicalCoord lane_offset =
                    lane_layout.inverse(lane_idx) *
                    LogicalCoord(LaneMmaShape::kM, LaneMmaShape::kN);
            return lane_offset;
        }
    };

    using OutputTileIterator =
            Dwconv2dPredicatedAccessTileIterator<Shape, Operator, Element,
                                                 Layout, Detail>;

    using AccumulatorTileShape =
            MatrixShape<Operator::Shape::kM / WarpShape::kRow,
                        Operator::Shape::kN / WarpShape::kColumn>;

    using MmaIterations =
            MatrixShape<AccumulatorTileShape::kRow / LaneMmaShape::kM,
                        AccumulatorTileShape::kColumn / LaneMmaShape::kN>;

public:
    struct SharedStorage {};

private:
public:
    CUTLASS_DEVICE
    Dwconv2dWgradDirectEpilogueSimt(
            SharedStorage& ,
            int ,
            int ,
            int
    ) {}

    CUTLASS_DEVICE
    void operator()(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const& accumulators) {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < MmaIterations::kRow; ++mma_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_n = 0; mma_n < MmaIterations::kColumn; ++mma_n) {
                int mma_accum_start = mma_m * LaneMmaShape::kM *
                                              AccumulatorTileShape::kColumn +
                                      mma_n * LaneMmaShape::kN;
                LogicalCoord mma_accum_coord(
                        mma_m * WarpShape::kRow * LaneMmaShape::kM,
                        mma_n * WarpShape::kColumn * LaneMmaShape::kN);

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < LaneMmaShape::kM; ++row) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int col = 0; col < LaneMmaShape::kN; ++col) {
                        int idx = mma_accum_start +
                                  row * AccumulatorTileShape::kColumn + col;
                        LogicalCoord accum_coord =
                                mma_accum_coord + MatrixCoord(row, col);

                        TensorCoord coord;
                        if (destination_iterator.valid(coord, accum_coord)) {
                            using FragmentAccumulator =
                                    typename OutputOp::FragmentAccumulator;
                            auto frag_ptr = reinterpret_cast<
                                    FragmentAccumulator const*>(
                                    &accumulators[idx]);
                            auto output = output_op(*frag_ptr);
                            auto pointer = destination_iterator.get(coord);
                            auto output_ptr =
                                    reinterpret_cast<Element*>(&output[0]);
                            ::atomicAdd(reinterpret_cast<Element*>(pointer),
                                        *output_ptr);
                        }
                    }
                }
            }
        }
    }
};


}
}
}

