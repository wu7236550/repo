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


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <typename Shape_,
          typename Operator_,
          int PartitionsK,
          typename Element_,
          typename OutputOp_,
          typename ConvertOp_
          >
class DirectEpilogueTensorOp {
public:
    using Shape = Shape_;
    using Operator = Operator_;

    using WarpCount = gemm::GemmShape<
            Shape::kM / Operator::Shape::kM, Shape::kN / Operator::Shape::kN, PartitionsK, >;

    static_assert(PartitionsK == 1,
                  "Direct epilogue cannot be used with when the threadblock "
                  "tile is partitioned along the K dimension.");

    using FragmentC = typename Operator::FragmentC;

    using Element = Element_;

    using Layout = layout::RowMajor;

    using OutputOp = OutputOp_;

    using ConvertOp = ConvertOp_;

    using TensorRef = TensorRef<Element, Layout::kRank, Layout>;

public:
    struct Params {

        TensorRef destination_ref;
        TensorRef source_ref;

        typename OutputOp::Params output_op;
        typename ConvertOp::Params convert_op;


        CUTLASS_HOST_DEVICE
        Params(TensorRef destination_ref_, TensorRef source_ref_,
               typename OutputOp::Params output_op_,
               typename ConvertOp::Params convert_op_)
                : destination_ref(destination_ref_),
                  source_ref(source_ref_),
                  output_op(output_op_),
                  convert_op(convert_op_) {}

        CUTLASS_HOST_DEVICE
        Params(TensorRef destination_ref_, TensorRef source_ref_,
               typename OutputOp::Params output_op_)
                : Params(destination_ref, source_ref, output_op,
                         ConvertOp::Params()) {}
    };

    struct SharedStorage {};

private:
    OutputOp output_op;
    ConvertOp convert_op;

    TensorRef destination_ref_;
    TensorRef source_ref_;

    MatrixCoord warp_origin_;

public:
    CUTLASS_DEVICE
    DirectEpilogueTensorOp(
            Params const& params,
            SharedStorage& shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx
            )
            : output_op(params.output_op),
              convert_op(params.convert_op),
              destination_ref_(params.destination_ref),
              source_ref_(params.source_ref) {

        int warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
        int warp_m = warp_mn % WarpCount::kM;
        int warp_n = warp_mn / WarpCount::kM;

        warp_origin_ = MatrixCoord{warp_m * Operator::Shape::kM,
                                   warp_n * Operator::Shape::kN};

        destination_ref_.add_coord_offset(warp_origin_);
        source_ref_.add_coord_offset(warp_origin_);
    }

    CUTLASS_DEVICE
    void operator()(
            gemm::GemmCoord
                    problem_size,
            gemm::GemmCoord
                    tb_tile_coord,
            FragmentC const& accumulators) {

        MatrixCoord thread_origin = MatrixCoord{tb_tile_coord.m() * Shape::kM,
                                                tb_tile_coord.n() * Shape::kN} +
                                    warp_origin_;

        using MmaIterations = MatrixShape<
                Operator::Shape::kM / Operator::Policy::Operator::Shape::kM,
                Operator::Shape::kN / Operator::Policy::Operator::Shape::kN>;

        int const kElementsPerAccess =
                Operator::Policy::Operator::Shape::kN / 4;
        int const kRowsPerTile = 8;
        int const kAccumulatorRows =
                Operator::Policy::Operator::Shape::kM / kRowsPerTile;

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < MmaIterations::kN; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < MmaIterations::kM; ++mma_m) {
                int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
                                      (mma_m * MmaIterations::kN + mma_n);

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < kAccumulatorRows; ++row) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int col = 0; col < kElementsPerAccess; ++col) {
                        int accum_m =
                                mma_m * Operator::Policy::Operator::Shape::kM +
                                row * kRowsPerTile;
                        int accum_n =
                                mma_n * Operator::Policy::Operator::Shape::kN +
                                col;
                        int idx = mma_accum_start + row * kElementsPerAccess +
                                  col;

                        MatrixCoord accum_coord = MatrixCoord{accum_m, accum_n};

                        MatrixCoord thread_coord = thread_origin + accum_coord;

                        if (thread_coord <
                            MatrixCoord{problem_size.m(), problem_size.n()}) {
                            typename ConvertOp::result_type converted_accum =
                                    convert_op(accumulators[idx]);

                            typename OutputOp::result_type output =
                                    output_op(converted_accum,
                                              source_ref_.at(accum_coord));

                            destination_ref_.at(accum_coord) = output;
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

