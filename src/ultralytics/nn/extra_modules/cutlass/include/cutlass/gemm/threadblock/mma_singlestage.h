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
#include "cutlass/aligned_buffer.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_base.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape_,
        typename IteratorA_,
        typename SmemIteratorA_,
        typename IteratorB_,
        typename SmemIteratorB_,
        typename ElementC_,
        typename LayoutC_,
        typename Policy_,
        typename Enable = bool>
class MmaSingleStage : public MmaBase<Shape_, Policy_, 1> {
public:
    using Base = MmaBase<Shape_, Policy_, 1>;

    using Shape =
            Shape_;
    using IteratorA =
            IteratorA_;
    using IteratorB =
            IteratorB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using Policy = Policy_;

    using SmemIteratorA = SmemIteratorA_;
    using SmemIteratorB = SmemIteratorB_;


    using FragmentA = typename IteratorA::Fragment;

    using FragmentB = typename IteratorB::Fragment;

    using FragmentC = typename Policy::Operator::FragmentC;

    using Operator = typename Policy::Operator;

    using ArchTag = arch::Sm70;

    static ComplexTransform const kTransformA = Operator::kTransformA;

    static ComplexTransform const kTransformB = Operator::kTransformB;

    static_assert((Base::kStages == 1),
                  "MmaSingleStage requires kStages set to value 1");

private:
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

protected:
    SmemIteratorA smem_iterator_A_;

    SmemIteratorB smem_iterator_B_;

public:
    CUTLASS_DEVICE
    MmaSingleStage(
            typename Base::SharedStorage&
                    shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx
            )
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
              smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx) {

        int warp_idx_mn =
                warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        this->warp_tile_iterator_A_.add_tile_offset(
                {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
        this->warp_tile_iterator_B_.add_tile_offset(
                {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    }

    CUTLASS_DEVICE
    void operator()(
            int gemm_k_iterations,
            FragmentC& accum,
            IteratorA iterator_A,
            IteratorB iterator_B,
            FragmentC const& src_accum) {


        accum = src_accum;

        FragmentA tb_frag_A;
        FragmentB tb_frag_B;

        tb_frag_A.clear();
        tb_frag_B.clear();

        iterator_A.load(tb_frag_A);
        iterator_B.load(tb_frag_B);

        ++iterator_A;
        ++iterator_B;

        WarpFragmentA warp_frag_A;
        WarpFragmentB warp_frag_B;

        Operator warp_mma;

        if (gemm_k_iterations <= 1) {
            iterator_A.clear_mask();
            iterator_B.clear_mask();
        }


        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > 0; --gemm_k_iterations) {
            this->smem_iterator_A_.store(tb_frag_A);
            this->smem_iterator_B_.store(tb_frag_B);

            __syncthreads();


            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
                 ++warp_mma_k) {

                this->warp_tile_iterator_A_.set_kgroup_index(
                        warp_mma_k % Base::kWarpGemmIterations);
                this->warp_tile_iterator_B_.set_kgroup_index(
                        warp_mma_k % Base::kWarpGemmIterations);

                this->warp_tile_iterator_A_.load(warp_frag_A);
                this->warp_tile_iterator_B_.load(warp_frag_B);

                ++this->warp_tile_iterator_A_;
                ++this->warp_tile_iterator_B_;

                warp_mma(accum, warp_frag_A, warp_frag_B, accum);
            }

            this->warp_tile_iterator_A_.add_tile_offset(
                    {0, -Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                    {-Policy::kPartitionsK * Base::kWarpGemmIterations, 0});

            __syncthreads();

            iterator_A.load(tb_frag_A);
            iterator_B.load(tb_frag_B);

            ++iterator_A;
            ++iterator_B;

            if (gemm_k_iterations <= 2) {
                iterator_A.clear_mask();
                iterator_B.clear_mask();
            }
        }
    }
};


}
}
}
