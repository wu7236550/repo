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
#include "cutlass/numeric_conversion.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"

#include "threadblock/b2b_mma_base.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape0_,
        typename IteratorA0_,
        typename SmemIteratorA0_,
        typename IteratorB0_,
        typename SmemIteratorB0_,
        typename Shape1_,
        typename FragmentIteratorA1_,
        typename IteratorB1_,
        typename SmemIteratorB1_,
        typename ElementC_,
        typename LayoutC_,
        typename OutputOp_,
        typename Policy0_,
        typename Policy1_,
        typename TransformA0_ =
                NumericArrayConverter<typename SmemIteratorA0_::Element,
                                      typename IteratorA0_::Element,
                                      IteratorA0_::Fragment::kElements>,
        typename TransformB0_ =
                NumericArrayConverter<typename SmemIteratorB0_::Element,
                                      typename IteratorB0_::Element,
                                      IteratorB0_::Fragment::kElements>,
        typename TransformB1_ =
                NumericArrayConverter<typename SmemIteratorB1_::Element,
                                      typename IteratorB1_::Element,
                                      IteratorB1_::Fragment::kElements>,
        typename Enable = bool>
class B2bMmaPipelined
        : public B2bMmaBase<Shape0_, Shape1_, Policy0_, Policy1_, 2> {
public:
    using Base = B2bMmaBase<Shape0_, Shape1_, Policy0_, Policy1_, 2>;

    using Shape0 =
            Shape0_;
    using IteratorA0 =
            IteratorA0_;
    using IteratorB0 =
            IteratorB0_;
    using Policy0 = Policy0_;

    using SmemIteratorA0 = SmemIteratorA0_;
    using SmemIteratorB0 = SmemIteratorB0_;

    using Shape1 =
            Shape1_;
    using FragmentIteratorA1 =
            FragmentIteratorA1_;
    using IteratorB1 =
            IteratorB1_;
    using Policy1 = Policy1_;

    using SmemIteratorB1 = SmemIteratorB1_;

    using ElementC = ElementC_;
    using LayoutC = LayoutC_;

    using OutputOp = OutputOp_;

    using TransformA0 = TransformA0_;
    using TransformB0 = TransformB0_;
    using TransformB1 = TransformB1_;


    using FragmentA0 = typename IteratorA0::Fragment;

    using FragmentB0 = typename IteratorB0::Fragment;

    using FragmentC0 = typename Policy0::Operator::FragmentC;

    using Operator0 = typename Policy0::Operator;

    using FragmentB1 = typename IteratorB1::Fragment;

    using FragmentC1 = typename Policy1::Operator::FragmentC;

    using Operator1 = typename Policy1::Operator;

    using ArchTag = typename Policy0::Operator::ArchTag;

    static ComplexTransform const kTransformA0 = Operator0::kTransformA;

    static ComplexTransform const kTransformB0 = Operator0::kTransformB;

    static ComplexTransform const kTransformB1 = Operator1::kTransformB;

    static_assert((Base::kStages == 2),
                  "MmaPipelined requires kStages set to value 2");

private:
    using WarpFragmentA0 = typename Operator0::FragmentA;
    using WarpFragmentB0 = typename Operator0::FragmentB;
    using WarpFragmentA1 = typename FragmentIteratorA1::Fragment;
    using WarpFragmentB1 = typename Operator1::FragmentB;

protected:
    SmemIteratorA0 smem_iterator_A_;

    SmemIteratorB0 smem_iterator_B0_;

    SmemIteratorB1 smem_iterator_B1_;

public:
    CUTLASS_DEVICE
    B2bMmaPipelined(
            typename Base::B2bMmaSharedStorage&
                    shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx
            )
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              smem_iterator_A_(shared_storage.sharedStorage0.operand_A_ref(),
                               thread_idx),
              smem_iterator_B0_(shared_storage.sharedStorage0.operand_B_ref(),
                                thread_idx),
              smem_iterator_B1_(shared_storage.sharedStorage1.operand_B_ref(),
                                thread_idx) {

        int warp_idx_mn =
                warp_idx % (Base::WarpCount0::kM * Base::WarpCount0::kN);
        int warp_idx_k =
                warp_idx / (Base::WarpCount0::kM * Base::WarpCount0::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount0::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount0::kM;

        int tile_offset_k_0 = Base::kWarpGemmIterations0 * warp_idx_k;
        int tile_offset_k_1 = Base::kWarpGemmIterations1 * warp_idx_k;

        this->warp_tile_iterator_A0_.add_tile_offset(
                {warp_idx_m, tile_offset_k_0});
        this->warp_tile_iterator_B0_.add_tile_offset(
                {tile_offset_k_0, warp_idx_n});
        this->warp_tile_iterator_B1_.add_tile_offset(
                {tile_offset_k_1, warp_idx_n});
    }

    CUTLASS_DEVICE
    void operator()(
            int gemm_k_iterations_0,
            FragmentC1& accum,
            IteratorA0
                    iterator_A,
            IteratorB0
                    iterator_B0,
            IteratorB1
                    iterator_B1,
            FragmentC0 const& src_accum,
            OutputOp output_op_0,
            TransformA0 transform_A0 =
                    TransformA0(),
            TransformB0 transform_B0 =
                    TransformB0(),
            TransformB1 transform_B1 =
                    TransformB1()) {


        FragmentC0 accum0 = src_accum;

        FragmentA0 tb_frag_A;
        FragmentB0 tb_frag_B0;

        tb_frag_A.clear();
        tb_frag_B0.clear();

        iterator_A.load(tb_frag_A);
        iterator_B0.load(tb_frag_B0);

        ++iterator_A;
        ++iterator_B0;

        this->smem_iterator_A_.store(tb_frag_A);
        this->smem_iterator_B0_.store(tb_frag_B0);

        ++this->smem_iterator_A_;
        ++this->smem_iterator_B0_;

        __syncthreads();

        WarpFragmentA0 warp_frag_A0[2];
        WarpFragmentB0 warp_frag_B0[2];

        this->warp_tile_iterator_A0_.set_kgroup_index(0);
        this->warp_tile_iterator_B0_.set_kgroup_index(0);

        this->warp_tile_iterator_A0_.load(warp_frag_A0[0]);
        this->warp_tile_iterator_B0_.load(warp_frag_B0[0]);

        ++this->warp_tile_iterator_A0_;
        ++this->warp_tile_iterator_B0_;

        Operator0 warp_mma0;

        int smem_write_stage_idx = 1;

        if (gemm_k_iterations_0 <= 1) {
            iterator_A.clear_mask();
            iterator_B0.clear_mask();
        }

        iterator_A.load(tb_frag_A);


        // Note: The main loop does not support Base::WarpGemmIterations == 2.
        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations_0 > 0; --gemm_k_iterations_0) {

            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations0;
                 ++warp_mma_k) {

                if (warp_mma_k == Base::kWarpGemmIterations0 - 1) {
                    this->smem_iterator_A_.store(tb_frag_A);

                    this->smem_iterator_B0_.store(tb_frag_B0);

                    __syncthreads();

                    iterator_A.load(tb_frag_A);

                    ++this->smem_iterator_B0_;
                    ++this->smem_iterator_A_;

                    if (smem_write_stage_idx == 1) {
                        this->smem_iterator_A_.add_tile_offset(
                                {0, -Base::kStages});
                        this->smem_iterator_B0_.add_tile_offset(
                                {-Base::kStages, 0});
                    } else {
                        this->warp_tile_iterator_A0_.add_tile_offset(
                                {0, -Base::kStages * Policy0::kPartitionsK *
                                            Base::kWarpGemmIterations0});
                        this->warp_tile_iterator_B0_.add_tile_offset(
                                {-Base::kStages * Policy0::kPartitionsK *
                                         Base::kWarpGemmIterations0,
                                 0});
                    }

                    smem_write_stage_idx ^= 1;
                }

                this->warp_tile_iterator_A0_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations0);
                this->warp_tile_iterator_B0_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations0);

                this->warp_tile_iterator_A0_.load(
                        warp_frag_A0[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_B0_.load(
                        warp_frag_B0[(warp_mma_k + 1) % 2]);

                ++this->warp_tile_iterator_A0_;
                ++this->warp_tile_iterator_B0_;

                if (warp_mma_k == 0) {
                    iterator_B0.load(tb_frag_B0);

                    ++iterator_A;
                    ++iterator_B0;

                    if (gemm_k_iterations_0 <= 2) {
                        iterator_A.clear_mask();
                        iterator_B0.clear_mask();
                    }
                }

                warp_mma0(accum0, warp_frag_A0[warp_mma_k % 2],
                          warp_frag_B0[warp_mma_k % 2], accum0);
            }
        }


        FragmentIteratorA1 warp_tile_iterator_A1_(accum0);


        FragmentB1 tb_frag_B1;

        tb_frag_B1.clear();

        iterator_B1.load(tb_frag_B1);

        ++iterator_B1;

        this->smem_iterator_B1_.store(tb_frag_B1);

        ++this->smem_iterator_B1_;

        __syncthreads();

        WarpFragmentA1 warp_frag_A1[2];
        WarpFragmentB1 warp_frag_B1[2];

        this->warp_tile_iterator_B1_.set_kgroup_index(0);

        warp_tile_iterator_A1_.load(warp_frag_A1[0], output_op_0);
        this->warp_tile_iterator_B1_.load(warp_frag_B1[0]);

        ++warp_tile_iterator_A1_;
        ++this->warp_tile_iterator_B1_;

        Operator1 warp_mma1;

        smem_write_stage_idx = 1;

        int gemm_k_iterations_1 = FragmentIteratorA1::Policy::kIterations /
                                  Base::kWarpGemmIterations1;

        if (gemm_k_iterations_1 <= 1) {
            iterator_B1.clear_mask();
        }


        // Note: The main loop does not support Base::WarpGemmIterations == 2.
        CUTLASS_PRAGMA_UNROLL
        for (; gemm_k_iterations_1 > 0; --gemm_k_iterations_1) {

            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations1;
                 ++warp_mma_k) {

                if (warp_mma_k == Base::kWarpGemmIterations1 - 1) {

                    this->smem_iterator_B1_.store(tb_frag_B1);

                    __syncthreads();
                    ++smem_iterator_B1_;

                    if (smem_write_stage_idx == 1) {
                        smem_iterator_B1_.add_tile_offset({-Base::kStages, 0});
                    } else {
                        this->warp_tile_iterator_B1_.add_tile_offset(
                                {-Base::kStages * Policy1::kPartitionsK *
                                         Base::kWarpGemmIterations1,
                                 0});
                    }

                    smem_write_stage_idx ^= 1;
                }

                this->warp_tile_iterator_B1_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations1);

                warp_tile_iterator_A1_.load(warp_frag_A1[(warp_mma_k + 1) % 2],
                                            output_op_0);
                this->warp_tile_iterator_B1_.load(
                        warp_frag_B1[(warp_mma_k + 1) % 2]);

                ++warp_tile_iterator_A1_;
                ++this->warp_tile_iterator_B1_;

                if (warp_mma_k == 0) {
                    iterator_B1.load(tb_frag_B1);
                    ++iterator_B1;

                    if (gemm_k_iterations_1 <= 2) {
                        iterator_B1.clear_mask();
                    }
                }

                warp_mma1(accum, warp_frag_A1[warp_mma_k % 2],
                          warp_frag_B1[warp_mma_k % 2], accum);
            }
        }
    }
};


}
}
}
