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
#include "cutlass/gemm/threadblock/mma_planar_complex_base.h"


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
        int Stages,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone>
class MmaPlanarComplexPipelined
        : public MmaPlanarComplexBase<Shape_, Policy_, Stages> {
public:
    using Base = MmaPlanarComplexBase<Shape_, Policy_, Stages>;

    using Shape = Shape_;

    using IteratorA = IteratorA_;

    using IteratorB = IteratorB_;

    using ElementC = ElementC_;

    using LayoutC = LayoutC_;

    using Policy = Policy_;

    using ArchTag = typename Policy::Operator::ArchTag;

    using SmemIteratorA = SmemIteratorA_;
    using SmemIteratorB = SmemIteratorB_;

    static ComplexTransform const kTransformA = TransformA;

    static ComplexTransform const kTransformB = TransformB;


    using FragmentC =
            ArrayPlanarComplex<typename Policy::Operator::FragmentC::Element,
                               Policy::Operator::FragmentC::kElements>;

    using Operator = typename Policy::Operator;

private:
    using FragmentA = typename IteratorA::Fragment;
    using FragmentB = typename IteratorB::Fragment;
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

private:

    SmemIteratorA smem_iterator_A_;

    SmemIteratorB smem_iterator_B_;

public:
    CUTLASS_DEVICE
    MmaPlanarComplexPipelined(
            typename Base::SharedStorage& shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx)
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

private:
    CUTLASS_DEVICE
    void warp_mma_planar_complex(Operator& warp_mma, FragmentC& accum,
                                 WarpFragmentA const& real_A,
                                 WarpFragmentA const& imag_A,
                                 WarpFragmentB const& real_B,
                                 WarpFragmentB const& imag_B) {
        cutlass::negate<Array<typename WarpFragmentB::Element,
                              WarpFragmentB::kElements>>
                neg_op_B;

        WarpFragmentB neg_real_B = neg_op_B(real_B);
        WarpFragmentB neg_imag_B = neg_op_B(imag_B);

        warp_mma(accum.real, real_A, real_B, accum.real);

        if (kTransformB == ComplexTransform::kNone) {
            warp_mma(accum.imag, real_A, imag_B, accum.imag);
        } else {
            warp_mma(accum.imag, real_A, neg_imag_B, accum.imag);
        }

        if (kTransformA == ComplexTransform::kNone) {
            warp_mma(accum.imag, imag_A, real_B, accum.imag);
        } else {
            warp_mma(accum.imag, imag_A, neg_real_B, accum.imag);
        }

        if (kTransformA == ComplexTransform::kNone ^
            kTransformB == ComplexTransform::kNone) {
            warp_mma(accum.real, imag_A, imag_B, accum.real);
        } else {
            warp_mma(accum.real, imag_A, neg_imag_B, accum.real);
        }
    }

public:
    CUTLASS_DEVICE
    void operator()(
            int gemm_k_iterations,
            FragmentC& accum,
            IteratorA iterator_A_real,
            IteratorA iterator_A_imag,
            IteratorB iterator_B_real,
            IteratorB iterator_B_imag,
            FragmentC const& src_accum) {

        accum = src_accum;

        FragmentA tb_frag_A_real;
        FragmentA tb_frag_A_imag;

        FragmentB tb_frag_B_real;
        FragmentB tb_frag_B_imag;

        tb_frag_A_real.clear();
        tb_frag_A_imag.clear();

        tb_frag_B_real.clear();
        tb_frag_B_imag.clear();

        iterator_A_real.load(tb_frag_A_real);
        iterator_A_imag.load(tb_frag_A_imag);

        iterator_B_real.load(tb_frag_B_real);
        iterator_B_imag.load(tb_frag_B_imag);

        ++iterator_A_real;
        ++iterator_A_imag;

        ++iterator_B_real;
        ++iterator_B_imag;

        this->smem_iterator_A_.store(tb_frag_A_real);
        this->smem_iterator_A_.store_with_pointer_offset(
                tb_frag_A_imag, Base::SharedStorage::kImaginaryStrideA);

        this->smem_iterator_B_.store(tb_frag_B_real);
        this->smem_iterator_B_.store_with_pointer_offset(
                tb_frag_B_imag, Base::SharedStorage::kImaginaryStrideB);

        ++this->smem_iterator_A_;
        ++this->smem_iterator_B_;

        __syncthreads();

        WarpFragmentA warp_frag_real_A[2];
        WarpFragmentA warp_frag_imag_A[2];

        WarpFragmentB warp_frag_real_B[2];
        WarpFragmentB warp_frag_imag_B[2];

        this->warp_tile_iterator_A_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.set_kgroup_index(0);

        this->warp_tile_iterator_A_.load(warp_frag_real_A[0]);
        this->warp_tile_iterator_A_.load_with_pointer_offset(
                warp_frag_imag_A[0], Base::SharedStorage::kImaginaryStrideA);

        this->warp_tile_iterator_B_.load(warp_frag_real_B[0]);
        this->warp_tile_iterator_B_.load_with_pointer_offset(
                warp_frag_imag_B[0], Base::SharedStorage::kImaginaryStrideB);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        Operator warp_mma;

        int smem_write_stage_idx = 1;

        if (gemm_k_iterations <= 1) {
            iterator_A_real.clear_mask();
            iterator_A_imag.clear_mask();

            iterator_B_real.clear_mask();
            iterator_B_imag.clear_mask();
        }



        // Note: The main loop does not support Base::kWarpGemmIterations == 2.
        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > 0; --gemm_k_iterations) {

            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
                 ++warp_mma_k) {

                if (warp_mma_k == Base::kWarpGemmIterations - 1) {
                    this->smem_iterator_A_.store(tb_frag_A_real);
                    this->smem_iterator_A_.store_with_pointer_offset(
                            tb_frag_A_imag,
                            Base::SharedStorage::kImaginaryStrideA);

                    this->smem_iterator_B_.store(tb_frag_B_real);
                    this->smem_iterator_B_.store_with_pointer_offset(
                            tb_frag_B_imag,
                            Base::SharedStorage::kImaginaryStrideB);

                    __syncthreads();

                    ++this->smem_iterator_B_;
                    ++this->smem_iterator_A_;

                    if (smem_write_stage_idx == 1) {
                        this->smem_iterator_A_.add_tile_offset(
                                {0, -Base::kStages});
                        this->smem_iterator_B_.add_tile_offset(
                                {-Base::kStages, 0});
                    } else {
                        this->warp_tile_iterator_A_.add_tile_offset(
                                {0, -Base::kStages * Policy::kPartitionsK *
                                            Base::kWarpGemmIterations});
                        this->warp_tile_iterator_B_.add_tile_offset(
                                {-Base::kStages * Policy::kPartitionsK *
                                         Base::kWarpGemmIterations,
                                 0});
                    }

                    smem_write_stage_idx ^= 1;
                }

                this->warp_tile_iterator_A_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);
                this->warp_tile_iterator_B_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);

                this->warp_tile_iterator_A_.load(
                        warp_frag_real_A[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_A_.load_with_pointer_offset(
                        warp_frag_imag_A[(warp_mma_k + 1) % 2],
                        Base::SharedStorage::kImaginaryStrideA);

                this->warp_tile_iterator_B_.load(
                        warp_frag_real_B[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_B_.load_with_pointer_offset(
                        warp_frag_imag_B[(warp_mma_k + 1) % 2],
                        Base::SharedStorage::kImaginaryStrideB);

                ++this->warp_tile_iterator_A_;
                ++this->warp_tile_iterator_B_;

                if (warp_mma_k == 0) {
                    iterator_A_real.load(tb_frag_A_real);
                    iterator_A_imag.load(tb_frag_A_imag);

                    iterator_B_real.load(tb_frag_B_real);
                    iterator_B_imag.load(tb_frag_B_imag);

                    ++iterator_A_real;
                    ++iterator_A_imag;
                    ++iterator_B_real;
                    ++iterator_B_imag;

                    if (gemm_k_iterations <= 2) {
                        iterator_A_real.clear_mask();
                        iterator_A_imag.clear_mask();
                        iterator_B_real.clear_mask();
                        iterator_B_imag.clear_mask();
                    }
                }

                warp_mma_planar_complex(warp_mma, accum,
                                        warp_frag_real_A[warp_mma_k % 2],
                                        warp_frag_imag_A[warp_mma_k % 2],
                                        warp_frag_real_B[warp_mma_k % 2],
                                        warp_frag_imag_B[warp_mma_k % 2]);
            }
        }
    }
};


}
}
}

