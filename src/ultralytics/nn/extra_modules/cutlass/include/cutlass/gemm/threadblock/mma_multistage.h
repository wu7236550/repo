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

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/threadblock/mma_base.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape_,
        typename IteratorA_,
        typename SmemIteratorA_,
        cutlass::arch::CacheOperation::Kind CacheOpA,
        typename IteratorB_,
        typename SmemIteratorB_,
        cutlass::arch::CacheOperation::Kind CacheOpB,
        typename ElementC_,
        typename LayoutC_,
        typename Policy_,
        int Stages,
        typename Enable = bool>
class MmaMultistage : public MmaBase<Shape_, Policy_, Stages> {
public:
    using Base = MmaBase<Shape_, Policy_, Stages>;
    using Shape = Shape_;
    using IteratorA = IteratorA_;
    using IteratorB = IteratorB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using Policy = Policy_;

    using SmemIteratorA = SmemIteratorA_;
    using SmemIteratorB = SmemIteratorB_;

    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;


    using FragmentC = typename Policy::Operator::FragmentC;

    using Operator = typename Policy::Operator;

    using ArchTag = arch::Sm80;

    static ComplexTransform const kTransformA = Operator::kTransformA;

    static ComplexTransform const kTransformB = Operator::kTransformB;

    struct Detail {
        static_assert(
                Base::kWarpGemmIterations > 1,
                "The pipelined structure requires at least two warp-level "
                "GEMM operations.");

        static int const AsyncCopyIterationsPerStageA =
                IteratorA::ThreadMap::Iterations::kCount;

        static int const AsyncCopyIterationsPerStageB =
                IteratorB::ThreadMap::Iterations::kCount;

        static int const kStages = Stages;

        static int const kAccessesPerGroupA =
                (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) /
                Base::kWarpGemmIterations;

        static int const kAccessesPerGroupB =
                (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) /
                Base::kWarpGemmIterations;
    };

private:
    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

private:

    SmemIteratorA smem_iterator_A_;

    SmemIteratorB smem_iterator_B_;

public:
    CUTLASS_DEVICE
    MmaMultistage(
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

    CUTLASS_DEVICE
    void copy_tiles_and_advance(IteratorA& iterator_A, IteratorB& iterator_B,
                                int group_start_A = 0, int group_start_B = 0) {
        iterator_A.set_iteration_index(group_start_A *
                                       IteratorA::kAccessesPerVector);
        this->smem_iterator_A_.set_iteration_index(group_start_A);

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
            if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
                typename IteratorA::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorA::AccessType*>(
                                this->smem_iterator_A_.get());

                int const kSrcBytes =
                        sizeof_bits<typename IteratorA::Element>::value *
                        IteratorA::ThreadMap::kElementsPerAccess /
                        IteratorA::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_A.get();

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                            dst_ptr + v, gmem_ptr, iterator_A.valid());

                    ++iterator_A;
                }

                ++this->smem_iterator_A_;
            }
        }

        iterator_B.set_iteration_index(group_start_B *
                                       IteratorB::kAccessesPerVector);
        this->smem_iterator_B_.set_iteration_index(group_start_B);

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
            if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
                typename IteratorB::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorB::AccessType*>(
                                this->smem_iterator_B_.get());

                int const kSrcBytes =
                        sizeof_bits<typename IteratorB::Element>::value *
                        IteratorB::ThreadMap::kElementsPerAccess /
                        IteratorB::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_B.get();

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                            dst_ptr + v, gmem_ptr, iterator_B.valid());

                    ++iterator_B;
                }
                ++this->smem_iterator_B_;
            }
        }
    }

    CUTLASS_DEVICE
    void operator()(
            int gemm_k_iterations,
            FragmentC& accum,
            IteratorA iterator_A,
            IteratorB iterator_B,
            FragmentC const& src_accum) {

        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < Base::kStages - 1;
             ++stage, --gemm_k_iterations) {
            if (gemm_k_iterations == 0) {
                iterator_A.clear_mask();
                iterator_B.clear_mask();
            }

            iterator_A.set_iteration_index(0);
            this->smem_iterator_A_.set_iteration_index(0);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
                typename IteratorA::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorA::AccessType*>(
                                this->smem_iterator_A_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
                    int const kSrcBytes =
                            sizeof_bits<typename IteratorA::Element>::value *
                            IteratorA::ThreadMap::kElementsPerAccess /
                            IteratorA::kAccessesPerVector / 8;

                    int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                            dst_ptr + v, iterator_A.get(), iterator_A.valid());

                    ++iterator_A;
                }

                ++this->smem_iterator_A_;
            }

            iterator_B.set_iteration_index(0);
            this->smem_iterator_B_.set_iteration_index(0);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
                typename IteratorB::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorB::AccessType*>(
                                this->smem_iterator_B_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
                    int const kSrcBytes =
                            sizeof_bits<typename IteratorB::Element>::value *
                            IteratorB::ThreadMap::kElementsPerAccess /
                            IteratorB::kAccessesPerVector / 8;

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                            dst_ptr + v, iterator_B.get(), iterator_B.valid());

                    ++iterator_B;
                }

                ++this->smem_iterator_B_;
            }

            iterator_A.add_tile_offset({0, 1});
            iterator_B.add_tile_offset({1, 0});

            this->smem_iterator_A_.add_tile_offset({0, 1});
            this->smem_iterator_B_.add_tile_offset({1, 0});

            cutlass::arch::cp_async_fence();
        }

        accum = src_accum;

        cutlass::arch::cp_async_wait<Base::kStages - 2>();
        __syncthreads();

        WarpLoadedFragmentA warp_loaded_frag_A[2];
        WarpLoadedFragmentB warp_loaded_frag_B[2];
        WarpTransformedFragmentA warp_transformed_frag_A[2];
        WarpTransformedFragmentB warp_transformed_frag_B[2];

        Operator warp_mma;

        this->warp_tile_iterator_A_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.set_kgroup_index(0);

        this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (gemm_k_iterations == 0) {
            iterator_A.clear_mask();
            iterator_B.clear_mask();
        }

        int smem_write_stage_idx = Base::kStages - 1;
        int smem_read_stage_idx = 0;

        warp_mma.transform(warp_transformed_frag_A[0],
                           warp_transformed_frag_B[0], warp_loaded_frag_A[0],
                           warp_loaded_frag_B[0]);


        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > (-Base::kStages + 1);) {

            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
                 ++warp_mma_k) {

                this->warp_tile_iterator_A_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);
                this->warp_tile_iterator_B_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);

                this->warp_tile_iterator_A_.load(
                        warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_B_.load(
                        warp_loaded_frag_B[(warp_mma_k + 1) % 2]);

                ++this->warp_tile_iterator_A_;
                ++this->warp_tile_iterator_B_;

                if (warp_mma_k > 0)
                    warp_mma.transform(warp_transformed_frag_A[warp_mma_k % 2],
                                       warp_transformed_frag_B[warp_mma_k % 2],
                                       warp_loaded_frag_A[warp_mma_k % 2],
                                       warp_loaded_frag_B[warp_mma_k % 2]);

                warp_mma(accum, warp_transformed_frag_A[warp_mma_k % 2],
                         warp_transformed_frag_B[warp_mma_k % 2], accum);

                if (warp_mma_k < Base::kWarpGemmIterations - 1) {
                    int group_start_iteration_A, group_start_iteration_B;

                    group_start_iteration_A =
                            warp_mma_k * Detail::kAccessesPerGroupA;
                    group_start_iteration_B =
                            warp_mma_k * Detail::kAccessesPerGroupB;

                    copy_tiles_and_advance(iterator_A, iterator_B,
                                           group_start_iteration_A,
                                           group_start_iteration_B);
                }

                if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
                    int group_start_iteration_A, group_start_iteration_B;
                    group_start_iteration_A =
                            (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
                    group_start_iteration_B =
                            (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

                    copy_tiles_and_advance(iterator_A, iterator_B,
                                           group_start_iteration_A,
                                           group_start_iteration_B);

                    cutlass::arch::cp_async_fence();

                    arch::cp_async_wait<Base::kStages - 2>();
                    __syncthreads();

                    iterator_A.add_tile_offset({0, 1});
                    iterator_B.add_tile_offset({1, 0});

                    this->smem_iterator_A_.add_tile_offset({0, 1});
                    this->smem_iterator_B_.add_tile_offset({1, 0});

                    if (smem_write_stage_idx == (Base::kStages - 1)) {
                        this->smem_iterator_A_.add_tile_offset(
                                {0, -Base::kStages});
                        this->smem_iterator_B_.add_tile_offset(
                                {-Base::kStages, 0});
                        smem_write_stage_idx = 0;
                    } else {
                        ++smem_write_stage_idx;
                    }

                    if (smem_read_stage_idx == (Base::kStages - 1)) {
                        this->warp_tile_iterator_A_.add_tile_offset(
                                {0, -Base::kStages * Policy::kPartitionsK *
                                            Base::kWarpGemmIterations});
                        this->warp_tile_iterator_B_.add_tile_offset(
                                {-Base::kStages * Policy::kPartitionsK *
                                         Base::kWarpGemmIterations,
                                 0});
                        smem_read_stage_idx = 0;
                    } else {
                        ++smem_read_stage_idx;
                    }

                    --gemm_k_iterations;
                    if (gemm_k_iterations == 0) {
                        iterator_A.clear_mask();
                        iterator_B.clear_mask();
                    }
                }

                if (warp_mma_k + 1 == Base::kWarpGemmIterations)
                    warp_mma.transform(
                            warp_transformed_frag_A[(warp_mma_k + 1) % 2],
                            warp_transformed_frag_B[(warp_mma_k + 1) % 2],
                            warp_loaded_frag_A[(warp_mma_k + 1) % 2],
                            warp_loaded_frag_B[(warp_mma_k + 1) % 2]);
            }
        }

        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
    }
};


}
}
}

