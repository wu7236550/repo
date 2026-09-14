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

#include "cutlass/gemm/threadblock/mma_sparse_base.h"


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
        typename IteratorE_,
        typename SmemIteratorE_,
        cutlass::arch::CacheOperation::Kind CacheOpE,
        typename Policy_,
        int Stages,
        typename Enable = bool>
class SparseMmaMultistage : public SparseMmaBase<Shape_, Policy_, Stages> {
public:
    using Base = SparseMmaBase<Shape_, Policy_, Stages>;
    using Shape = Shape_;
    using IteratorA = IteratorA_;
    using IteratorB = IteratorB_;
    using IteratorE = IteratorE_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using Policy = Policy_;

    using SmemIteratorA = SmemIteratorA_;
    using SmemIteratorB = SmemIteratorB_;
    using SmemIteratorE = SmemIteratorE_;

    static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
    static cutlass::arch::CacheOperation::Kind const kCacheOpE = CacheOpE;

    static int const kSparse = Policy::Operator::kSparse;
    static int const kMetaSizeInBits = Policy::Operator::kMetaSizeInBits;
    static int const kMaxID2 = Policy::Operator::kMaxID2;
    static int const kElementsPerElementE =
            Policy::Operator::kElementsPerElementE;


    using FragmentC = typename Policy::Operator::FragmentC;

    using Operator = typename Policy::Operator;

    using ElementE = typename IteratorE::Element;

    using LayoutE = typename IteratorE::Layout;

    using ArchTag = arch::Sm80;

    static ComplexTransform const kTransformA = Operator::kTransformA;

    static ComplexTransform const kTransformB = Operator::kTransformB;

    struct Detail {
        static_assert(
                Base::kWarpGemmIterations > 1,
                "The pipelined structure requires at least two warp-level "
                "GEMM operations.");

        static int const TBLDGSTSIterationsA =
                IteratorA::ThreadMap::Iterations::kCount;

        static int const TBLDGSTSIterationsB =
                IteratorB::ThreadMap::Iterations::kCount;

        static int const TBLDGSTSIterationsE =
                IteratorE::ThreadMap::Iterations::kCount;

        static int const kStages = Stages;

        static int const kAccessesPerGroupA =
                (TBLDGSTSIterationsA + Base::kWarpGemmIterations - 1) /
                Base::kWarpGemmIterations;

        static int const kAccessesPerGroupB =
                (TBLDGSTSIterationsB + Base::kWarpGemmIterations - 1) /
                Base::kWarpGemmIterations;

        static int const kAccessesPerGroupE =
                (TBLDGSTSIterationsE + Base::kWarpGemmIterations - 1) /
                Base::kWarpGemmIterations;

        static int const kValidWarps = IteratorE::ThreadMap::kThreads / 32;

        static int const kBBufferSize =
                ((sizeof(typename Operator::ElementC) == 4) &&
                 ((platform::is_same<
                           typename Operator::Policy::Operator::ElementA,
                           typename Operator::ElementA>::value &&
                   platform::is_same<
                           typename Operator::Policy::Operator::ElementB,
                           typename Operator::ElementB>::value)) &&
                 (Operator::Shape::kM >= 64 && Operator::Shape::kN >= 64))
                        ? 1
                        : 2;
    };

private:
    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
    using WarpFragmentE = typename Operator::FragmentE;

private:

    SmemIteratorA smem_iterator_A_;

    SmemIteratorB smem_iterator_B_;

    SmemIteratorE smem_iterator_E_;

    bool is_warp_valid_;

public:
    CUTLASS_DEVICE
    SparseMmaMultistage(
            typename Base::SharedStorage& shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx)
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
              smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
              smem_iterator_E_(shared_storage.operand_E_ref(), thread_idx) {
        is_warp_valid_ = warp_idx < Detail::kValidWarps;


        int warp_idx_mn =
                warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        this->warp_tile_iterator_A_.add_tile_offset(
                {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
        this->warp_tile_iterator_B_.add_tile_offset(
                {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
        this->warp_tile_iterator_E_.add_tile_offset(
                {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    }

    CUTLASS_DEVICE
    void copy_tiles_and_advance(IteratorA& iterator_A, IteratorB& iterator_B,
                                IteratorE& iterator_E, int group_start_A = 0,
                                int group_start_B = 0, int group_start_E = 0) {
        iterator_A.set_iteration_index(group_start_A *
                                       IteratorA::kAccessesPerVector);
        this->smem_iterator_A_.set_iteration_index(group_start_A);

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
            if (group_start_A + j < Detail::TBLDGSTSIterationsA) {
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

                    cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
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
            if (group_start_B + j < Detail::TBLDGSTSIterationsB) {
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

                    cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                            dst_ptr + v, gmem_ptr, iterator_B.valid());

                    ++iterator_B;
                }
                ++this->smem_iterator_B_;
            }
        }

        iterator_E.set_iteration_index(group_start_E);
        this->smem_iterator_E_.set_iteration_index(group_start_E);

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupE; ++j) {
            if (group_start_E + j < Detail::TBLDGSTSIterationsE) {
                typename IteratorE::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorE::AccessType*>(
                                this->smem_iterator_E_.get());

                int const kSrcBytes =
                        sizeof_bits<typename IteratorE::Element>::value *
                        IteratorE::ThreadMap::kElementsPerAccess / 8;

                auto gmem_ptr = iterator_E.get();

                cutlass::arch::cp_async<kSrcBytes, kCacheOpE>(
                        dst_ptr, gmem_ptr,
                        iterator_E.valid() && is_warp_valid_);

                ++iterator_E;
                ++this->smem_iterator_E_;
            }
        }
    }

    CUTLASS_DEVICE
    void operator()(
            int gemm_k_iterations,
            FragmentC& accum,
            IteratorA iterator_A,
            IteratorB iterator_B,
            IteratorE iterator_E,
            FragmentC const& src_accum) {

        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < Base::kStages - 1;
             ++stage, --gemm_k_iterations) {
            if (gemm_k_iterations == 0) {
                iterator_A.clear_mask();
                iterator_B.clear_mask();
                iterator_E.clear_mask();
            }

            iterator_A.set_iteration_index(0);
            this->smem_iterator_A_.set_iteration_index(0);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::TBLDGSTSIterationsA; ++j) {
                typename IteratorA::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorA::AccessType*>(
                                this->smem_iterator_A_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
                    int const kSrcBytes =
                            sizeof_bits<typename IteratorA::Element>::value *
                            IteratorA::ThreadMap::kElementsPerAccess /
                            IteratorA::kAccessesPerVector / 8;

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                            dst_ptr + v, iterator_A.get(), iterator_A.valid());

                    ++iterator_A;
                }

                ++this->smem_iterator_A_;
            }

            iterator_B.set_iteration_index(0);
            this->smem_iterator_B_.set_iteration_index(0);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::TBLDGSTSIterationsB; ++j) {
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

            iterator_E.set_iteration_index(0);
            this->smem_iterator_E_.set_iteration_index(0);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::TBLDGSTSIterationsE; ++j) {
                typename IteratorE::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorE::AccessType*>(
                                this->smem_iterator_E_.get());

                int const kSrcBytes =
                        sizeof_bits<typename IteratorE::Element>::value *
                        IteratorE::ThreadMap::kElementsPerAccess / 8;
                if (is_warp_valid_)
                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpE>(
                            dst_ptr, iterator_E.get(), iterator_E.valid());

                ++iterator_E;

                ++this->smem_iterator_E_;
            }

            iterator_A.add_tile_offset({0, 1});
            iterator_B.add_tile_offset({1, 0});
            iterator_E.add_tile_offset({0, 1});

            this->smem_iterator_A_.add_tile_offset({0, 1});
            this->smem_iterator_B_.add_tile_offset({1, 0});
            this->smem_iterator_E_.add_tile_offset({0, 1});

            cutlass::arch::cp_async_fence();
        }

        accum = src_accum;

        cutlass::arch::cp_async_wait<Base::kStages - 2>();
        __syncthreads();

        WarpLoadedFragmentA warp_loaded_frag_A[2];
        WarpLoadedFragmentB warp_loaded_frag_B[Detail::kBBufferSize];
        WarpTransformedFragmentA warp_transformed_frag_A[2];
        WarpTransformedFragmentB warp_transformed_frag_B[Detail::kBBufferSize];
        WarpFragmentE warp_frag_E[2];

        Operator warp_mma;

        this->warp_tile_iterator_A_.set_kgroup_index(0);
        this->warp_tile_iterator_B_.set_kgroup_index(0);
        this->warp_tile_iterator_E_.set_kgroup_index(0);

        this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
        this->warp_tile_iterator_E_.load(warp_frag_E[0]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;
        ++this->warp_tile_iterator_E_;

        if (gemm_k_iterations == 0) {
            iterator_A.clear_mask();
            iterator_B.clear_mask();
            iterator_E.clear_mask();
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
                this->warp_tile_iterator_E_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);

                this->warp_tile_iterator_A_.load(
                        warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_E_.load(
                        warp_frag_E[(warp_mma_k + 1) % 2]);

                ++this->warp_tile_iterator_A_;
                ++this->warp_tile_iterator_E_;

                if (Detail::kBBufferSize == 2) {
                    this->warp_tile_iterator_B_.set_kgroup_index(
                            (warp_mma_k + 1) % Base::kWarpGemmIterations);
                    this->warp_tile_iterator_B_.load(
                            warp_loaded_frag_B[(warp_mma_k + 1) %
                                               Detail::kBBufferSize]);
                    ++this->warp_tile_iterator_B_;
                }

                if (warp_mma_k > 0)
                    warp_mma.transform(
                            warp_transformed_frag_A[warp_mma_k % 2],
                            warp_transformed_frag_B[warp_mma_k %
                                                    Detail::kBBufferSize],
                            warp_loaded_frag_A[warp_mma_k % 2],
                            warp_loaded_frag_B[warp_mma_k %
                                               Detail::kBBufferSize]);

                warp_mma(accum, warp_transformed_frag_A[warp_mma_k % 2],
                         warp_transformed_frag_B[warp_mma_k %
                                                 Detail::kBBufferSize],
                         accum, warp_frag_E[warp_mma_k % 2]);

                if (Detail::kBBufferSize == 1) {
                    this->warp_tile_iterator_B_.set_kgroup_index(
                            (warp_mma_k + 1) % Base::kWarpGemmIterations);
                    this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
                    ++this->warp_tile_iterator_B_;
                }

                if (warp_mma_k < Base::kWarpGemmIterations - 1) {
                    int group_start_iteration_A, group_start_iteration_B,
                            group_start_iteration_E;

                    group_start_iteration_A =
                            warp_mma_k * Detail::kAccessesPerGroupA;
                    group_start_iteration_B =
                            warp_mma_k * Detail::kAccessesPerGroupB;
                    group_start_iteration_E =
                            warp_mma_k * Detail::kAccessesPerGroupE;

                    copy_tiles_and_advance(iterator_A, iterator_B, iterator_E,
                                           group_start_iteration_A,
                                           group_start_iteration_B,
                                           group_start_iteration_E);
                }

                if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
                    int group_start_iteration_A, group_start_iteration_B,
                            group_start_iteration_E;
                    group_start_iteration_A =
                            (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
                    group_start_iteration_B =
                            (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
                    group_start_iteration_E =
                            (warp_mma_k + 1) * Detail::kAccessesPerGroupE;

                    copy_tiles_and_advance(iterator_A, iterator_B, iterator_E,
                                           group_start_iteration_A,
                                           group_start_iteration_B,
                                           group_start_iteration_E);

                    cutlass::arch::cp_async_fence();

                    arch::cp_async_wait<Base::kStages - 2>();
                    __syncthreads();

                    iterator_A.add_tile_offset({0, 1});
                    iterator_B.add_tile_offset({1, 0});
                    iterator_E.add_tile_offset({0, 1});

                    this->smem_iterator_A_.add_tile_offset({0, 1});
                    this->smem_iterator_B_.add_tile_offset({1, 0});
                    this->smem_iterator_E_.add_tile_offset({0, 1});

                    if (smem_write_stage_idx == (Base::kStages - 1)) {
                        this->smem_iterator_A_.add_tile_offset(
                                {0, -Base::kStages});
                        this->smem_iterator_B_.add_tile_offset(
                                {-Base::kStages, 0});
                        this->smem_iterator_E_.add_tile_offset(
                                {0, -Base::kStages});
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
                        this->warp_tile_iterator_E_.add_tile_offset(
                                {0, -Base::kStages * Policy::kPartitionsK *
                                            Base::kWarpGemmIterations});
                        smem_read_stage_idx = 0;
                    } else {
                        ++smem_read_stage_idx;
                    }

                    --gemm_k_iterations;
                    if (gemm_k_iterations == 0) {
                        iterator_A.clear_mask();
                        iterator_B.clear_mask();
                        iterator_E.clear_mask();
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
    }
};


}
}
}

