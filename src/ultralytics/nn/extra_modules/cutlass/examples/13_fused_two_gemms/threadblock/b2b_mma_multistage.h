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

#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"

#include "threadblock/b2b_mma_base.h"


namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape0_,
        typename IteratorA0_,
        typename SmemIteratorA0_,
        cutlass::arch::CacheOperation::Kind CacheOpA0,
        typename IteratorB0_,
        typename SmemIteratorB0_,
        cutlass::arch::CacheOperation::Kind CacheOpB0,
        typename Shape1_,
        typename FragmentIteratorA1_,
        typename IteratorB1_,
        typename SmemIteratorB1_,
        cutlass::arch::CacheOperation::Kind CacheOpB1,
        typename ElementC_,
        typename LayoutC_,
        typename OutputOp_,
        typename Policy0_,
        typename Policy1_,
        int Stages,
        typename Enable = bool>
class B2bMmaMultistage
        : public B2bMmaBase<Shape0_, Shape1_, Policy0_, Policy1_, Stages> {
public:
    using Base = B2bMmaBase<Shape0_, Shape1_, Policy0_, Policy1_, Stages>;
    using Shape0 = Shape0_;
    using IteratorA0 = IteratorA0_;
    using IteratorB0 = IteratorB0_;
    using Policy0 = Policy0_;

    using SmemIteratorA0 = SmemIteratorA0_;
    using SmemIteratorB0 = SmemIteratorB0_;

    using Shape1 = Shape1_;
    using FragmentIteratorA1 = FragmentIteratorA1_;
    using IteratorB1 = IteratorB1_;
    using Policy1 = Policy1_;

    using SmemIteratorB1 = SmemIteratorB1_;

    using ElementC = ElementC_;
    using LayoutC = LayoutC_;

    using OutputOp = OutputOp_;

    static cutlass::arch::CacheOperation::Kind const kCacheOpA0 = CacheOpA0;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB0 = CacheOpB0;
    static cutlass::arch::CacheOperation::Kind const kCacheOpB1 = CacheOpB1;


    using FragmentC0 = typename Policy0::Operator::FragmentC;

    using Operator0 = typename Policy0::Operator;

    using FragmentC1 = typename Policy1::Operator::FragmentC;

    using Operator1 = typename Policy1::Operator;

    using ArchTag = arch::Sm80;

    static ComplexTransform const kTransformA0 = Operator0::kTransformA;

    static ComplexTransform const kTransformB0 = Operator0::kTransformB;

    static ComplexTransform const kTransformB1 = Operator1::kTransformB;

    struct Detail {
        static_assert(
                Base::kWarpGemmIterations0 > 1,
                "The pipelined structure requires at least two warp-level "
                "GEMM operations.");
        static_assert(
                Base::kWarpGemmIterations1 > 1,
                "The pipelined structure requires at least two warp-level "
                "GEMM operations.");

        static int const TBLDGSTSIterationsA0 =
                IteratorA0::ThreadMap::Iterations::kCount;

        static int const TBLDGSTSIterationsB0 =
                IteratorB0::ThreadMap::Iterations::kCount;

        static int const TBLDGSTSIterationsB1 =
                IteratorB1::ThreadMap::Iterations::kCount;

        static int const kStages = Stages;

        static int const kAccessesPerGroupA0 =
                (TBLDGSTSIterationsA0 + Base::kWarpGemmIterations0 - 1) /
                Base::kWarpGemmIterations0;

        static int const kAccessesPerGroupB0 =
                (TBLDGSTSIterationsB0 + Base::kWarpGemmIterations0 - 1) /
                Base::kWarpGemmIterations0;

        static int const kAccessesPerGroupB1 =
                (TBLDGSTSIterationsB1 + Base::kWarpGemmIterations1 - 1) /
                Base::kWarpGemmIterations1;
    };

private:
    using WarpLoadedFragmentA0 = typename Operator0::FragmentA;
    using WarpLoadedFragmentB0 = typename Operator0::FragmentB;
    using WarpLoadedFragmentA1 = typename FragmentIteratorA1::Fragment;
    using WarpLoadedFragmentB1 = typename Operator1::FragmentB;
    using WarpTransformedFragmentA0 = typename Operator0::TransformedFragmentA;
    using WarpTransformedFragmentB0 = typename Operator0::TransformedFragmentB;
    using WarpTransformedFragmentA1 = typename Operator1::TransformedFragmentA;
    using WarpTransformedFragmentB1 = typename Operator1::TransformedFragmentB;

private:

    SmemIteratorA0 smem_iterator_A0_;

    SmemIteratorB0 smem_iterator_B0_;

    SmemIteratorB1 smem_iterator_B1_;

public:
    CUTLASS_DEVICE
    B2bMmaMultistage(
            typename Base::B2bMmaSharedStorage& shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx)
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              smem_iterator_A0_(shared_storage.sharedStorage0.operand_A_ref(),
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

        this->warp_tile_iterator_A0_.add_tile_offset(
                {warp_idx_m, Base::kWarpGemmIterations0 * warp_idx_k});
        this->warp_tile_iterator_B0_.add_tile_offset(
                {Base::kWarpGemmIterations0 * warp_idx_k, warp_idx_n});
        this->warp_tile_iterator_B1_.add_tile_offset(
                {Base::kWarpGemmIterations1 * warp_idx_k, warp_idx_n});
    }

    CUTLASS_DEVICE
    void copy_tiles_and_advance_0(IteratorA0& iterator_A0,
                                  IteratorB0& iterator_B0,
                                  int group_start_A0 = 0,
                                  int group_start_B0 = 0) {
        iterator_A0.set_iteration_index(group_start_A0 *
                                        IteratorA0::kAccessesPerVector);
        this->smem_iterator_A0_.set_iteration_index(group_start_A0);

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupA0; ++j) {
            if (group_start_A0 + j < Detail::TBLDGSTSIterationsA0) {
                typename IteratorA0::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorA0::AccessType*>(
                                this->smem_iterator_A0_.get());

                int const kSrcBytes =
                        sizeof_bits<typename IteratorA0::Element>::value *
                        IteratorA0::ThreadMap::kElementsPerAccess /
                        IteratorA0::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA0::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_A0.get();

                    cutlass::arch::cp_async<kSrcBytes, kCacheOpA0>(
                            dst_ptr + v, gmem_ptr, iterator_A0.valid());

                    ++iterator_A0;
                }

                ++this->smem_iterator_A0_;
            }
        }

        iterator_B0.set_iteration_index(group_start_B0 *
                                        IteratorB0::kAccessesPerVector);
        this->smem_iterator_B0_.set_iteration_index(group_start_B0);

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupB0; ++j) {
            if (group_start_B0 + j < Detail::TBLDGSTSIterationsB0) {
                typename IteratorB0::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorB0::AccessType*>(
                                this->smem_iterator_B0_.get());

                int const kSrcBytes =
                        sizeof_bits<typename IteratorB0::Element>::value *
                        IteratorB0::ThreadMap::kElementsPerAccess /
                        IteratorB0::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB0::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_B0.get();

                    cutlass::arch::cp_async<kSrcBytes, kCacheOpB0>(
                            dst_ptr + v, gmem_ptr, iterator_B0.valid());

                    ++iterator_B0;
                }
                ++this->smem_iterator_B0_;
            }
        }
    }

    CUTLASS_DEVICE
    void copy_tiles_and_advance_1(IteratorB1& iterator_B1,
                                  int group_start_B1 = 0) {
        iterator_B1.set_iteration_index(group_start_B1 *
                                        IteratorB1::kAccessesPerVector);
        this->smem_iterator_B1_.set_iteration_index(group_start_B1);

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < Detail::kAccessesPerGroupB1; ++j) {
            if (group_start_B1 + j < Detail::TBLDGSTSIterationsB1) {
                typename IteratorB1::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorB1::AccessType*>(
                                this->smem_iterator_B1_.get());

                int const kSrcBytes =
                        sizeof_bits<typename IteratorB1::Element>::value *
                        IteratorB1::ThreadMap::kElementsPerAccess /
                        IteratorB1::kAccessesPerVector / 8;

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
                    auto gmem_ptr = iterator_B1.get();

                    cutlass::arch::cp_async<kSrcBytes, kCacheOpB1>(
                            dst_ptr + v, gmem_ptr, iterator_B1.valid());

                    ++iterator_B1;
                }
                ++this->smem_iterator_B1_;
            }
        }
    }

    CUTLASS_DEVICE
    void operator()(
            int gemm_k_iterations_0,
            FragmentC1& accum,
            IteratorA0 iterator_A0,
            IteratorB0 iterator_B0,
            IteratorB1 iterator_B1,
            FragmentC0 const& src_accum,
            OutputOp output_op_0) {

        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < Base::kStages - 1;
             ++stage, --gemm_k_iterations_0) {
            if (gemm_k_iterations_0 == 0) {
                iterator_A0.clear_mask();
                iterator_B0.clear_mask();
            }

            iterator_A0.set_iteration_index(0);
            this->smem_iterator_A0_.set_iteration_index(0);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::TBLDGSTSIterationsA0; ++j) {
                typename IteratorA0::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorA0::AccessType*>(
                                this->smem_iterator_A0_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorA0::kAccessesPerVector; ++v) {
                    int const kSrcBytes =
                            sizeof_bits<typename IteratorA0::Element>::value *
                            IteratorA0::ThreadMap::kElementsPerAccess /
                            IteratorA0::kAccessesPerVector / 8;

                    int src_bytes = (iterator_A0.valid() ? kSrcBytes : 0);

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA0>(
                            dst_ptr + v, iterator_A0.get(),
                            iterator_A0.valid());

                    ++iterator_A0;
                }

                ++this->smem_iterator_A0_;
            }

            iterator_B0.set_iteration_index(0);
            this->smem_iterator_B0_.set_iteration_index(0);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::TBLDGSTSIterationsB0; ++j) {
                typename IteratorB0::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorB0::AccessType*>(
                                this->smem_iterator_B0_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB0::kAccessesPerVector; ++v) {
                    int const kSrcBytes =
                            sizeof_bits<typename IteratorB0::Element>::value *
                            IteratorB0::ThreadMap::kElementsPerAccess /
                            IteratorB0::kAccessesPerVector / 8;

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB0>(
                            dst_ptr + v, iterator_B0.get(),
                            iterator_B0.valid());

                    ++iterator_B0;
                }

                ++this->smem_iterator_B0_;
            }

            iterator_A0.add_tile_offset({0, 1});
            iterator_B0.add_tile_offset({1, 0});

            this->smem_iterator_A0_.add_tile_offset({0, 1});
            this->smem_iterator_B0_.add_tile_offset({1, 0});

            cutlass::arch::cp_async_fence();
        }

        FragmentC0 accum0 = src_accum;

        cutlass::arch::cp_async_wait<Base::kStages - 2>();
        __syncthreads();

        WarpLoadedFragmentA0 warp_loaded_frag_A0[2];
        WarpLoadedFragmentB0 warp_loaded_frag_B0[2];
        WarpTransformedFragmentA0 warp_transformed_frag_A0[2];
        WarpTransformedFragmentB0 warp_transformed_frag_B0[2];

        Operator0 warp_mma0;

        this->warp_tile_iterator_A0_.set_kgroup_index(0);
        this->warp_tile_iterator_B0_.set_kgroup_index(0);

        this->warp_tile_iterator_A0_.load(warp_loaded_frag_A0[0]);
        this->warp_tile_iterator_B0_.load(warp_loaded_frag_B0[0]);

        ++this->warp_tile_iterator_A0_;
        ++this->warp_tile_iterator_B0_;

        if (gemm_k_iterations_0 == 0) {
            iterator_A0.clear_mask();
            iterator_B0.clear_mask();
        }

        int smem_write_stage_idx = Base::kStages - 1;
        int smem_read_stage_idx = 0;

        warp_mma0.transform(warp_transformed_frag_A0[0],
                            warp_transformed_frag_B0[0], warp_loaded_frag_A0[0],
                            warp_loaded_frag_B0[0]);


        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations_0 > (-Base::kStages + 1);) {

            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations0;
                 ++warp_mma_k) {

                this->warp_tile_iterator_A0_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations0);
                this->warp_tile_iterator_B0_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations0);

                this->warp_tile_iterator_A0_.load(
                        warp_loaded_frag_A0[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_B0_.load(
                        warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);

                ++this->warp_tile_iterator_A0_;
                ++this->warp_tile_iterator_B0_;

                if (warp_mma_k > 0)
                    warp_mma0.transform(
                            warp_transformed_frag_A0[warp_mma_k % 2],
                            warp_transformed_frag_B0[warp_mma_k % 2],
                            warp_loaded_frag_A0[warp_mma_k % 2],
                            warp_loaded_frag_B0[warp_mma_k % 2]);

                warp_mma0(accum0, warp_transformed_frag_A0[warp_mma_k % 2],
                          warp_transformed_frag_B0[warp_mma_k % 2], accum0);

                if (warp_mma_k < Base::kWarpGemmIterations0 - 1) {
                    int group_start_iteration_A0, group_start_iteration_B0;

                    group_start_iteration_A0 =
                            warp_mma_k * Detail::kAccessesPerGroupA0;
                    group_start_iteration_B0 =
                            warp_mma_k * Detail::kAccessesPerGroupB0;

                    copy_tiles_and_advance_0(iterator_A0, iterator_B0,
                                             group_start_iteration_A0,
                                             group_start_iteration_B0);
                }

                if (warp_mma_k + 2 == Base::kWarpGemmIterations0) {
                    int group_start_iteration_A0, group_start_iteration_B0;
                    group_start_iteration_A0 =
                            (warp_mma_k + 1) * Detail::kAccessesPerGroupA0;
                    group_start_iteration_B0 =
                            (warp_mma_k + 1) * Detail::kAccessesPerGroupB0;

                    copy_tiles_and_advance_0(iterator_A0, iterator_B0,
                                             group_start_iteration_A0,
                                             group_start_iteration_B0);

                    cutlass::arch::cp_async_fence();

                    arch::cp_async_wait<Base::kStages - 2>();
                    __syncthreads();

                    iterator_A0.add_tile_offset({0, 1});
                    iterator_B0.add_tile_offset({1, 0});

                    this->smem_iterator_A0_.add_tile_offset({0, 1});
                    this->smem_iterator_B0_.add_tile_offset({1, 0});

                    if (smem_write_stage_idx == (Base::kStages - 1)) {
                        this->smem_iterator_A0_.add_tile_offset(
                                {0, -Base::kStages});
                        this->smem_iterator_B0_.add_tile_offset(
                                {-Base::kStages, 0});
                        smem_write_stage_idx = 0;
                    } else {
                        ++smem_write_stage_idx;
                    }

                    if (smem_read_stage_idx == (Base::kStages - 1)) {
                        this->warp_tile_iterator_A0_.add_tile_offset(
                                {0, -Base::kStages * Policy0::kPartitionsK *
                                            Base::kWarpGemmIterations0});
                        this->warp_tile_iterator_B0_.add_tile_offset(
                                {-Base::kStages * Policy0::kPartitionsK *
                                         Base::kWarpGemmIterations0,
                                 0});
                        smem_read_stage_idx = 0;
                    } else {
                        ++smem_read_stage_idx;
                    }

                    --gemm_k_iterations_0;
                    if (gemm_k_iterations_0 == 0) {
                        iterator_A0.clear_mask();
                        iterator_B0.clear_mask();
                    }
                }

                if (warp_mma_k + 1 == Base::kWarpGemmIterations0)
                    warp_mma0.transform(
                            warp_transformed_frag_A0[(warp_mma_k + 1) % 2],
                            warp_transformed_frag_B0[(warp_mma_k + 1) % 2],
                            warp_loaded_frag_A0[(warp_mma_k + 1) % 2],
                            warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);
            }
        }


        FragmentIteratorA1 warp_tile_iterator_A1_(accum0);

        int gemm_k_iterations_1 = FragmentIteratorA1::Policy::kIterations /
                                  Base::kWarpGemmIterations1;

        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < Base::kStages - 1;
             ++stage, --gemm_k_iterations_1) {
            if (gemm_k_iterations_1 == 0) {
                iterator_B1.clear_mask();
            }

#if 0
      iterator_A1.set_iteration_index(0);
      this->smem_iterator_A1_.set_iteration_index(0);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLDGSTSIterationsA1; ++j) {
        typename IteratorA1::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA1::AccessType *>(
                this->smem_iterator_A1_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA1::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA1::Element>::value *
              IteratorA1::ThreadMap::kElementsPerAccess /
              IteratorA1::kAccessesPerVector / 8;

          int src_bytes = (iterator_A0.valid() ? kSrcBytes : 0);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA0>(
              dst_ptr + v, iterator_A0.get(), iterator_A0.valid());

          ++iterator_A0;
        }

        ++this->smem_iterator_A0_;
      }
#endif

            iterator_B1.set_iteration_index(0);
            this->smem_iterator_B1_.set_iteration_index(0);

            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < Detail::TBLDGSTSIterationsB1; ++j) {
                typename IteratorB1::AccessType* dst_ptr =
                        reinterpret_cast<typename IteratorB1::AccessType*>(
                                this->smem_iterator_B1_.get());

                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
                    int const kSrcBytes =
                            sizeof_bits<typename IteratorB1::Element>::value *
                            IteratorB1::ThreadMap::kElementsPerAccess /
                            IteratorB1::kAccessesPerVector / 8;

                    cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB1>(
                            dst_ptr + v, iterator_B1.get(),
                            iterator_B1.valid());

                    ++iterator_B1;
                }

                ++this->smem_iterator_B1_;
            }

            iterator_B1.add_tile_offset({1, 0});

            this->smem_iterator_B1_.add_tile_offset({1, 0});

            cutlass::arch::cp_async_fence();
        }


        cutlass::arch::cp_async_wait<Base::kStages - 2>();
        __syncthreads();

        WarpLoadedFragmentA1 warp_loaded_frag_A1[2];
        WarpLoadedFragmentB1 warp_loaded_frag_B1[2];
        WarpTransformedFragmentA1 warp_transformed_frag_A1[2];
        WarpTransformedFragmentB1 warp_transformed_frag_B1[2];

        Operator1 warp_mma1;

        this->warp_tile_iterator_B1_.set_kgroup_index(0);

        warp_tile_iterator_A1_.load(warp_loaded_frag_A1[0], output_op_0);
        this->warp_tile_iterator_B1_.load(warp_loaded_frag_B1[0]);

        ++warp_tile_iterator_A1_;
        ++this->warp_tile_iterator_B1_;

        if (gemm_k_iterations_1 == 0) {
            iterator_B1.clear_mask();
        }

        smem_write_stage_idx = Base::kStages - 1;
        smem_read_stage_idx = 0;

        warp_mma1.transform(warp_transformed_frag_A1[0],
                            warp_transformed_frag_B1[0], warp_loaded_frag_A1[0],
                            warp_loaded_frag_B1[0]);


        CUTLASS_PRAGMA_UNROLL
        for (gemm_k_iterations_1 = FragmentIteratorA1::Policy::kIterations /
                                           Base::kWarpGemmIterations1 -
                                   (Base::kStages - 1);
             gemm_k_iterations_1 > (-Base::kStages + 1);
             gemm_k_iterations_1--) {

            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations1;
                 ++warp_mma_k) {

                this->warp_tile_iterator_B1_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations1);

                warp_tile_iterator_A1_.load(
                        warp_loaded_frag_A1[(warp_mma_k + 1) % 2], output_op_0);
                this->warp_tile_iterator_B1_.load(
                        warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);

                ++warp_tile_iterator_A1_;
                ++this->warp_tile_iterator_B1_;

                if (warp_mma_k > 0)
                    warp_mma1.transform(
                            warp_transformed_frag_A1[warp_mma_k % 2],
                            warp_transformed_frag_B1[warp_mma_k % 2],
                            warp_loaded_frag_A1[warp_mma_k % 2],
                            warp_loaded_frag_B1[warp_mma_k % 2]);

                warp_mma1(accum, warp_transformed_frag_A1[warp_mma_k % 2],
                          warp_transformed_frag_B1[warp_mma_k % 2], accum);

                if (warp_mma_k < Base::kWarpGemmIterations1 - 1) {
                    int group_start_iteration_B1;

                    group_start_iteration_B1 =
                            warp_mma_k * Detail::kAccessesPerGroupB1;

                    copy_tiles_and_advance_1(iterator_B1,
                                             group_start_iteration_B1);
                }

                if (warp_mma_k + 2 == Base::kWarpGemmIterations1) {
                    int group_start_iteration_B1;
                    group_start_iteration_B1 =
                            (warp_mma_k + 1) * Detail::kAccessesPerGroupB1;

                    copy_tiles_and_advance_1(iterator_B1,
                                             group_start_iteration_B1);

                    cutlass::arch::cp_async_fence();

                    arch::cp_async_wait<Base::kStages - 2>();
                    __syncthreads();

                    iterator_B1.add_tile_offset({1, 0});

                    this->smem_iterator_B1_.add_tile_offset({1, 0});

                    if (smem_write_stage_idx == (Base::kStages - 1)) {
                        this->smem_iterator_B1_.add_tile_offset(
                                {-Base::kStages, 0});
                        smem_write_stage_idx = 0;
                    } else {
                        ++smem_write_stage_idx;
                    }

                    if (smem_read_stage_idx == (Base::kStages - 1)) {
                        this->warp_tile_iterator_B1_.add_tile_offset(
                                {-Base::kStages * Policy0::kPartitionsK *
                                         Base::kWarpGemmIterations1,
                                 0});
                        smem_read_stage_idx = 0;
                    } else {
                        ++smem_read_stage_idx;
                    }

                    if (gemm_k_iterations_1 == 1) {
                        iterator_B1.clear_mask();
                    }
                }

                if (warp_mma_k + 1 == Base::kWarpGemmIterations1)
                    warp_mma1.transform(
                            warp_transformed_frag_A1[(warp_mma_k + 1) % 2],
                            warp_transformed_frag_B1[(warp_mma_k + 1) % 2],
                            warp_loaded_frag_A1[(warp_mma_k + 1) % 2],
                            warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
            }
        }
    }
};


}
}
}

