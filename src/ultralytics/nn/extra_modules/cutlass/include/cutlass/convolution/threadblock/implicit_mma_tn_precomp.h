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
 * \file include/cutlass/convolution/threadblock/implicit_mma_tn_precomp.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/convolution/threadblock/implicit_mma_tn_base.h"


namespace cutlass {
namespace conv {
namespace threadblock {


template <
        typename Shape_,
        typename IteratorSrc_,
        typename SmemIteratorSrc_,
        typename IteratorFilter_,
        typename SmemIteratorFilter_,
        typename ElementDst_,
        typename LayoutDst_,
        typename Policy_,
        typename TransformSrc_ =
                NumericArrayConverter<typename SmemIteratorSrc_::Element,
                                      typename IteratorSrc_::Element,
                                      IteratorSrc_::Fragment::kElements>,
        typename TransformFilter_ =
                NumericArrayConverter<typename SmemIteratorFilter_::Element,
                                      typename IteratorFilter_::Element,
                                      IteratorFilter_::Fragment::kElements>,
        typename Enable = bool>
class MmaTnPrecompPipelined : public MmaTnBase<Shape_, Policy_, 2> {
public:
    using Base = MmaTnBase<Shape_, Policy_, 2>;

    using Shape =
            Shape_;
    using IteratorSrc = IteratorSrc_;
    using IteratorFilter =
            IteratorFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using Policy = Policy_;

    using SmemIteratorSrc = SmemIteratorSrc_;
    using SmemIteratorFilter = SmemIteratorFilter_;

    using TransformSrc = TransformSrc_;
    using TransformFilter = TransformFilter_;


    using FragmentSrc = typename IteratorSrc::Fragment;

    using FragmentFilter = typename IteratorFilter::Fragment;

    using FragmentDst = typename Policy::Operator::FragmentC;

    using Operator = typename Policy::Operator;

    using ArchTag = typename Policy::Operator::ArchTag;

    static ComplexTransform const kTransformSrc = Operator::kTransformA;

    static ComplexTransform const kTransformFilter = Operator::kTransformB;

    static_assert((Base::kStages == 2),
                  "MmaTnPipelined requires kStages set to value 2");

private:
    using WarpFragmentSrc = typename Operator::FragmentA;
    using WarpFragmentFilter = typename Operator::FragmentB;

protected:
    SmemIteratorSrc smem_iterator_src_;

    SmemIteratorFilter smem_iterator_filter_;

public:
    CUTLASS_DEVICE
    MmaTnPrecompPipelined(
            typename Base::SharedStorage&
                    shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx
            )
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              smem_iterator_src_(shared_storage.operand_src_ref(), thread_idx),
              smem_iterator_filter_(shared_storage.operand_filter_ref(),
                                    thread_idx) {

        int warp_idx_mn =
                warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        this->warp_tile_iterator_src_.add_tile_offset(
                {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
        this->warp_tile_iterator_filter_.add_tile_offset(
                {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    }

    CUTLASS_DEVICE
    void operator()(
            int conv_k_iterations,
            FragmentDst& accum,
            IteratorSrc iterator_src,
            IteratorFilter iterator_filter,
            FragmentDst const& src_accum,
            TransformSrc transform_src =
                    TransformSrc(),
            TransformFilter transform_filter =
                    TransformFilter()) {

        accum = src_accum;

        FragmentSrc tb_frag_src;
        FragmentFilter tb_frag_filter;

        tb_frag_src.clear();
        tb_frag_filter.clear();

        iterator_src.load(tb_frag_src);
        iterator_filter.load(tb_frag_filter);

        ++iterator_src;
        ++iterator_filter;

        this->smem_iterator_src_.store(transform_src(tb_frag_src));
        this->smem_iterator_filter_.store(transform_filter(tb_frag_filter));

        ++this->smem_iterator_src_;
        ++this->smem_iterator_filter_;

        __syncthreads();

        WarpFragmentSrc warp_frag_src[2];
        WarpFragmentFilter warp_frag_filter[2];

        this->warp_tile_iterator_src_.set_kgroup_index(0);
        this->warp_tile_iterator_filter_.set_kgroup_index(0);

        this->warp_tile_iterator_src_.load(warp_frag_src[0]);
        this->warp_tile_iterator_filter_.load(warp_frag_filter[0]);

        ++this->warp_tile_iterator_src_;
        ++this->warp_tile_iterator_filter_;

        Operator warp_mma;

        int smem_write_stage_idx = 1;

        if (conv_k_iterations <= 1) {
            iterator_src.clear_mask();
            iterator_filter.clear_mask();
        }

        CUTLASS_GEMM_LOOP
        for (; conv_k_iterations > 0; --conv_k_iterations) {
            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
                 ++warp_mma_k) {
                if (warp_mma_k == Base::kWarpGemmIterations - 1) {
                    this->smem_iterator_src_.store(transform_src(tb_frag_src));

                    this->smem_iterator_filter_.store(
                            transform_filter(tb_frag_filter));

                    __syncthreads();

                    ++this->smem_iterator_src_;
                    ++this->smem_iterator_filter_;

                    if (smem_write_stage_idx == 1) {
                        this->smem_iterator_src_.add_tile_offset(
                                {0, -Base::kStages});
                        this->smem_iterator_filter_.add_tile_offset(
                                {-Base::kStages, 0});
                    } else {
                        this->warp_tile_iterator_src_.add_tile_offset(
                                {0, -Base::kStages * Policy::kPartitionsK *
                                            Base::kWarpGemmIterations});
                        this->warp_tile_iterator_filter_.add_tile_offset(
                                {-Base::kStages * Policy::kPartitionsK *
                                         Base::kWarpGemmIterations,
                                 0});
                    }

                    smem_write_stage_idx ^= 1;
                }

                this->warp_tile_iterator_src_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);
                this->warp_tile_iterator_filter_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);

                this->warp_tile_iterator_src_.load(
                        warp_frag_src[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_filter_.load(
                        warp_frag_filter[(warp_mma_k + 1) % 2]);

                ++this->warp_tile_iterator_src_;
                ++this->warp_tile_iterator_filter_;

                if (warp_mma_k == 0) {
                    iterator_src.load(tb_frag_src);
                    iterator_filter.load(tb_frag_filter);

                    ++iterator_src;
                    ++iterator_filter;

                    if (conv_k_iterations <= 2) {
                        iterator_src.clear_mask();
                        iterator_filter.clear_mask();
                    }
                }

                warp_mma(accum, warp_frag_src[warp_mma_k % 2],
                         warp_frag_filter[warp_mma_k % 2], accum);
            }
        }
    }
};

template <
        typename Shape_,
        typename IteratorSrc_,
        typename SmemIteratorSrc_,
        typename IteratorFilter_,
        typename SmemIteratorFilter_,
        typename ElementDst_,
        typename LayoutDst_,
        typename Policy_,
        typename TransformSrc_ =
                NumericArrayConverter<typename SmemIteratorSrc_::Element,
                                      typename IteratorSrc_::Element,
                                      IteratorSrc_::Fragment::kElements>,
        typename TransformFilter_ =
                NumericArrayConverter<typename SmemIteratorFilter_::Element,
                                      typename IteratorFilter_::Element,
                                      IteratorFilter_::Fragment::kElements>,
        typename Enable = bool>
class MmaTnPrecompSingleStage : public MmaTnBase<Shape_, Policy_, 1> {
public:
    using Base = MmaTnBase<Shape_, Policy_, 1>;

    using Shape =
            Shape_;
    using IteratorSrc = IteratorSrc_;
    using IteratorFilter =
            IteratorFilter_;
    using ElementDst = ElementDst_;
    using LayoutDst = LayoutDst_;
    using Policy = Policy_;

    using SmemIteratorSrc = SmemIteratorSrc_;
    using SmemIteratorFilter = SmemIteratorFilter_;

    using TransformSrc = TransformSrc_;
    using TransformFilter = TransformFilter_;


    using FragmentSrc = typename IteratorSrc::Fragment;

    using FragmentFilter = typename IteratorFilter::Fragment;

    using FragmentDst = typename Policy::Operator::FragmentC;

    using Operator = typename Policy::Operator;

    using ArchTag = typename Policy::Operator::ArchTag;

    static ComplexTransform const kTransformSrc = Operator::kTransformA;

    static ComplexTransform const kTransformFilter = Operator::kTransformB;

    static_assert((Base::kStages == 1),
                  "MmaTnSingleStage requires kStages set to value 1");

private:
    using WarpFragmentSrc = typename Operator::FragmentA;
    using WarpFragmentFilter = typename Operator::FragmentB;

protected:
    SmemIteratorSrc smem_iterator_src_;

    SmemIteratorFilter smem_iterator_filter_;

public:
    CUTLASS_DEVICE
    MmaTnPrecompSingleStage(
            typename Base::SharedStorage&
                    shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx
            )
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              smem_iterator_src_(shared_storage.operand_src_ref(), thread_idx),
              smem_iterator_filter_(shared_storage.operand_filter_ref(),
                                    thread_idx) {

        int warp_idx_mn =
                warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
        int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

        int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
        int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

        this->warp_tile_iterator_src_.add_tile_offset(
                {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
        this->warp_tile_iterator_filter_.add_tile_offset(
                {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    }

    CUTLASS_DEVICE
    void operator()(
            int conv_k_iterations,
            FragmentDst& accum,
            IteratorSrc iterator_src,
            IteratorFilter iterator_filter,
            FragmentDst const& src_accum,
            TransformSrc transform_src =
                    TransformSrc(),
            TransformFilter transform_filter =
                    TransformFilter()) {


        accum = src_accum;

        FragmentSrc tb_frag_src;
        FragmentFilter tb_frag_filter;

        tb_frag_src.clear();
        tb_frag_filter.clear();

        iterator_src.load(tb_frag_src);
        iterator_filter.load(tb_frag_filter);

        ++iterator_src;
        ++iterator_filter;

        this->smem_iterator_src_.store(transform_src(tb_frag_src));
        this->smem_iterator_filter_.store(transform_filter(tb_frag_filter));

        __syncthreads();

        WarpFragmentSrc warp_frag_src[2];
        WarpFragmentFilter warp_frag_filter[2];

        this->warp_tile_iterator_src_.set_kgroup_index(0);
        this->warp_tile_iterator_filter_.set_kgroup_index(0);

        this->warp_tile_iterator_src_.load(warp_frag_src[0]);
        this->warp_tile_iterator_filter_.load(warp_frag_filter[0]);

        ++this->warp_tile_iterator_src_;
        ++this->warp_tile_iterator_filter_;

        Operator warp_mma;

        if (conv_k_iterations <= 1) {
            iterator_src.clear_mask();
            iterator_filter.clear_mask();
        }

        CUTLASS_GEMM_LOOP
        for (; conv_k_iterations > 0; --conv_k_iterations) {
            CUTLASS_PRAGMA_UNROLL
            for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
                 ++warp_mma_k) {
                if (warp_mma_k == Base::kWarpGemmIterations - 1) {
                    __syncthreads();
                    this->smem_iterator_src_.store(transform_src(tb_frag_src));

                    this->smem_iterator_filter_.store(
                            transform_filter(tb_frag_filter));

                    __syncthreads();

                    this->warp_tile_iterator_src_.add_tile_offset(
                            {0, -Policy::kPartitionsK *
                                        Base::kWarpGemmIterations});
                    this->warp_tile_iterator_filter_.add_tile_offset(
                            {-Policy::kPartitionsK * Base::kWarpGemmIterations,
                             0});
                }

                this->warp_tile_iterator_src_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);
                this->warp_tile_iterator_filter_.set_kgroup_index(
                        (warp_mma_k + 1) % Base::kWarpGemmIterations);

                this->warp_tile_iterator_src_.load(
                        warp_frag_src[(warp_mma_k + 1) % 2]);
                this->warp_tile_iterator_filter_.load(
                        warp_frag_filter[(warp_mma_k + 1) % 2]);

                ++this->warp_tile_iterator_src_;
                ++this->warp_tile_iterator_filter_;

                if (warp_mma_k == 0) {
                    iterator_src.load(tb_frag_src);
                    iterator_filter.load(tb_frag_filter);

                    ++iterator_src;
                    ++iterator_filter;

                    if (conv_k_iterations <= 2) {
                        iterator_src.clear_mask();
                        iterator_filter.clear_mask();
                    }
                }

                warp_mma(accum, warp_frag_src[warp_mma_k % 2],
                         warp_frag_filter[warp_mma_k % 2], accum);
            }
        }
    }
};


}
}
}

