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
 * \file
 * include/cutlass/convolution/threadblock/conv2d_tile_iterator_tn_fprop_nhwc_precomp.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/convolution/threadblock/conv2d_tile_params.h"


namespace cutlass {
namespace conv {
namespace threadblock {

namespace detail {
struct PrecompRSC {
    uint8_t fw, fh;
    uint16_t c;
};

template <typename Shape_, int AccessSize>
CUTLASS_HOST_DEVICE void compute_offset_fprop_nhwc(int* constant_offset_,
                                                   int fh_, int fw_) {
    using Shape = Shape_;
    using Index = int;
    static int const kAccessSize = AccessSize;

    PrecompRSC* precomp_ptr = reinterpret_cast<PrecompRSC*>(constant_offset_);
    Index s = 0;
    Index filter_pixels = fh_ * fw_;
    Index inc_step = Shape::kColumn / kAccessSize;

    CUTLASS_PRAGMA_UNROLL
    for (; s < inc_step; ++s) {
        Index c = s / (filter_pixels);
        Index fhfw = s - (filter_pixels)*c;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        PrecompRSC cur;
        cur.fw = (uint8_t)fw;
        cur.fh = (uint8_t)fh;
        cur.c = (uint16_t)(c * kAccessSize);
        *precomp_ptr = cur;
        precomp_ptr++;
    }

    CUTLASS_PRAGMA_UNROLL
    for (; s < (1 + filter_pixels) * inc_step; ++s) {
        Index c = s / (filter_pixels);
        Index fhfw = s - (filter_pixels)*c;
        Index fh = fhfw / fw_;
        Index fw = fhfw - fw_ * fh;
        PrecompRSC cur;
        cur.fw = (uint8_t)fw;
        cur.fh = (uint8_t)fh;
        cur.c = (uint16_t)(c * kAccessSize);
        *precomp_ptr = cur;
        precomp_ptr++;
    }
}
}


template <typename Shape, typename Element, typename Layout, typename ThreadMap,
          int AccessSize,
          SpecialOptimizeDesc SpecialOpt = SpecialOptimizeDesc::NONE>
class FpropPrecompNHWCParams;

template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize>
class FpropPrecompNHWCParams<Shape_, Element_, layout::TensorNHWC, ThreadMap_,
                             AccessSize, SpecialOptimizeDesc::NONE> {
public:
    static int const kAccessSize = AccessSize;
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNHWC;
    using ThreadMap = ThreadMap_;

    using ExtraParam = typename platform::conditional<
            platform::is_same<Element, uint4b_t>::value, ExtraParamZeroPoint,
            platform::none_type>::type;

    using ShortIndex = int8_t;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static int const kElementSize =
            sizeof(detail::PrecompRSC) * 8 / cutlass::sizeof_bits<Index>::value;

    static int const kPrecomputedOffsetBufferSize = 848;
    static int const kMaxFilterPixels =
            kPrecomputedOffsetBufferSize /
                    (kElementSize * Shape::kColumn / kAccessSize) -
            1;

    static_assert(!(Shape::kColumn % kAccessSize),
                  "Shape::kColumn must be divisible by the AccessSize.");

    Layout layout_;

    Index constant_offset_max_;
    Index constant_offset_rewind_;
    Index constant_offset_[kPrecomputedOffsetBufferSize];
    uint32_t pack_pad_;

    FastDivmod pq_divmod;
    FastDivmod q_divmod;

    int N, H, W, C, R, S;
    int pad_h, pad_w;
    int stride_h, stride_w;

    CUTLASS_HOST_DEVICE
    FpropPrecompNHWCParams() : layout_(Layout()) {}

    CUTLASS_HOST_DEVICE
    FpropPrecompNHWCParams(Layout const& layout,
                           Conv2dProblemSize const& problem_size,
                           ExtraParam const& extra_param = {})
            : layout_(layout),
              pq_divmod(problem_size.P * problem_size.Q),
              q_divmod(problem_size.Q) {
        detail::compute_offset_fprop_nhwc<Shape, kAccessSize>(
                constant_offset_, problem_size.R, problem_size.S);
        constant_offset_max_ = (problem_size.R * problem_size.S) *
                               Shape::kColumn / kAccessSize;
        constant_offset_rewind_ = (1 - problem_size.R * problem_size.S) *
                                  Shape::kColumn / kAccessSize;

        N = problem_size.N;
        H = problem_size.H;
        W = problem_size.W;
        C = problem_size.C;
        R = problem_size.R;
        S = problem_size.S;
        pad_h = problem_size.pad_h;
        pad_w = problem_size.pad_w;
        stride_h = problem_size.stride_h;
        stride_w = problem_size.stride_w;

        pack_pad_ = detail::prepare_pack_pad<Element, ExtraParam>(extra_param);
    }
};

template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize>
class FpropPrecompNHWCParams<Shape_, Element_, layout::TensorNHWC, ThreadMap_,
                             AccessSize,
                             SpecialOptimizeDesc::CONV_FILTER_UNITY> {
public:
    static int const kAccessSize = AccessSize;
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNHWC;
    using ThreadMap = ThreadMap_;

    using ExtraParam = typename platform::conditional<
            platform::is_same<Element, uint4b_t>::value, ExtraParamZeroPoint,
            platform::none_type>::type;

    using ShortIndex = int8_t;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;
    using LogicalCoord = typename LogicalLayout::TensorCoord;

    Layout layout_;

    uint32_t pack_pad_;

    FastDivmod pq_divmod;
    FastDivmod q_divmod;

    int N, H, W, C;
    int pad_h, pad_w;
    int stride_h, stride_w;

    CUTLASS_HOST_DEVICE
    FpropPrecompNHWCParams() : layout_(Layout()) {}

    CUTLASS_HOST_DEVICE
    FpropPrecompNHWCParams(Layout const& layout,
                           Conv2dProblemSize const& problem_size,
                           ExtraParam const& extra_param = {})
            : layout_(layout),
              pq_divmod(problem_size.P * problem_size.Q),
              q_divmod(problem_size.Q) {
        N = problem_size.N;
        H = problem_size.H;
        W = problem_size.W;
        C = problem_size.C;
        pad_h = problem_size.pad_h;
        pad_w = problem_size.pad_w;
        stride_h = problem_size.stride_h;
        stride_w = problem_size.stride_w;
        pack_pad_ = detail::prepare_pack_pad<Element, ExtraParam>(extra_param);
    }
};


template <typename Shape_, typename Element_, typename Layout_,
          typename ThreadMap_, int AccessSize,
          SpecialOptimizeDesc SpecialOpt = SpecialOptimizeDesc::NONE>
class Conv2dTileSrcIteratorFpropPrecompNHWC;

template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize>
class Conv2dTileSrcIteratorFpropPrecompNHWC<
        Shape_, Element_, layout::TensorNHWC, ThreadMap_, AccessSize,
        SpecialOptimizeDesc::NONE> {
public:

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNHWC;
    using TensorCoord = typename Layout::TensorCoord;
    using ThreadMap = ThreadMap_;
    static int const kAccessSize = AccessSize;
    using AccessType = AlignedArray<Element, kAccessSize>;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using ConvProblemSize = typename conv::Conv2dProblemSize;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    static_assert(ThreadMap::Iterations::kContiguous == 1,
                  "Require Iterations::kContiguous == 1");

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;


    using Params =
            FpropPrecompNHWCParams<Shape, Element, Layout, ThreadMap,
                                   kAccessSize, SpecialOptimizeDesc::NONE>;

private:
    Params const& params_;

    Index constant_offset_;
    Index c_offset_;

    char const* pointer_[ThreadMap::Iterations::kStrided];

    Index masks_[ThreadMap::Iterations::kStrided][2];

public:
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecompNHWC(Params const& params,
                                          Element const* ptr,
                                          LogicalCoord extent, int thread_idx,
                                          MatrixCoord const& threadblock_offset)
            : params_(params) {
        layout::PitchLinearCoord thread_coord =
                ThreadMap::initial_offset(thread_idx);

        constant_offset_ = thread_coord.contiguous() / kAccessSize;

        c_offset_ = 0;

        int offset_n[ThreadMap::Iterations::kStrided];
        int offset_p[ThreadMap::Iterations::kStrided];
        int offset_q[ThreadMap::Iterations::kStrided];

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] = reinterpret_cast<char const*>(ptr);

            int offset_npq = threadblock_offset.row() + thread_coord.strided() +
                             s * ThreadMap::Delta::kStrided;

            int residual;

            params_.pq_divmod(offset_n[s], residual, offset_npq);
            params_.q_divmod(offset_p[s], offset_q[s], residual);

            TensorCoord coord = TensorCoord(
                    offset_n[s], offset_p[s] * params_.stride_h - params_.pad_h,
                    offset_q[s] * params_.stride_w - params_.pad_w, 0);

            pointer_[s] +=
                    params_.layout_(coord) * sizeof_bits<Element>::value / 8;
        }

        clear_mask();

        CUTLASS_PRAGMA_NO_UNROLL
        for (int r = 0; r < params_.R; ++r) {
            CUTLASS_PRAGMA_UNROLL
            for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided;
                 ++s_idx) {

                int h = offset_p[s_idx] * params_.stride_h - params_.pad_h + r;

                bool pred = (offset_n[s_idx] < params_.N && h >= 0 &&
                             h < params_.H);
                masks_[s_idx][0] |= (pred << r);
            }
        }

        CUTLASS_PRAGMA_NO_UNROLL
        for (int s = 0; s < params_.S; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int s_idx = 0; s_idx < ThreadMap::Iterations::kStrided;
                 ++s_idx) {

                int w = offset_q[s_idx] * params_.stride_w - params_.pad_w + s;

                bool pred = (w >= 0 && w < params_.W);
                masks_[s_idx][1] |= (pred << s);
            }
        }
    }

private:
    CUTLASS_HOST_DEVICE
    void add_byte_offset_(LongIndex byte_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] += byte_offset;
        }
    }

public:
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        add_byte_offset_(pointer_offset * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_HOST_DEVICE
    void clear_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            masks_[s][0] = 0;
            masks_[s][1] = 0;
        }
    }

    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecompNHWC& operator++() {
        if (constant_offset_ < params_.constant_offset_max_) {
            constant_offset_ += Shape::kColumn / kAccessSize;
        } else {
            constant_offset_ += params_.constant_offset_rewind_;
            c_offset_ += Shape::kColumn;
        }
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecompNHWC operator++(int) {
        Conv2dTileSrcIteratorFpropPrecompNHWC self(*this);
        operator++();
        return self;
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        load_with_byte_offset(frag,
                              pointer_offset * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
            detail::PrecompRSC precomp_crs =
                    *(detail::PrecompRSC*)&params_
                             .constant_offset_[constant_offset_ + v];

            uint32_t filter_s = precomp_crs.fw;
            uint32_t filter_r = precomp_crs.fh;
            uint32_t filter_c = precomp_crs.c + c_offset_;

            bool guard_ = (filter_c < params_.C);

            TensorCoord coord = TensorCoord(0, filter_r, filter_s, filter_c);

            Index stride =
                    params_.layout_(coord) * sizeof_bits<Element>::value / 8;

            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
                bool guard = guard_ &&
                             (masks_[s][0] & (Index(1) << filter_r)) &&
                             (masks_[s][1] & (Index(1) << filter_s));

                char const* byte_ptr =
                        reinterpret_cast<char const*>(pointer_[s] + stride) +
                        byte_offset;

                AccessType const* access_ptr =
                        reinterpret_cast<AccessType const*>(byte_ptr);

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                        frag_ptr[s * kAccessesPerVector + v], access_ptr, guard,
                        params_.pack_pad_);
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    static Status can_implement(ConvProblemSize& problem_size) {
        if (problem_size.mode != Mode::kCrossCorrelation) {
            return Status::kErrorInvalidProblem;
        }

        if (problem_size.R * problem_size.S > Params::kMaxFilterPixels) {
            return Status::kErrorInvalidProblem;
        }

        if (problem_size.dilation_h != 1 || problem_size.dilation_w != 1) {
            return Status::kErrorInvalidProblem;
        }

        return Status::kSuccess;
    }
};

template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize>
class Conv2dTileSrcIteratorFpropPrecompNHWC<
        Shape_, Element_, layout::TensorNHWC, ThreadMap_, AccessSize,
        SpecialOptimizeDesc::CONV_FILTER_UNITY> {
public:

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNHWC;
    using TensorCoord = typename Layout::TensorCoord;
    using ThreadMap = ThreadMap_;
    static int const kAccessSize = AccessSize;
    using AccessType = AlignedArray<Element, kAccessSize>;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using ConvProblemSize = typename conv::Conv2dProblemSize;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static_assert(ThreadMap::Iterations::kContiguous == 1,
                  "Require Iterations::kContiguous == 1");

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;


    using Params =
            FpropPrecompNHWCParams<Shape, Element, Layout, ThreadMap,
                                   kAccessSize,
                                   SpecialOptimizeDesc::CONV_FILTER_UNITY>;

private:
    Params const& params_;

    Index c_offset_;

    char const* pointer_[ThreadMap::Iterations::kStrided];

    uint32_t predicates_;

    static_assert(ThreadMap::Iterations::kStrided < sizeof(uint32_t) * 8,
                  "Currently, the number of loads per iteration is limited by "
                  "the size of the predicates container.");

public:
    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecompNHWC(Params const& params,
                                          Element const* ptr,
                                          LogicalCoord extent, int thread_idx,
                                          MatrixCoord const& threadblock_offset)
            : params_(params) {
        layout::PitchLinearCoord thread_coord =
                ThreadMap::initial_offset(thread_idx);

        c_offset_ = thread_coord.contiguous();

        predicates_ = 0;

        int offset_n[ThreadMap::Iterations::kStrided];
        int offset_p[ThreadMap::Iterations::kStrided];
        int offset_q[ThreadMap::Iterations::kStrided];

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] = reinterpret_cast<char const*>(ptr);

            int offset_npq = threadblock_offset.row() + thread_coord.strided() +
                             s * ThreadMap::Delta::kStrided;

            int residual;

            params_.pq_divmod(offset_n[s], residual, offset_npq);
            params_.q_divmod(offset_p[s], offset_q[s], residual);

            TensorCoord coord = TensorCoord(
                    offset_n[s], offset_p[s] * params_.stride_h - params_.pad_h,
                    offset_q[s] * params_.stride_w - params_.pad_w, 0);

            pointer_[s] +=
                    params_.layout_(coord) * sizeof_bits<Element>::value / 8;

            uint32_t pred = ((coord.n() < params_.N && coord.h() >= 0 &&
                              coord.h() < params_.H && coord.w() >= 0 &&
                              coord.w() < params_.W)
                                     ? 1u
                                     : 0);
            predicates_ |= (pred << s);
        }
    }

private:
    CUTLASS_HOST_DEVICE
    void add_byte_offset_(LongIndex byte_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] += byte_offset;
        }
    }

public:
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        add_byte_offset_(pointer_offset * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_HOST_DEVICE
    void clear_mask() { predicates_ = 0; }

    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecompNHWC& operator++() {
        c_offset_ += Shape::kColumn;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Conv2dTileSrcIteratorFpropPrecompNHWC operator++(int) {
        Conv2dTileSrcIteratorFpropPrecompNHWC self(*this);
        operator++();
        return self;
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        load_with_byte_offset(frag,
                              pointer_offset * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
            uint32_t cur_c = c_offset_ + v * kAccessSize;

            bool guard_ = (cur_c < params_.C);

            Index stride = cur_c * sizeof_bits<Element>::value / 8;

            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
                bool guard = guard_ && (predicates_ & (1u << s));

                char const* byte_ptr =
                        reinterpret_cast<char const*>(pointer_[s] + stride) +
                        byte_offset;

                AccessType const* access_ptr =
                        reinterpret_cast<AccessType const*>(byte_ptr);

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                        frag_ptr[s * kAccessesPerVector + v], access_ptr, guard,
                        params_.pack_pad_);
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    static Status can_implement(ConvProblemSize& problem_size) {
        if (problem_size.mode != Mode::kCrossCorrelation) {
            return Status::kErrorInvalidProblem;
        }

        if (problem_size.R != 1 || problem_size.S != 1) {
            return Status::kErrorInvalidProblem;
        }

        if (problem_size.dilation_h != 1 || problem_size.dilation_w != 1) {
            return Status::kErrorInvalidProblem;
        }

        return Status::kSuccess;
    }
};

template <typename Shape_, typename Element_, typename Layout,
          typename ThreadMap_, int AccessSize>
class Conv2dTileFilterIteratorFpropKCxRSx;

template <typename Shape_, typename Element_, typename ThreadMap_,
          int AccessSize>
class Conv2dTileFilterIteratorFpropKCxRSx<Shape_, Element_,
                                          layout::TensorNCxHWx<AccessSize>,
                                          ThreadMap_, AccessSize> {
public:
    static int const kAccessSize = AccessSize;
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNCxHWx<kAccessSize>;
    using TensorCoord = typename Layout::TensorCoord;
    using ThreadMap = ThreadMap_;
    using AccessType = AlignedArray<Element, kAccessSize>;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using ConvProblemSize = typename conv::Conv2dProblemSize;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static_assert(ThreadMap::Iterations::kContiguous == 1,
                  "Require Iterations::kContiguous == 1");

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;


    class Params {
    public:
        int stride_crs_, K;
        CUTLASS_HOST_DEVICE
        Params() : stride_crs_(0), K(0) {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, Conv2dProblemSize const& problem_size)
                : stride_crs_(layout.stride()[2]), K(problem_size.K) {}
    };

private:
    Params const& params_;

    Index crs_;

    char const* pointer_[ThreadMap::Iterations::kStrided];

    uint32_t predicates_;

    static_assert(ThreadMap::Iterations::kStrided < sizeof(uint32_t) * 8,
                  "Currently, the number of loads per iteration is limited by "
                  "the size of the predicates container.");

public:
    CUTLASS_HOST_DEVICE
    Conv2dTileFilterIteratorFpropKCxRSx(Params const& params,
                                        Element const* ptr, LogicalCoord extent,
                                        int thread_idx,
                                        MatrixCoord const& threadblock_offset)
            : params_(params), predicates_(0) {
        layout::PitchLinearCoord thread_coord =
                ThreadMap::initial_offset(thread_idx);

        crs_ = threadblock_offset.row() + thread_coord.contiguous();

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] = reinterpret_cast<char const*>(ptr);

            Index offset_k = threadblock_offset.column() +
                             thread_coord.strided() +
                             s * ThreadMap::Delta::kStrided;
            uint32_t pred = ((offset_k < params_.K) ? 1u : 0);
            predicates_ |= (pred << s);

            pointer_[s] += offset_k * params_.stride_crs_ *
                           sizeof_bits<Element>::value / 8;
        }
    }

private:
    CUTLASS_HOST_DEVICE
    void add_byte_offset_(LongIndex byte_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            pointer_[s] += byte_offset;
        }
    }

public:
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        add_byte_offset_(pointer_offset * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_HOST_DEVICE
    void clear_mask() { predicates_ = 0; }

    CUTLASS_HOST_DEVICE
    Conv2dTileFilterIteratorFpropKCxRSx& operator++() {
        crs_ += Shape::kRow;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Conv2dTileFilterIteratorFpropKCxRSx operator++(int) {
        Conv2dTileFilterIteratorFpropKCxRSx self(*this);
        operator++();
        return self;
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        load_with_byte_offset(frag,
                              pointer_offset * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
            uint32_t cur_crs = crs_ + v * kAccessSize;

            bool guard_ = (cur_crs < params_.stride_crs_);

            Index stride = cur_crs * sizeof_bits<Element>::value / 8;

            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
                bool guard = guard_ && (predicates_ & (1u << s));

                char const* byte_ptr =
                        reinterpret_cast<char const*>(pointer_[s] + stride) +
                        byte_offset;

                AccessType const* access_ptr =
                        reinterpret_cast<AccessType const*>(byte_ptr);

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                        frag_ptr[s * kAccessesPerVector + v], access_ptr,
                        guard);
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }
};


}
}
}

