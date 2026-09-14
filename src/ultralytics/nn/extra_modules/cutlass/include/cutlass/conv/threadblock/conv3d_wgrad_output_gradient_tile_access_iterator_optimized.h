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
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv3d_problem_size.h"


namespace cutlass {
namespace conv {
namespace threadblock {


template <typename Shape_, typename Element_, typename ThreadMap_>
class Conv3dWgradOutputGradientTileAccessIteratorOptimized {
public:
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::TensorNDHWC;
    using ThreadMap = ThreadMap_;
    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    static IteratorAlgorithm const kIteratorAlgorithm =
            conv::IteratorAlgorithm::kOptimized;
    static StrideSupport const kStrideSupport = conv::StrideSupport::kStrided;
    static int const kConvDim = 3;
    using ConvProblemSize = typename conv::Conv3dProblemSize;

    static_assert(sizeof_bits<Element>::value >= 8,
                  "WGRAD requires elements of size 8b or greater.");


    struct Params {
        Layout layout;

        int NZPQ;
        int ZPQ;
        unsigned zpq_mul;
        unsigned zpq_shr;

        int PQ;
        unsigned pq_mul;
        unsigned pq_shr;

        unsigned q_mul;
        unsigned q_shr;

        LongIndex offset_next_strided;
        LongIndex offset_next_contiguous;
        LongIndex inc_next_nzpq;


        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Conv3dProblemSize const& problem_size, Layout const& layout)
                : layout(layout) {
            offset_next_strided =
                    (ThreadMap::Delta::kStrided * layout.stride()[0]) *
                    sizeof_bits<Element>::value / 8;

            offset_next_contiguous =
                    (ThreadMap::Delta::kContiguous)*sizeof_bits<
                            Element>::value /
                    8;

            inc_next_nzpq = (Shape::kColumn * problem_size.split_k_slices *
                             layout.stride()[0]) *
                            sizeof_bits<Element>::value / 8;

            NZPQ = problem_size.N * problem_size.Z * problem_size.P *
                   problem_size.Q;
            ZPQ = problem_size.Z * problem_size.P * problem_size.Q;
            find_divisor(zpq_mul, zpq_shr, ZPQ);

            PQ = problem_size.P * problem_size.Q;
            find_divisor(pq_mul, pq_shr, PQ);

            find_divisor(q_mul, q_shr, problem_size.Q);
        }
    };

private:
    Params const& params_;
    Conv3dProblemSize const& problem_size_;
    LongIndex iteration_contiguous_;
    LongIndex iteration_strided_;
    char const* pointer_;

    uint32_t predicates_;
    int filter_k_;
    int offset_nzpq_;

public:
    CUTLASS_HOST_DEVICE
    Conv3dWgradOutputGradientTileAccessIteratorOptimized(
            Params const& params, Conv3dProblemSize const& problem_size,
            Element const* ptr, int thread_idx,
            MatrixCoord const& threadblock_offset = MatrixCoord())
            : params_(params),
              problem_size_(problem_size),
              pointer_(reinterpret_cast<char const*>(ptr)),
              predicates_(0),
              filter_k_(0),
              offset_nzpq_(0) {
        layout::PitchLinearCoord thread_coord =
                ThreadMap::initial_offset(thread_idx);

        filter_k_ = threadblock_offset.row() + thread_coord.contiguous();
        offset_nzpq_ = threadblock_offset.column() + thread_coord.strided();

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                int filter_k = filter_k_ + c * ThreadMap::Delta::kContiguous;
                int offset_nzpq = offset_nzpq_ + s * ThreadMap::Delta::kStrided;

                bool predicate = valid_(at_(offset_nzpq, filter_k));

                uint32_t pred = (predicate ? 1u : 0);

                int pred_idx = c + s * ThreadMap::Iterations::kContiguous;

                predicates_ |= (pred << pred_idx);
            }
        }

        pointer_ += (offset_nzpq_ * params.layout.stride()[0] + filter_k_) *
                    sizeof_bits<Element>::value / 8;

        set_iteration_index(0);
    }

    CUTLASS_HOST_DEVICE
    void set_iteration_index(Index index) {
        iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
        iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_HOST_DEVICE
    void advance() {
        offset_nzpq_ += Shape::kColumn * problem_size_.split_k_slices;

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            if (offset_nzpq_ + s * ThreadMap::Delta::kStrided >= params_.NZPQ) {
                uint32_t kClearMask =
                        ((1u << ThreadMap::Iterations::kContiguous) - 1)
                        << (s * ThreadMap::Iterations::kContiguous);
                predicates_ = (predicates_ & (~kClearMask));
            }
        }
        pointer_ += params_.inc_next_nzpq;
    }

private:
    CUTLASS_HOST_DEVICE
    TensorCoord at_(int offset_nzpq, int k) const {

        int residual, n, z, p, q;
        fast_divmod(n, residual, offset_nzpq, params_.ZPQ, params_.zpq_mul,
                    params_.zpq_shr);
        fast_divmod(z, residual, residual, params_.PQ, params_.pq_mul,
                    params_.pq_shr);
        fast_divmod(p, q, residual, problem_size_.Q, params_.q_mul,
                    params_.q_shr);

        return TensorCoord(n, z, p, q, k);
    }

    CUTLASS_HOST_DEVICE
    bool valid_(TensorCoord coord) const {
        return coord.n() < problem_size_.N && coord.c() < problem_size_.K;
    }

public:
    CUTLASS_HOST_DEVICE
    bool valid() const {
        LongIndex pred_idx =
                iteration_contiguous_ +
                iteration_strided_ * ThreadMap::Iterations::kContiguous;
        return (predicates_ & (1u << pred_idx));
    }

    CUTLASS_HOST_DEVICE
    AccessType const* get() const {
        return reinterpret_cast<AccessType const*>(
                pointer_ + iteration_strided_ * params_.offset_next_strided +
                iteration_contiguous_ * params_.offset_next_contiguous);
    }

    CUTLASS_HOST_DEVICE
    Conv3dWgradOutputGradientTileAccessIteratorOptimized& operator++() {
        ++iteration_contiguous_;
        if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
            return *this;
        }
        iteration_contiguous_ = 0;
        ++iteration_strided_;
        if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
            return *this;
        }
        iteration_strided_ = 0;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    static Status can_implement(Conv3dProblemSize const& problem_size) {
        if (problem_size.C % (128 / sizeof_bits<Element>::value)) {
            return Status::kErrorInvalidProblem;
        }

        return Status::kSuccess;
    }
};

}
}
}

