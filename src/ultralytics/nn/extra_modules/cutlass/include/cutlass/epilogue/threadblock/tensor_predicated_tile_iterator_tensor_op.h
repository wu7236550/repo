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
 * \file include/cutlass/epilogue/threadblock/tensor_predicated_tile_iterator.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/convolution_output_tile_thread_map.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"


namespace cutlass {


namespace epilogue {
namespace threadblock {

template <typename ThreadMap_,
          typename Layout_,
          typename Element_
          >
class TensorPredicatedTileIteratorTensorOp;


template <typename ThreadMap_, typename Element_, int Interleaved>
class TensorPredicatedTileIteratorTensorOp<
        ThreadMap_, layout::TensorNCxHWx<Interleaved>, Element_> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::TensorNCxHWx<Interleaved>;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using ConvProblemSize = typename conv::Conv2dProblemSize;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kCount;
    static int const kInterleaved = Interleaved;

    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");
    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(
            ThreadMap::kElementsPerAccess <= kInterleaved,
            "Elements per access cannot be greater than interleaving quantity");

    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::kElementsPerAccess>;

    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;


    struct Params {

        LongIndex stride;

        LongIndex increment_row;
        LongIndex advance_row;

        Layout layout_;

        Index hw_, w_;


        CUTLASS_HOST_DEVICE
        Status initialize(Index stride_) {
            stride = LongIndex(stride_);

            increment_row = stride * ThreadMap::Delta::kRow / kInterleaved;

            advance_row = stride * ThreadMap::Shape::kRow / kInterleaved;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() : layout_(Layout()) { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, conv::Operator conv_operator,
               ConvProblemSize const& problem_size)
                : layout_(layout) {
            w_ = (conv_operator == conv::Operator::kFprop) ? problem_size.Q
                                                           : problem_size.W;
            hw_ = (conv_operator == conv::Operator::kFprop)
                          ? problem_size.P * problem_size.Q
                          : problem_size.H * problem_size.W;
            initialize(layout.stride()[1] * sizeof_bits<Element>::value / 8);
        }
        CUTLASS_DEVICE
        TensorCoord operator()(LogicalCoord const& coord) const {
            Index n = coord.column() / hw_;
            Index hw = coord.column() - hw_ * n;
            Index h = hw / w_;
            Index w = hw - w_ * h;
            return TensorCoord(n, h, w, 0);
        }
    };

    struct Mask {
        static int const kCount =
                ThreadMap::Iterations::kRow * ThreadMap::Iterations::kColumn < 8
                        ? 8
                        : ThreadMap::Iterations::kRow *
                                  ThreadMap::Iterations::kColumn;

        bool predicates[kCount];

        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        CUTLASS_DEVICE void enable() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:

    Params params_;

    uint8_t* byte_pointer_;

    Mask mask_;

    Index thread_start_col_;

    Index extent_row_;
    Index extent_col_;

    int state_;

private:

public:
    CUTLASS_DEVICE
    void compute_predicates_() {
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < ThreadMap::Iterations::kRow; ++r) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
                mask_.predicates[r * ThreadMap::Iterations::kColumn + c] =
                        (ThreadMap::Delta::kColumn * c < extent_col_) &&
                        (ThreadMap::Delta::kRow * r < extent_row_);
            }
        }
    }

    CUTLASS_DEVICE
    TensorPredicatedTileIteratorTensorOp(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset =
                ThreadMap::initial_offset(thread_idx) + threadblock_offset;

        thread_start_col_ = thread_offset.column();
        extent_row_ = extent.row() - thread_offset.row();
        extent_col_ = extent.column() - thread_offset.column();

        compute_predicates_();

        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        (thread_offset.row() / kInterleaved) * params_.stride +
                        (thread_offset.row() % kInterleaved) *
                                sizeof_bits<Element>::value / 8;

        state_ = 0;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            CUTLASS_PRAGMA_UNROLL
            for (int column = 0; column < ThreadMap::Iterations::kColumn;
                 ++column) {
                int access_idx = row * ThreadMap::Iterations::kColumn + column;
                bool guard = mask_.predicates[access_idx];
                int col_offset =
                        thread_start_col_ + column * ThreadMap::Delta::kColumn;
                MatrixCoord iteration_coord(0, col_offset);
                TensorCoord coord = params_(iteration_coord);
                AccessType* memory_pointer = reinterpret_cast<AccessType*>(
                        byte_pointer + byte_offset +
                        params_.layout_(coord) * sizeof_bits<Element>::value /
                                8);
                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                        frag_ptr[access_idx], (void*)(memory_pointer), guard);
            }

            if (row + 1 < ThreadMap::Iterations::kRow) {
                byte_pointer += params_.increment_row;
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            CUTLASS_PRAGMA_UNROLL
            for (int column = 0; column < ThreadMap::Iterations::kColumn;
                 ++column) {
                int access_idx = row * ThreadMap::Iterations::kColumn + column;
                bool guard = mask_.predicates[access_idx];
                int col_offset =
                        thread_start_col_ + column * ThreadMap::Delta::kColumn;
                MatrixCoord iteration_coord(0, col_offset);
                TensorCoord coord = params_(iteration_coord);
                AccessType* memory_pointer = reinterpret_cast<AccessType*>(
                        byte_pointer + byte_offset +
                        params_.layout_(coord) * sizeof_bits<Element>::value /
                                8);
                cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                        frag_ptr[access_idx], (void*)(memory_pointer), guard);
            }

            if (row + 1 < ThreadMap::Iterations::kRow) {
                byte_pointer += params_.increment_row;
            }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int iteration) {}

    CUTLASS_HOST_DEVICE
    TensorPredicatedTileIteratorTensorOp& operator++() {
        ++state_;
        thread_start_col_ += ThreadMap::Shape::kColumn;
        extent_col_ -= ThreadMap::Shape::kColumn;

        if (state_ == ThreadMap::Count::kColumn) {
            state_ = 0;
            byte_pointer_ += params_.advance_row;

            thread_start_col_ -=
                    ThreadMap::Count::kColumn * ThreadMap::Shape::kColumn;
            extent_col_ +=
                    ThreadMap::Count::kColumn * ThreadMap::Shape::kColumn;
            extent_row_ -= ThreadMap::Shape::kRow;
        }

        compute_predicates_();

        return *this;
    }

    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    CUTLASS_DEVICE void get_mask(Mask& mask) { return mask_; }

    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }

    CUTLASS_DEVICE
    TensorPredicatedTileIteratorTensorOp& add_coord_offset(
            TensorCoord const& coord_offset) {
        add_pointer_offset(params_.layout_(coord_offset));
        return *this;
    }
};

}
}
}

