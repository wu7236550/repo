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

#include "cutlass/array.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/epilogue/warp/tensor_op_policy.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape,
          typename OperatorShape,
          typename Element,
          typename Layout
          >
class TileIteratorTensorOp;


template <typename WarpShape_,
          typename OperatorShape_,
          typename Element_
          >
class TileIteratorTensorOp<WarpShape_, OperatorShape_, Element_,
                           layout::RowMajor> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using Element = Element_;
    using Layout = layout::RowMajor;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord =
            MatrixCoord;
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Shape = MatrixShape<Policy::kRowsPerIteration, WarpShape::kN>;

    using Fragment = Array<Element, Policy::OperatorCount::kColumn *
                                            Policy::kElementsPerAccess>;


    static int const kIterations = Policy::kIterations;

    struct Detail {
        static int const kLanesInQuad = 4;
    };

    using Padding =
            MatrixShape<0, Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:
    using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;


    AccessType* pointer_;

    Layout layout_;

    MatrixCoord thread_offset_;

public:
    CUTLASS_HOST_DEVICE
    TileIteratorTensorOp() : pointer_(nullptr) {}

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOp(TensorRef const& ref, unsigned lane_id)
            : pointer_(reinterpret_cast<AccessType*>(ref.data())),
              layout_(ref.stride()[0] / Policy::kElementsPerAccess) {
        int quad_id = (lane_id / Detail::kLanesInQuad);
        int lane_in_quad = (lane_id % Detail::kLanesInQuad);

        thread_offset_ = {quad_id, lane_in_quad * Policy::kElementsPerAccess};

        pointer_ +=
                layout_({thread_offset_.row(),
                         thread_offset_.column() / Policy::kElementsPerAccess});
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOp& add_pointer_offset(Index pointer_offset) {
        pointer_ += pointer_offset / Policy::kElementsPerAccess;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOp& add_tile_offset(TensorCoord const& tile_offset) {
        MatrixCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);

        thread_offset_ += coord_offset;

        pointer_ +=
                layout_({coord_offset.row(),
                         coord_offset.column() / Policy::kElementsPerAccess});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOp& operator+=(TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            pointer_[n * Detail::kLanesInQuad +
                     pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n];
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            frag_ptr[n] = pointer_[n * Detail::kLanesInQuad +
                                   pointer_offset / Policy::kElementsPerAccess];
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOp& operator++() { return add_tile_offset({1, 0}); }
};


template <typename WarpShape_,
          typename OperatorShape_,
          typename Element_,
          typename Layout_>
class TileIteratorTensorOpCanonical {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using Element = Element_;
    using Layout = Layout_;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord =
            MatrixCoord;
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

    static int const kAccessSize = 1;
    static int const kAccessCount = Policy::kElementsPerAccess / kAccessSize;

    using Shape = MatrixShape<Policy::kRowsPerIteration, WarpShape::kN>;

    using Fragment = Array<Element, Policy::OperatorCount::kColumn *
                                            Policy::kElementsPerAccess>;


    static int const kIterations = Policy::kIterations;

    struct Detail {
        static int const kLanesInQuad = 4;
    };

    using Padding =
            MatrixShape<0, Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:
    using AccessType = AlignedArray<Element, kAccessSize>;


    AccessType* pointer_;

    Layout layout_;

    bool divisible_;

    MatrixCoord extent_;

    MatrixCoord thread_offset_;

public:
    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpCanonical() : pointer_(nullptr) {}

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpCanonical(TensorRef const& ref, unsigned lane_id)
            : pointer_(reinterpret_cast<AccessType*>(ref.data())),
              layout_(ref.stride()[0]),
              divisible_(true),
              extent_(WarpShape::kM, WarpShape::kN) {
        int quad_id = (lane_id / Detail::kLanesInQuad);
        int lane_in_quad = (lane_id % Detail::kLanesInQuad);

        thread_offset_ = {quad_id, lane_in_quad * Policy::kElementsPerAccess};

        pointer_ += layout_({thread_offset_.row(), thread_offset_.column()});
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpCanonical(TensorRef const& ref,
                                  TensorCoord const& extent, unsigned lane_id)
            : pointer_(reinterpret_cast<AccessType*>(ref.data())),
              layout_(ref.stride()[0]),
              divisible_(false),
              extent_(extent) {
        int quad_id = (lane_id / Detail::kLanesInQuad);
        int lane_in_quad = (lane_id % Detail::kLanesInQuad);

        thread_offset_ = {quad_id, lane_in_quad * Policy::kElementsPerAccess};

        pointer_ += layout_({thread_offset_.row(), thread_offset_.column()});
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpCanonical& add_pointer_offset(Index pointer_offset) {
        pointer_ += pointer_offset;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpCanonical& add_tile_offset(
            TensorCoord const& tile_offset) {
        MatrixCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);

        thread_offset_ += coord_offset;

        pointer_ += layout_({coord_offset.row(), coord_offset.column()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpCanonical& operator+=(TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            CUTLASS_PRAGMA_UNROLL
            for (int a = 0; a < kAccessCount; ++a) {
                int ptr_idx = n * Detail::kLanesInQuad * kAccessCount +
                              pointer_offset + a;
                int frag_idx = n * kAccessCount + a;

                int col =
                        thread_offset_.column() +
                        n * Detail::kLanesInQuad * Policy::kElementsPerAccess +
                        a;

                if (divisible_ || (thread_offset_.row() < extent_.row() &&
                                   col < extent_.column())) {
                    pointer_[ptr_idx] = frag_ptr[frag_idx];
                }
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            CUTLASS_PRAGMA_UNROLL
            for (int a = 0; a < kAccessCount; ++a) {
                int ptr_idx = n * Detail::kLanesInQuad * kAccessCount +
                              pointer_offset + a;
                int frag_idx = n * kAccessCount + a;

                int col =
                        thread_offset_.column() +
                        n * Detail::kLanesInQuad * Policy::kElementsPerAccess +
                        a;

                if (divisible_ || (thread_offset_.row() < extent_.row() &&
                                   col < extent_.column())) {
                    frag_ptr[frag_idx] = pointer_[ptr_idx];
                }
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpCanonical& operator++() {
        return add_tile_offset({1, 0});
    }
};


}
}
}

