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
 * \file include/cutlass/epilogue/warp/interleaved_tile_iterator_simt.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/epilogue/warp/tensor_op_policy.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape,
          typename OperatorShape,
          typename Element,
          typename SmemLayout,
          typename GmemLayout
          >
class InterleavedTileIteratorTensorOp;

template <typename WarpShape_,
          typename OperatorShape_,
          typename Element_,
          int InterleavedK
          >
class InterleavedTileIteratorTensorOp<WarpShape_, OperatorShape_, Element_,
                                      layout::RowMajor,
                                      layout::TensorNCxHWx<InterleavedK>> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using Element = Element_;
    using SmemLayout = layout::RowMajor;
    using GmemLayout = layout::TensorNCxHWx<InterleavedK>;
    using Layout = SmemLayout;

    using TensorRef =
            TensorRef<Element, SmemLayout>;
    using TensorCoord =
            MatrixCoord;
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy =
            TensorOpPolicy<WarpShape, OperatorShape, SmemLayout, GmemLayout>;

    static int const kInterleavedK = InterleavedK;

    using Shape = typename Policy::Shape;

    using Fragment = Array<Element, kInterleavedK / OperatorShape::kM *
                                            Policy::kElementsPerAccess>;


    static int const kIterations = Policy::kIterations;

    struct Detail {
        static int const kLanesInQuad = 4;
    };

    using Padding = MatrixShape<0, Policy::kElementsPerAccess>;

private:
    using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;


    AccessType* pointer_;

    SmemLayout layout_;

public:
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp() : pointer_(nullptr) {}

    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp(TensorRef const& ref, unsigned lane_id)
            : pointer_(reinterpret_cast<AccessType*>(ref.data())),
              layout_(ref.stride()[0] / Policy::kElementsPerAccess) {
        int quad_id = (lane_id / Detail::kLanesInQuad);
        int lane_in_quad = (lane_id % Detail::kLanesInQuad);

        pointer_ += layout_({quad_id, lane_in_quad});
    }

    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp& add_pointer_offset(Index pointer_offset) {
        pointer_ += pointer_offset / Policy::kElementsPerAccess;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp& add_tile_offset(
            TensorCoord const& tile_offset) {
        pointer_ += layout_({tile_offset.row() * Shape::kRow,
                             (tile_offset.column() * Shape::kColumn /
                              Policy::kElementsPerAccess)});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Fragment::kElements / Policy::kElementsPerAccess;
             ++m) {
            AccessType* pointer =
                    pointer_ + layout_({m * OperatorShape::kM, 0});
            pointer[pointer_offset / Policy::kElementsPerAccess] = frag_ptr[m];
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Fragment::kElements / Policy::kElementsPerAccess;
             ++m) {
            AccessType* pointer =
                    pointer_ + layout_({m * OperatorShape::kM, 0});
            frag_ptr[m] = pointer[pointer_offset / Policy::kElementsPerAccess];
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }
};


template <typename WarpShape_,
          typename OperatorShape_,
          typename Element_
          >
class InterleavedTileIteratorTensorOp<WarpShape_, OperatorShape_, Element_,
                                      layout::RowMajor,
                                      layout::TensorNCxHWx<4>> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using Element = Element_;
    using SmemLayout = layout::RowMajor;
    using GmemLayout = layout::TensorNCxHWx<4>;
    using Layout = SmemLayout;

    using TensorRef =
            TensorRef<Element, SmemLayout>;
    using TensorCoord =
            MatrixCoord;
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy =
            TensorOpPolicy<WarpShape, OperatorShape, SmemLayout, GmemLayout>;

    static int const kInterleavedK = 4;

    using Shape = typename Policy::Shape;

    using Fragment =
            Array<Element, Policy::kColumnsPerIteration / OperatorShape::kN *
                                   Policy::kElementsPerAccess>;


    static int const kIterations = Policy::kIterations;

    struct Detail {
        static int const kLanesInQuad = 4;
    };

    using Padding = MatrixShape<0, Policy::kElementsPerAccess>;

private:
    using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;


    AccessType* pointer_;

    SmemLayout layout_;

public:
    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp() : pointer_(nullptr) {}

    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp(TensorRef const& ref, unsigned lane_id)
            : pointer_(reinterpret_cast<AccessType*>(ref.data())),
              layout_(ref.stride()[0] / Policy::kElementsPerAccess) {
        int quad_id = (lane_id / Detail::kLanesInQuad);
        int lane_in_quad = (lane_id % Detail::kLanesInQuad);

        pointer_ += layout_({quad_id, lane_in_quad});
    }

    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp& add_pointer_offset(Index pointer_offset) {
        pointer_ += pointer_offset / Policy::kElementsPerAccess;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp& add_tile_offset(
            TensorCoord const& tile_offset) {
        pointer_ += layout_({tile_offset.row() * Shape::kRow,
                             (tile_offset.column() * Shape::kColumn /
                              Policy::kElementsPerAccess)});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    InterleavedTileIteratorTensorOp& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Fragment::kElements / Policy::kElementsPerAccess;
             ++n) {
            AccessType* pointer =
                    pointer_ + layout_({0, n * OperatorShape::kN /
                                                   Policy::kElementsPerAccess});
            pointer[pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n];
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Fragment::kElements / Policy::kElementsPerAccess;
             ++n) {
            AccessType* pointer =
                    pointer_ + layout_({0, n * OperatorShape::kN /
                                                   Policy::kElementsPerAccess});
            frag_ptr[n] = pointer[pointer_offset / Policy::kElementsPerAccess];
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }
};


}
}
}

