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

#if !(defined(__clang__) && defined(__CUDA__))

#include "cutlass/cutlass.h"
#include "cutlass/wmma_array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/epilogue/warp/wmma_tensor_op_policy.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape,
          typename OperatorShape,
          typename OperatorFragment,
          typename Layout
          >
class TileIteratorWmmaTensorOp;


template <typename WarpShape_,
          typename OperatorShape_,
          typename OperatorFragment_
          >
class TileIteratorWmmaTensorOp<WarpShape_, OperatorShape_, OperatorFragment_,
                               layout::RowMajor> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using OperatorFragment = OperatorFragment_;
    using Layout = layout::RowMajor;

    using WmmaDataType = typename OperatorFragment::element_type;
    using Element = typename cutlass::arch::WmmaToCutlassDataType<
            WmmaDataType>::Type;
    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord =
            MatrixCoord;
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy = WmmaTensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Shape = MatrixShape<Policy::kRowsPerIteration, WarpShape::kN>;

    using Fragment = WmmaFragmentArray<OperatorFragment,
                                       Policy::OperatorCount::kColumn *
                                               Policy::kWmmaFragmentsPerAccess>;


    using Padding = MatrixShape<0, 4 * Policy::kElementsPerAccess>;

private:


    TensorRef ref_;

public:
    CUTLASS_HOST_DEVICE
    TileIteratorWmmaTensorOp() : ref_(nullptr) {}

    CUTLASS_HOST_DEVICE
    TileIteratorWmmaTensorOp(TensorRef const& ref, unsigned lane_id)
            : ref_(ref) {}

    CUTLASS_HOST_DEVICE
    TileIteratorWmmaTensorOp& add_pointer_offset(Index pointer_offset) {
        ref_.add_pointer_offset(pointer_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorWmmaTensorOp& add_tile_offset(TensorCoord const& tile_offset) {
        ref_.add_coord_offset({tile_offset.row() * OperatorShape::kM,
                               tile_offset.column() * WarpShape::kN});
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorWmmaTensorOp& operator+=(TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        for (int n = 0; n < Policy::OperatorCount::kColumn; n++) {
            WmmaDataType* ptr = reinterpret_cast<WmmaDataType*>(
                    ref_.data() + ref_.offset({0, n * OperatorShape::kN}) +
                    pointer_offset);

            nvcuda::wmma::store_matrix_sync(
                    ptr, frag[n], ref_.stride()[0],
                    nvcuda::wmma::layout_t::mem_row_major);
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        for (int n = 0; n < Policy::OperatorCount::kColumn; n++) {
            WmmaDataType* ptr = reinterpret_cast<WmmaDataType*>(
                    ref_.data() + ref_.offset({0, n * OperatorShape::kN}) +
                    pointer_offset);

            nvcuda::wmma::load_matrix_sync(
                    frag[n], ptr, ref_.stride()[0],
                    nvcuda::wmma::layout_t::mem_row_major);
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }
};


}
}
}


#endif
