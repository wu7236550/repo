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
 * \file include/cutlass/convolution/threadblock/conv2d_tile_iterator_nt.h
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
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"


namespace cutlass {
namespace conv {
namespace threadblock {


template <typename Shape_, typename Element_, typename Layout_,
          int InterleavedK, typename ThreadMap_, int AccessSize,
          typename TileMap_,
          ImplicitGemmMode GemmMode = ImplicitGemmMode::GEMM_NT>
class Conv2dTileIterator {
public:
    using Shape = Shape_;
    using Element = Element_;
    static int const kInterleavedK = InterleavedK;
    static ImplicitGemmMode const kGemmMode = GemmMode;
    using Layout = Layout_;
    using ThreadMap = ThreadMap_;
    using TileMap = TileMap_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::ColumnMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = transform::threadblock::PredicatedTileIterator<
            layout::PitchLinearShape<Shape::kColumn * kInterleavedK,
                                     Shape::kRow / kInterleavedK>,
            Element, layout::PitchLinear, 1, ThreadMap, AccessSize>;

    using AccessType = typename UnderlyingIterator::AccessType;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend Conv2dTileIterator;

        typename UnderlyingIterator::Params params_;

        Layout layout_;

    public:
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout,
               Conv2dProblemSize const& problem_size = Conv2dProblemSize())
                : params_(layout::PitchLinear(
                          layout.stride()[TileMap::kStrideAxis])),
                  layout_(layout) {}
    };

private:

    Params const& params_;

    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    Conv2dTileIterator(
            Params const& params,
            Pointer pointer,
            LogicalCoord extent,
            int thread_id,
            LogicalCoord const& threadblock_offset)
            : params_(params),
              iterator_(params.params_, pointer,
                        (kGemmMode == ImplicitGemmMode::GEMM_NT)
                                ? layout::PitchLinearCoord(
                                          extent.row() * kInterleavedK,
                                          extent.column() / kInterleavedK)
                                : layout::PitchLinearCoord(
                                          extent.column() * kInterleavedK,
                                          extent.row() / kInterleavedK),
                        thread_id,
                        (kGemmMode == ImplicitGemmMode::GEMM_NT)
                                ? layout::PitchLinearCoord(
                                          threadblock_offset.row() *
                                                  kInterleavedK,
                                          threadblock_offset.column() /
                                                  kInterleavedK)
                                : layout::PitchLinearCoord(
                                          threadblock_offset.column() *
                                                  kInterleavedK,
                                          threadblock_offset.row() /
                                                  kInterleavedK)) {}

    CUTLASS_HOST_DEVICE
    Conv2dTileIterator(Params const& params,
                       Pointer pointer,
                       LogicalCoord extent,
                       int thread_id
                       )
            : Conv2dTileIterator(params, pointer, extent, thread_id,
                                 make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    Conv2dTileIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Conv2dTileIterator operator++(int) {
        Conv2dTileIterator self(*this);
        operator++();
        return self;
    }

    CUTLASS_HOST_DEVICE
    void clear_mask() { iterator_.clear_mask(); }

    CUTLASS_HOST_DEVICE
    void enable_mask() { iterator_.enable_mask(); }

    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) { iterator_.set_mask(mask); }

    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) { iterator_.get_mask(mask); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        iterator_.load_with_pointer_offset(frag, pointer_offset);
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        iterator_.store_with_pointer_offset(frag, pointer_offset);
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    Conv2dTileIterator& add_coord_offset(TensorCoord const& coord_offset) {
        add_pointer_offset(params_.layout_(coord_offset));
        return *this;
    }
};

}
}
}

