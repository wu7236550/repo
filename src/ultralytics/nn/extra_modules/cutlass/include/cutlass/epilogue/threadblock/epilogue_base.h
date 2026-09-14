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

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <typename Shape_,
          typename WarpShape_,
          int PartitionsK,
          typename AccumulatorFragmentIterator_,
          typename WarpTileIterator_,
          typename Padding_
          >
class EpilogueBase {
public:
    using Shape = Shape_;
    using WarpShape = WarpShape_;
    static int const kPartitionsK = PartitionsK;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using WarpTileIterator = WarpTileIterator_;
    using Padding = Padding_;

    using Layout = layout::RowMajor;

    using AccumulatorTile =
            typename AccumulatorFragmentIterator::AccumulatorTile;

    using ElementAccumulator = typename AccumulatorTile::Element;

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, kPartitionsK>;

public:
    struct SharedStorage {

        using Element = typename WarpTileIterator::Element;

        using TensorRef = typename WarpTileIterator::TensorRef;

        using Layout = typename WarpTileIterator::Layout;

        using Shape =
                MatrixShape<WarpCount::kM * WarpTileIterator::Shape::kRow *
                                    WarpCount::kK,
                            WarpCount::kN * WarpTileIterator::Shape::kColumn>;

        using StorageShape = MatrixShape<Shape::kRow + Padding::kRow,
                                         Shape::kColumn + Padding::kColumn>;


        AlignedBuffer<Element, StorageShape::kCount> storage;


        CUTLASS_DEVICE
        Element* data() { return storage.data(); }

        CUTLASS_DEVICE
        TensorRef reference() {
            return TensorRef(storage.data(),
                             Layout::packed({StorageShape::kRow,
                                             StorageShape::kColumn}));
        }
    };

protected:

    SharedStorage& shared_storage_;

    WarpTileIterator warp_tile_iterator_;

public:
    CUTLASS_DEVICE
    EpilogueBase(SharedStorage& shared_storage,
                 int thread_idx,
                 int warp_idx,
                 int lane_idx
                 )
            : shared_storage_(shared_storage),
              warp_tile_iterator_(shared_storage.reference(), lane_idx) {

        int warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
        int warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
        int warp_m = warp_mn % WarpCount::kM;
        int warp_n = warp_mn / WarpCount::kM;

        MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m, warp_n};

        warp_tile_iterator_.add_tile_offset(warp_offset);
    }
};


}
}
}

