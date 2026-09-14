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
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/array_planar_complex.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <typename TileIterator_>
class TileIteratorPlanarComplex {
public:
    using TileIterator = TileIterator_;

    using Element = typename TileIterator::Element;

    using Layout = typename TileIterator::Layout;

    using TensorRef = typename TileIterator::TensorRef;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Fragment =
            ArrayPlanarComplex<Element, TileIterator::Fragment::kElements>;

public:
    TileIterator tile_iterator_;

    LongIndex imaginary_offset_;

public:
    CUTLASS_HOST_DEVICE
    TileIteratorPlanarComplex() : imaginary_offset_(0) {}

    CUTLASS_DEVICE
    TileIteratorPlanarComplex(TensorRef const& ref, int lane_id,
                              LongIndex imaginary_offset)
            : tile_iterator_(ref, lane_id),
              imaginary_offset_(
                      (imaginary_offset * sizeof_bits<Element>::value) / 8) {}

    CUTLASS_DEVICE
    TileIteratorPlanarComplex& add_pointer_offset(LongIndex offset) {
        tile_iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorPlanarComplex& add_tile_offset(TensorCoord const& tile_offset) {
        tile_iterator_.add_tile_offset(tile_offset);

        return *this;
    }

    CUTLASS_DEVICE
    TileIteratorPlanarComplex& operator++() {
        ++tile_iterator_;
        return *this;
    }


    CUTLASS_HOST_DEVICE
    TileIteratorPlanarComplex& operator--() {
        --tile_iterator_;
        return *this;
    }

    CUTLASS_DEVICE
    TileIteratorPlanarComplex& operator+=(TensorCoord const& tile_offset) {
        tile_iterator_.add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    TileIteratorPlanarComplex& operator-=(TensorCoord const& tile_offset) {
        tile_iterator_.add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const {
        tile_iterator_.load_with_byte_offset(frag.real, 0);
        tile_iterator_.load_with_byte_offset(frag.imag, imaginary_offset_);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        tile_iterator_.load_with_byte_offset(frag.real, byte_offset);
        tile_iterator_.load_with_byte_offset(frag.imag,
                                             byte_offset + imaginary_offset_);
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        Index byte_offset = (pointer_offset * sizeof_bits<Element>::value) / 8;

        tile_iterator_.load_with_byte_offset(frag.real, byte_offset);
        tile_iterator_.load_with_byte_offset(frag.imag,
                                             byte_offset + imaginary_offset_);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
        tile_iterator_.load_with_byte_offset(frag.real, tile_offset, 0);
        tile_iterator_.load_with_byte_offset(frag.imag, tile_offset,
                                             imaginary_offset_);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        Index byte_offset = (pointer_offset * sizeof_bits<Element>::value) / 8;

        tile_iterator_.load_with_byte_offset(frag.real, tile_offset,
                                             byte_offset);
        tile_iterator_.load_with_byte_offset(frag.real, tile_offset,
                                             byte_offset + imaginary_offset_);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        tile_iterator_.load_with_byte_offset(frag.real, tile_offset,
                                             byte_offset);
        tile_iterator_.load_with_byte_offset(frag.imag, tile_offset,
                                             byte_offset + imaginary_offset_);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
        tile_iterator_.set_kgroup_index(k_group);
    }
};


}
}
}

