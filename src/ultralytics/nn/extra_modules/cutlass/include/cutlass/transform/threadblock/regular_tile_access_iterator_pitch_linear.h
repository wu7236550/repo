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
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/transform/threadblock/regular_tile_access_iterator.h"


namespace cutlass {
namespace transform {
namespace threadblock {


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment>
class RegularTileAccessIterator<Shape_, Element_, layout::PitchLinear,
                                AdvanceRank, ThreadMap_, Alignment> {
public:
    static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                  "Specialization for pitch-linear iterator may along advance "
                  "along the "
                  "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::PitchLinear;
    static int const kAdvanceRank = AdvanceRank;
    static int const kAlignment = Alignment;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using ThreadMap = ThreadMap_;

    using AccessType = Array<Element, ThreadMap::kElementsPerAccess>;

private:

    Index stride_;

    AccessType* pointer_;

    Index byte_offset_;

    int iteration_contiguous_;

    int iteration_strided_;

public:
    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator(
            TensorRef ref,
            int thread_id
            )
            : stride_(ref.stride(0) / ThreadMap::kElementsPerAccess),
              byte_offset_(0) {
        layout::PitchLinearCoord thread_offset_base =
                ThreadMap::initial_offset(thread_id);

        pointer_ = reinterpret_cast<AccessType*>(
                ref.data() + ref.offset(thread_offset_base));

        set_iteration_index(0);
    }

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
        iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
        iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_offset_ += pointer_offset * sizeof(Element);
    }

    CUTLASS_DEVICE
    AccessType* get() const {
        AccessType* access_ptr = pointer_;

        int access_offset =
                iteration_strided_ * ThreadMap::Delta::kStrided * stride_ +
                iteration_contiguous_ * ThreadMap::Delta::kContiguous /
                        ThreadMap::kElementsPerAccess;

        char* access_byte_ptr =
                reinterpret_cast<char*>(access_ptr + access_offset);

        return reinterpret_cast<AccessType*>(access_byte_ptr + byte_offset_);
    }

    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator& operator++() {
        ++iteration_contiguous_;

        if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
            return *this;

        iteration_contiguous_ = 0;
        ++iteration_strided_;

        if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
            return *this;
        }

        iteration_strided_ = 0;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator operator++(int) {
        RegularTileAccessIterator prev(*this);
        this->operator++();

        return prev;
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& coord) {
        add_pointer_offset(coord.contiguous() * Shape::kContiguous +
                           coord.strided() * Shape::kStrided * stride_ *
                                   ThreadMap::kElementsPerAccess);
    }
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment>
class RegularTileAccessIterator<Shape_, Element_, layout::ColumnMajor,
                                AdvanceRank, ThreadMap_, Alignment> {
public:
    static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                  "Specialization for pitch-linear iterator may along advance "
                  "along the "
                  "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::ColumnMajor;
    static int const kAdvanceRank = AdvanceRank;
    static int const kAlignment = Alignment;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using ThreadMap = ThreadMap_;

    using UnderlyingIterator = RegularTileAccessIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
            layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

    using AccessType = typename UnderlyingIterator::AccessType;

private:
    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator(
            TensorRef ref,
            int thread_id
            )
            : iterator_({ref.data(), ref.stride()}, thread_id) {}

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
        iterator_.set_iteration_index(index);
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    AccessType* get() const {
        return reinterpret_cast<AccessType*>(iterator_.get());
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& coord) {
        iterator_.add_tile_offset({coord.row(), coord.column()});
    }

    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator operator++(int) {
        RegularTileAccessIterator prev(*this);
        ++iterator_;

        return prev;
    }
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment>
class RegularTileAccessIterator<Shape_, Element_, layout::RowMajor, AdvanceRank,
                                ThreadMap_, Alignment> {
public:
    static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                  "Specialization for pitch-linear iterator may along advance "
                  "along the "
                  "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::RowMajor;
    static int const kAdvanceRank = AdvanceRank;
    static int const kAlignment = Alignment;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using ThreadMap = ThreadMap_;

    using UnderlyingIterator = RegularTileAccessIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
            layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

    using AccessType = typename UnderlyingIterator::AccessType;

private:
    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator(
            TensorRef ref,
            int thread_id
            )
            : iterator_({ref.data(), ref.stride()}, thread_id) {}

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
        iterator_.set_iteration_index(index);
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    AccessType* get() const {
        return reinterpret_cast<AccessType*>(iterator_.get());
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& coord) {
        iterator_.add_tile_offset({coord.column(), coord.row()});
    }

    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    RegularTileAccessIterator operator++(int) {
        RegularTileAccessIterator prev(*this);
        ++iterator_;

        return prev;
    }
};


}
}
}

