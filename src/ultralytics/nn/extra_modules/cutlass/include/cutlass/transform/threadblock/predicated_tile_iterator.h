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

#include "cutlass/arch/memory.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"


namespace cutlass {
namespace transform {
namespace threadblock {


template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, int AccessSize = ThreadMap::kElementsPerAccess>
class PredicatedTileIterator;


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize>
class PredicatedTileIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank,
                             ThreadMap_, AccessSize> {
public:
    static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                  "Specialization for pitch-linear iterator may along advance "
                  "along the "
                  "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::PitchLinear;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using AccessType =
            AlignedArray<Element, AccessSize,
                         (AccessSize * sizeof_bits<Element>::value / 8)>;

    using TileAccessIterator =
            PredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank,
                                         ThreadMap, AccessType>;

    static int const kAccessesPerVector =
            TileAccessIterator::kAccessesPerVector;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using Mask = typename TileAccessIterator::Mask;

    class Params {
    public:
        friend PredicatedTileIterator;

    private:
        typename TileAccessIterator::Params params_;

    public:
        CUTLASS_HOST_DEVICE
        Params(Layout const& layout) : params_(layout) {}

        CUTLASS_HOST_DEVICE
        Params() {}
    };

private:
    using BytePointer = char*;

private:

    TileAccessIterator address_iterator_;

public:
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const& threadblock_offset)
            : address_iterator_(params.params_, pointer, extent, thread_id,
                                threadblock_offset) {}

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        address_iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++() {
        if (kAdvanceRank)
            address_iterator_.add_tile_offset({0, 1});
        else
            address_iterator_.add_tile_offset({1, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator operator++(int) {
        PredicatedTileIterator self(*this);
        operator++();
        return self;
    }

    CUTLASS_HOST_DEVICE
    void clear_mask() { address_iterator_.clear_mask(); }

    CUTLASS_HOST_DEVICE
    void enable_mask() { address_iterator_.enable_mask(); }

    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) { address_iterator_.set_mask(mask); }

    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) { address_iterator_.get_mask(mask); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        load_with_byte_offset(frag,
                              pointer_offset * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);

                    address_iterator_.set_iteration_index(idx);
                    char const* byte_ptr = reinterpret_cast<char const*>(
                                                   address_iterator_.get()) +
                                           byte_offset;

                    AccessType const* access_ptr =
                            reinterpret_cast<AccessType const*>(byte_ptr);

                    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[idx], access_ptr,
                            address_iterator_.valid());

                    ++address_iterator_;
                }
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        store_with_byte_offset(
                frag, pointer_offset * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
        address_iterator_.set_iteration_index(0);
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int v = 0; v < kAccessesPerVector; ++v) {
                    int idx = v +
                              kAccessesPerVector *
                                      (c +
                                       s * ThreadMap::Iterations::kContiguous);

                    char* byte_ptr =
                            reinterpret_cast<char*>(address_iterator_.get()) +
                            byte_offset;
                    AccessType* access_ptr =
                            reinterpret_cast<AccessType*>(byte_ptr);

                    if (address_iterator_.valid()) {
                        *access_ptr = frag_ptr[idx];
                    }
                    ++address_iterator_;
                }
            }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize>
class PredicatedTileIterator<Shape_, Element_, layout::ColumnMajor, AdvanceRank,
                             ThreadMap_, AccessSize> {
public:
    static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                  "Specialization for pitch-linear iterator may along advance "
                  "along the "
                  "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::ColumnMajor;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
            layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap,
            AccessSize>;

    using AccessType = typename UnderlyingIterator::AccessType;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileIterator;

        typename UnderlyingIterator::Params params_;

    public:
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
                : params_(layout::PitchLinear(layout.stride(0))) {}
    };

private:

    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const&
                    threadblock_offset
            )
            : iterator_(params.params_, pointer,
                        layout::PitchLinearCoord(extent.row(), extent.column()),
                        thread_id,
                        layout::PitchLinearCoord(threadblock_offset.row(),
                                                 threadblock_offset.column())) {
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator operator++(int) {
        PredicatedTileIterator self(*this);
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
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        iterator_.load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        iterator_.store_with_pointer_offset(frag, pointer_offset);
    }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
        iterator_.store_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize>
class PredicatedTileIterator<Shape_, Element_, layout::RowMajor, AdvanceRank,
                             ThreadMap_, AccessSize> {
public:
    static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                  "Specialization for pitch-linear iterator may along advance "
                  "along the "
                  "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::RowMajor;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
            layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap,
            AccessSize>;

    using AccessType = typename UnderlyingIterator::AccessType;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileIterator;

        typename UnderlyingIterator::Params params_;

    public:
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
                : params_(layout::PitchLinear(layout.stride(0))){

                  };
    };

private:

    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const&
                    threadblock_offset
            )
            : iterator_(params.params_, pointer,
                        layout::PitchLinearCoord(extent.column(), extent.row()),
                        thread_id,
                        layout::PitchLinearCoord(threadblock_offset.column(),
                                                 threadblock_offset.row())) {}

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator operator++(int) {
        PredicatedTileIterator self(*this);
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
    void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
        iterator_.load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        iterator_.store_with_pointer_offset(frag, pointer_offset);
    }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
        iterator_.store_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};



template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int InterleavedK>
class PredicatedTileIterator<Shape_, Element_,
                             layout::ColumnMajorInterleaved<InterleavedK>,
                             AdvanceRank, ThreadMap_, AccessSize> {
public:
    static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                  "Specialization for pitch-linear iterator may along advance "
                  "along the "
                  "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    static int const kInterleavedK = InterleavedK;
    using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileIterator<
            layout::PitchLinearShape<Shape::kRow * kInterleavedK,
                                     Shape::kColumn / kInterleavedK>,
            Element, layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1),
            ThreadMap, AccessSize>;

    using AccessType = typename UnderlyingIterator::AccessType;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileIterator;

        typename UnderlyingIterator::Params params_;

    public:
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
                : params_(layout::PitchLinear(layout.stride(0))) {}
    };

private:

    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const& threadblock_offset)
            : iterator_(
                      params.params_, pointer,
                      layout::PitchLinearCoord(extent.row() * kInterleavedK,
                                               extent.column() / kInterleavedK),
                      thread_id,
                      layout::PitchLinearCoord(
                              threadblock_offset.row() * kInterleavedK,
                              threadblock_offset.column() / kInterleavedK)) {}

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator operator++(int) {
        PredicatedTileIterator self(*this);
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
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int InterleavedK>
class PredicatedTileIterator<Shape_, Element_,
                             layout::RowMajorInterleaved<InterleavedK>,
                             AdvanceRank, ThreadMap_, AccessSize> {
public:
    static_assert(AdvanceRank == 0 || AdvanceRank == 1,
                  "Specialization for pitch-linear iterator may along advance "
                  "along the "
                  "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Element = Element_;
    static int const kInterleavedK = InterleavedK;
    using Layout = layout::RowMajorInterleaved<kInterleavedK>;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileIterator<
            layout::PitchLinearShape<Shape::kColumn * kInterleavedK,
                                     Shape::kRow / kInterleavedK>,
            Element, layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0),
            ThreadMap, AccessSize>;

    using AccessType = typename UnderlyingIterator::AccessType;

    using Fragment =
            cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                            ThreadMap::kElementsPerAccess>;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileIterator;

        typename UnderlyingIterator::Params params_;

    public:
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
                : params_(layout::PitchLinear(layout.stride(0))) {}
    };

private:

    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const& threadblock_offset)
            : iterator_(
                      params.params_, pointer,
                      layout::PitchLinearCoord(extent.column() * kInterleavedK,
                                               extent.row() / kInterleavedK),
                      thread_id,
                      layout::PitchLinearCoord(
                              threadblock_offset.column() * kInterleavedK,
                              threadblock_offset.row() / kInterleavedK)) {}

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator operator++(int) {
        PredicatedTileIterator self(*this);
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
};


}
}
}

