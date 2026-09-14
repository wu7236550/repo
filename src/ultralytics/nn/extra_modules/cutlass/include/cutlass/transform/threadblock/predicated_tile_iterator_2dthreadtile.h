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

#include "cutlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h"
#include "cutlass/transform/thread/transpose.h"


namespace cutlass {
namespace transform {
namespace threadblock {


template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, bool Transpose = false>
class PredicatedTileIterator2dThreadTile;


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, bool Transpose_>
class PredicatedTileIterator2dThreadTile<Shape_, Element_, layout::PitchLinear,
                                         AdvanceRank, ThreadMap_, Transpose_> {
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

    struct alignas((ThreadMap::kElementsPerAccess *
                    sizeof_bits<Element>::value / 8)) AccessType {
        Array<Element, ThreadMap::kElementsPerAccess> storage;

        static int const kElements = ThreadMap::kElementsPerAccess;
    };

    using Transform =
            thread::Transpose<ThreadMap::Iterations::kCount *
                                      ThreadMap::ThreadAccessShape::kCount,
                              layout::PitchLinearShape<4, 4>, Element>;
    static bool const transpose = Transpose_;

    using TileAccessIterator = PredicatedTileAccessIterator2dThreadTile<
            Shape, Element, Layout, kAdvanceRank, ThreadMap, AccessType>;

    using Fragment =
            cutlass::Array<Element,
                           ThreadMap::Iterations::kCount *
                                   ThreadMap::ThreadAccessShape::kCount>;

    using Mask = typename TileAccessIterator::Mask;

    class Params {
    public:
        friend PredicatedTileIterator2dThreadTile;

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
    PredicatedTileIterator2dThreadTile(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const& threadblock_offset)
            : address_iterator_(params.params_, pointer, extent, thread_id,
                                threadblock_offset) {}

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator2dThreadTile(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileIterator2dThreadTile(params, pointer, extent,
                                                 thread_id, make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        address_iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator2dThreadTile& operator++() {
        if (kAdvanceRank)
            address_iterator_.add_tile_offset({0, 1});
        else
            address_iterator_.add_tile_offset({1, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator2dThreadTile operator++(int) {
        PredicatedTileIterator2dThreadTile self(*this);
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
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided;
                     ts++) {
                    int access_idx =
                            ts + c * ThreadMap::ThreadAccessShape::kStrided +
                            s * ThreadMap::Iterations::kContiguous *
                                    ThreadMap::ThreadAccessShape::kStrided;

                    address_iterator_.set_iteration_index(access_idx);
                    if (address_iterator_.valid()) {
                        frag_ptr[access_idx] =
                                *(address_iterator_.get() + pointer_offset);
                    }

                    ++address_iterator_;
                }
            }
        }

        if (transpose) {
            Transform t;
            t.transform(frag, frag);
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided;
                     ts++) {
                    int access_idx =
                            ts + c * ThreadMap::ThreadAccessShape::kStrided +
                            s * ThreadMap::Iterations::kContiguous *
                                    ThreadMap::ThreadAccessShape::kStrided;

                    address_iterator_.set_iteration_index(access_idx);
                    if (address_iterator_.valid()) {
                        *(address_iterator_.get() + pointer_offset) =
                                frag_ptr[access_idx];
                    }
                    ++address_iterator_;
                }
            }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, bool Transpose_>
class PredicatedTileIterator2dThreadTile<Shape_, Element_, layout::ColumnMajor,
                                         AdvanceRank, ThreadMap_, Transpose_> {
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
    static bool const Transpose = Transpose_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileIterator2dThreadTile<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
            layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap,
            Transpose>;

    using AccessType = typename UnderlyingIterator::AccessType;

    using Fragment =
            cutlass::Array<Element,
                           ThreadMap::Iterations::kCount *
                                   ThreadMap::ThreadAccessShape::kCount>;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileIterator2dThreadTile;

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
    PredicatedTileIterator2dThreadTile(
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
    PredicatedTileIterator2dThreadTile(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileIterator2dThreadTile(params, pointer, extent,
                                                 thread_id, make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator2dThreadTile& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator2dThreadTile operator++(int) {
        PredicatedTileIterator2dThreadTile self(*this);
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
          typename ThreadMap_, bool Transpose_>
class PredicatedTileIterator2dThreadTile<Shape_, Element_, layout::RowMajor,
                                         AdvanceRank, ThreadMap_, Transpose_> {
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
    static bool const Transpose = Transpose_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileIterator2dThreadTile<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
            layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap,
            Transpose>;

    using AccessType = typename UnderlyingIterator::AccessType;

    using Fragment =
            cutlass::Array<Element,
                           ThreadMap::Iterations::kCount *
                                   ThreadMap::ThreadAccessShape::kCount>;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileIterator2dThreadTile;

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
    PredicatedTileIterator2dThreadTile(
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
    PredicatedTileIterator2dThreadTile(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileIterator2dThreadTile(params, pointer, extent,
                                                 thread_id, make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator2dThreadTile& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIterator2dThreadTile operator++(int) {
        PredicatedTileIterator2dThreadTile self(*this);
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

