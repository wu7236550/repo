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
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"



namespace cutlass {
namespace transform {
namespace threadblock {


template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, typename AccessType>
class PredicatedTileAccessIterator;


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PredicatedTileAccessIterator<Shape_, Element_, layout::PitchLinear,
                                   AdvanceRank, ThreadMap_, AccessType_> {
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
    using AccessType = AccessType_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    static int const kAccessesPerVector =
            ThreadMap::kElementsPerAccess / AccessType::kElements;

    static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
                  "Vectors implied by the thread map must be divisible by the "
                  "access type.");

    static int const kPredicatesPerByte = 4;
    static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

    static int const kPredicateCount =
            ThreadMap::Iterations::kCount * kAccessesPerVector;

    static int const kPredicateByteCount =
            (kPredicateCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
    static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;

    static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

    static_assert(kPredicateWordCount <= 4, "Too many predicates.");

    using Mask = Array<uint32_t, kPredicateWordCount>;

    class Params {
    public:
        friend PredicatedTileAccessIterator;

    private:
        int stride_;
        LongIndex inc_strided_;
        LongIndex inc_next_;
        LongIndex inc_advance_;

    public:
        CUTLASS_HOST_DEVICE
        Params() : stride_(0), inc_strided_(0), inc_next_(0), inc_advance_(0) {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout) : stride_(layout.stride(0)) {
            inc_strided_ = (LongIndex(stride_) * ThreadMap::Delta::kStrided) *
                           sizeof_bits<Element>::value / 8;

            if (kAdvanceRank) {
                inc_advance_ = Shape::kStrided * LongIndex(stride_) *
                               sizeof_bits<Element>::value / 8;
            } else {
                inc_advance_ =
                        Shape::kContiguous * sizeof_bits<Element>::value / 8;
            }

            inc_next_ = inc_advance_ -
                        LongIndex(ThreadMap::Iterations::kStrided - 1) *
                                ThreadMap::Delta::kStrided *
                                LongIndex(stride_) *
                                sizeof_bits<Element>::value / 8;
        };
    };

private:
    using BytePointer = char*;

private:

    Params const& params_;

    BytePointer pointer_;

    uint32_t predicates_[kPredicateWordCount];

    TensorCoord extent_;

    TensorCoord thread_offset_;

    TensorCoord residue_offset_;

    bool is_residue_tile_;

    int iteration_vector_;

    int iteration_contiguous_;

    int iteration_strided_;

private:
    CUTLASS_DEVICE
    void compute_predicates_(
            TensorCoord extent,
            bool is_steady_state = false) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0u;
        }

        CUTLASS_PRAGMA_UNROLL
        for (int access_idx = 0;
             access_idx < ThreadMap::Iterations::kCount * kAccessesPerVector;
             ++access_idx) {
            int s = access_idx /
                    (ThreadMap::Iterations::kContiguous * kAccessesPerVector);

            int access_residual =
                    access_idx %
                    (ThreadMap::Iterations::kContiguous * kAccessesPerVector);

            int c = access_residual / kAccessesPerVector;
            int v = access_residual % kAccessesPerVector;

            TensorCoord iteration_coord(c * ThreadMap::Delta::kContiguous +
                                                v * AccessType::kElements,
                                        s * ThreadMap::Delta::kStrided);

            TensorCoord coord = thread_offset_ + iteration_coord;

            bool guard;

            if (is_steady_state) {
                if (kAdvanceRank == 0) {
                    guard = (coord.strided() < extent.strided());
                } else {
                    guard = (coord.contiguous() < extent.contiguous());
                }
            } else {
                guard = (coord.strided() < extent.strided() &&
                         coord.contiguous() < extent.contiguous());
            }

            int pred_idx =
                    v + kAccessesPerVector *
                                (c + ThreadMap::Iterations::kContiguous * s);

            int word_idx = pred_idx / kPredicatesPerWord;
            int residual = pred_idx % kPredicatesPerWord;
            int byte_idx = residual / kPredicatesPerByte;
            int bit_idx = residual % kPredicatesPerByte;

            predicates_[word_idx] |=
                    (unsigned(guard) << (byte_idx * 8 + bit_idx));
        }
    }

public:
    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const& threadblock_offset)
            : params_(params),
              pointer_(reinterpret_cast<BytePointer>(
                      const_cast<NonConstPointer>(pointer))),
              extent_(extent),
              is_residue_tile_(true) {
        TensorCoord residue_extent;
        if (kAdvanceRank) {
            Index residue_size =
                    (extent_[kAdvanceRank] - threadblock_offset.strided()) %
                    Shape::kStrided;
            if (!residue_size) {
                residue_size = Shape::kStrided;
            }

            residue_offset_ = make_Coord(0, residue_size);
            residue_extent =
                    make_Coord(extent_.contiguous(),
                               min(threadblock_offset.strided() + residue_size,
                                   extent_.strided()));
        } else {
            Index residue_size =
                    (extent_[kAdvanceRank] - threadblock_offset.contiguous()) %
                    Shape::kContiguous;
            if (!residue_size) {
                residue_size = Shape::kContiguous;
            }

            residue_offset_ = make_Coord(residue_size, 0);

            residue_extent = make_Coord(
                    min(extent_.contiguous(),
                        threadblock_offset.contiguous() + residue_size),
                    extent_.strided());
        }

        thread_offset_ =
                threadblock_offset + ThreadMap::initial_offset(thread_id);

        Layout layout(params_.stride_);
        add_pointer_offset(layout(thread_offset_));

        compute_predicates_(residue_extent, false);

        set_iteration_index(0);
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id)
            : PredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                           make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
        iteration_vector_ = index % kAccessesPerVector;
        int residual_access = index / kAccessesPerVector;

        iteration_contiguous_ =
                residual_access % ThreadMap::Iterations::kContiguous;
        iteration_strided_ =
                residual_access / ThreadMap::Iterations::kContiguous;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        pointer_ += sizeof_bits<Element>::value * pointer_offset / 8;
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& tile_offset) {
        if (is_residue_tile_) {
            thread_offset_ += residue_offset_;

            Layout layout(params_.stride_);
            add_pointer_offset(layout(residue_offset_));

            compute_predicates_(extent_, true);

            if (kAdvanceRank) {
                pointer_ += params_.inc_advance_ *
                            LongIndex(tile_offset.strided() - 1);
                pointer_ += Shape::kContiguous * tile_offset.contiguous();
            } else {
                pointer_ += params_.inc_advance_ *
                            LongIndex(tile_offset.contiguous() - 1);
                pointer_ += Shape::kStrided * tile_offset.strided();
            }
        } else {
            if (kAdvanceRank) {
                pointer_ +=
                        params_.inc_advance_ * LongIndex(tile_offset.strided());
                pointer_ += Shape::kContiguous * tile_offset.contiguous();
            } else {
                pointer_ += params_.inc_advance_ *
                            LongIndex(tile_offset.contiguous());
                pointer_ += Shape::kStrided * tile_offset.strided();
            }
        }
        is_residue_tile_ = false;
    }

    CUTLASS_HOST_DEVICE
    AccessType* get() const {
        return reinterpret_cast<AccessType*>(
                       pointer_ + iteration_contiguous_ *
                                          (ThreadMap::Delta::kContiguous *
                                           sizeof_bits<Element>::value) /
                                          8) +
               iteration_vector_;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator& operator++() {
        ++iteration_vector_;
        if (iteration_vector_ < kAccessesPerVector) {
            return *this;
        }

        iteration_vector_ = 0;
        ++iteration_contiguous_;

        if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
            return *this;
        }

        iteration_contiguous_ = 0;
        ++iteration_strided_;

        if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
            pointer_ += params_.inc_strided_;
            return *this;
        }

        iteration_strided_ = 0;

        pointer_ += params_.inc_next_;

        pointer_ -= params_.inc_advance_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator operator++(int) {
        PredicatedTileAccessIterator self(*this);
        operator++();
        return self;
    }

    CUTLASS_HOST_DEVICE
    void clear_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0u;
        }
    }

    CUTLASS_HOST_DEVICE
    void enable_mask() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = 0xffffffff;
        }
    }

    CUTLASS_HOST_DEVICE
    void set_mask(Mask const& mask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            predicates_[i] = mask[i];
        }
    }

    CUTLASS_HOST_DEVICE
    void get_mask(Mask& mask) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPredicateWordCount; ++i) {
            mask[i] = predicates_[i];
        }
    }

    CUTLASS_HOST_DEVICE
    bool valid() {
        int pred_idx = iteration_vector_ +
                       kAccessesPerVector *
                               (iteration_contiguous_ +
                                iteration_strided_ *
                                        ThreadMap::Iterations::kContiguous);

        int word_idx = pred_idx / kPredicatesPerWord;
        int residual = pred_idx % kPredicatesPerWord;
        int byte_idx = residual / kPredicatesPerByte;
        int bit_idx = residual % kPredicatesPerByte;

        bool pred =
                (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
        return pred;

    }
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PredicatedTileAccessIterator<Shape_, Element_, layout::ColumnMajor,
                                   AdvanceRank, ThreadMap_, AccessType_> {
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
    using AccessType = AccessType_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileAccessIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
            layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap,
            AccessType>;

    using Mask = typename UnderlyingIterator::Mask;

    static int const kAccessesPerVector =
            UnderlyingIterator::kAccessesPerVector;

    class Params {
    private:
        friend PredicatedTileAccessIterator;

        typename UnderlyingIterator::Params params_;

    public:
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
                : params_(layout::PitchLinear(layout.stride(0))){};
    };

private:

    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const& threadblock_offset)
            : iterator_(params.params_, pointer,
                        layout::PitchLinearCoord(extent.row(), extent.column()),
                        thread_id,
                        layout::PitchLinearCoord(threadblock_offset.row(),
                                                 threadblock_offset.column())) {
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                           make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
        iterator_.set_iteration_index(index);
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
    }

    CUTLASS_HOST_DEVICE
    AccessType* get() const {
        return reinterpret_cast<AccessType*>(iterator_.get());
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator operator++(int) {
        PredicatedTileAccessIterator self(*this);
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

    CUTLASS_HOST_DEVICE
    bool valid() { return iterator_.valid(); }
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PredicatedTileAccessIterator<Shape_, Element_, layout::RowMajor,
                                   AdvanceRank, ThreadMap_, AccessType_> {
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
    using AccessType = AccessType_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileAccessIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
            layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap,
            AccessType>;

    static int const kAccessesPerVector =
            UnderlyingIterator::kAccessesPerVector;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileAccessIterator;

        typename UnderlyingIterator::Params params_;

    public:
        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
                : params_(layout::PitchLinear(layout.stride(0))){};
    };

private:

    UnderlyingIterator iterator_;

public:
    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id,
            TensorCoord const& threadblock_offset)
            : iterator_(params.params_, pointer,
                        layout::PitchLinearCoord(extent.column(), extent.row()),
                        thread_id,
                        layout::PitchLinearCoord(threadblock_offset.column(),
                                                 threadblock_offset.row())) {}

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                           make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
        iterator_.set_iteration_index(index);
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
    }

    CUTLASS_HOST_DEVICE
    AccessType* get() const {
        return reinterpret_cast<AccessType*>(iterator_.get());
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator operator++(int) {
        PredicatedTileAccessIterator self(*this);
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

    CUTLASS_HOST_DEVICE
    bool valid() { return iterator_.valid(); }
};



template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, int InterleavedK>
class PredicatedTileAccessIterator<Shape_, Element_,
                                   layout::ColumnMajorInterleaved<InterleavedK>,
                                   AdvanceRank, ThreadMap_, AccessType_> {
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
    using AccessType = AccessType_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileAccessIterator<
            layout::PitchLinearShape<Shape::kRow * kInterleavedK,
                                     Shape::kColumn / kInterleavedK>,
            Element, layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1),
            ThreadMap, AccessType>;

    static int const kAccessesPerVector =
            UnderlyingIterator::kAccessesPerVector;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileAccessIterator;

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
    PredicatedTileAccessIterator(
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
    PredicatedTileAccessIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                           make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
        iterator_.set_iteration_index(index);
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
    }

    CUTLASS_HOST_DEVICE
    AccessType* get() const {
        return reinterpret_cast<AccessType*>(iterator_.get());
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator operator++(int) {
        PredicatedTileAccessIterator self(*this);
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

    CUTLASS_HOST_DEVICE
    bool valid() { return iterator_.valid(); }
};


template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, int InterleavedK>
class PredicatedTileAccessIterator<Shape_, Element_,
                                   layout::RowMajorInterleaved<InterleavedK>,
                                   AdvanceRank, ThreadMap_, AccessType_> {
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
    using AccessType = AccessType_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using UnderlyingIterator = PredicatedTileAccessIterator<
            layout::PitchLinearShape<Shape::kColumn * kInterleavedK,
                                     Shape::kRow / kInterleavedK>,
            Element, layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0),
            ThreadMap, AccessType>;

    static int const kAccessesPerVector =
            UnderlyingIterator::kAccessesPerVector;

    using Mask = typename UnderlyingIterator::Mask;

    class Params {
    private:
        friend PredicatedTileAccessIterator;

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
    PredicatedTileAccessIterator(
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
    PredicatedTileAccessIterator(
            Params const& params,
            Pointer pointer,
            TensorCoord extent,
            int thread_id
            )
            : PredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                           make_Coord(0, 0)) {}

    CUTLASS_HOST_DEVICE
    void set_iteration_index(int index) {
        iterator_.set_iteration_index(index);
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        iterator_.add_pointer_offset(pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    void add_tile_offset(TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
    }

    CUTLASS_HOST_DEVICE
    AccessType* get() const {
        return reinterpret_cast<AccessType*>(iterator_.get());
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator& operator++() {
        ++iterator_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileAccessIterator operator++(int) {
        PredicatedTileAccessIterator self(*this);
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

    CUTLASS_HOST_DEVICE
    bool valid() { return iterator_.valid(); }
};


}
}
}

