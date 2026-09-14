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
 * \file include/cutlass/epilogue/threadblock/bias_tile_iterator.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/convolution_output_tile_thread_map.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"


namespace cutlass {


namespace epilogue {
namespace threadblock {

template <typename ThreadMap_,
          typename Layout_,
          typename Element_,
          int ElementsPerAccess = 1
          >
class PerChannelBiasPredicatedTileIterator;

template <typename ThreadMap_,
          typename Layout_,
          typename Element_,
          int ElementsPerAccess>
class PerChannelBiasPredicatedTileIterator {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = Layout_;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Shape::kRow == kElementsPerAccess,
                  "ThreadMap::Shape::kRow must equal to elements per access");
    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0,
                  "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");

    using Fragment =
            Array<Element,
                  ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup *
                          ThreadMap::Iterations::kCluster * kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;


    struct Params {
        LongIndex stride;

        LongIndex advance_row;

        LongIndex advance_group;

        LongIndex advance_cluster;

        LongIndex advance_tile;


        CUTLASS_HOST_DEVICE
        Status initialize(Index ) {
            stride = sizeof_bits<Element>::value / 8;

            advance_row = stride * ThreadMap::Shape::kRow;

            advance_group = stride * (ThreadMap::Shape::kGroup - 1) *
                            ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            advance_cluster = stride * ThreadMap::Count::kGroup *
                              ThreadMap::Shape::kGroup *
                              ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

            advance_tile = stride * ThreadMap::Shape::kGroup *
                           ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
                           ThreadMap::Shape::kTile;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& ) { initialize(0); }
    };

    struct Mask {};

private:

    Params params_;

    uint8_t* byte_pointer_;

    Index extent_row_;

    Index thread_start_row_;

    int state_[3];

private:

public:

    CUTLASS_DEVICE
    PerChannelBiasPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset_ = ThreadMap::initial_offset(thread_idx);
        Index channel_offset = thread_offset_.row() + threadblock_offset.row();

        extent_row_ = extent.row();
        thread_start_row_ = channel_offset;

        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        channel_offset * sizeof_bits<Element>::value / 8;

        state_[0] = state_[1] = state_[2] = 0;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[frag_row_idx],
                            (void*)&memory_pointer[row_offset /
                                                   kElementsPerAccess],
                            guard);
                }
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool guard =
                            ((row_offset + thread_start_row_) < extent_row_);
                    cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[frag_row_idx],
                            (void*)&memory_pointer[row_offset /
                                                   kElementsPerAccess],
                            guard);
                }
            }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    PerChannelBiasPredicatedTileIterator& operator++() {
        ++state_[0];
        byte_pointer_ += params_.advance_row;
        thread_start_row_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {
            state_[0] = 0;
            ++state_[1];
            byte_pointer_ += params_.advance_group;

            thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
                                 ThreadMap::Shape::kRow *
                                 ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];
                byte_pointer_ += params_.advance_cluster;

                thread_start_row_ +=
                        ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
                        ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    byte_pointer_ += params_.advance_tile;
                }
            }
        }

        return *this;
    }

    CUTLASS_DEVICE void clear_mask() {}

    CUTLASS_DEVICE void enable_mask() {}

    CUTLASS_DEVICE void get_mask(Mask& ) {}

    CUTLASS_DEVICE void set_mask(Mask const& ) {}

    CUTLASS_DEVICE bool valid() { return true; }
};


template <typename ThreadMap_,
          typename Element_,
          int ElementsPerAccess>
class PerChannelBiasPredicatedTileIterator<ThreadMap_, layout::TensorNCxHWx<32>,
                                           Element_, ElementsPerAccess> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::TensorNCxHWx<32>;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0,
                  "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");

    using Fragment =
            Array<Element, ThreadMap::Iterations::kRow * kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;


    struct Params {
        LongIndex stride;


        CUTLASS_HOST_DEVICE
        Status initialize(Index ) {
            stride = sizeof_bits<Element>::value / 8;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& ) { initialize(0); }
    };

    struct Mask {};

private:

    Params params_;

    uint8_t* byte_pointer_;

    Index extent_row_;

    Index thread_start_row_;

    Index state_;

private:

public:

    CUTLASS_DEVICE
    PerChannelBiasPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset_ = ThreadMap::initial_offset(thread_idx);
        Index channel_offset = thread_offset_.row() + threadblock_offset.row();

        extent_row_ = extent.row();
        thread_start_row_ = channel_offset;

        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        channel_offset * sizeof_bits<Element>::value / 8;

        state_ = 0;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            int row_offset = row * ThreadMap::Delta::kRow;
            bool guard = ((row_offset + thread_start_row_) < extent_row_);

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[row],
                    (void*)&memory_pointer[row_offset / kElementsPerAccess],
                    guard);
        }

        if (state_ == 0)
            state_++;
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            int row_offset = row * ThreadMap::Delta::kRow;

            bool guard = ((row_offset + thread_start_row_) < extent_row_);
            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                    frag_ptr[row],
                    (void*)&memory_pointer[row_offset / kElementsPerAccess],
                    guard);
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    PerChannelBiasPredicatedTileIterator& operator++() { return *this; }

    CUTLASS_DEVICE void clear_mask() {}

    CUTLASS_DEVICE void enable_mask() {}

    CUTLASS_DEVICE void get_mask(Mask& ) {}

    CUTLASS_DEVICE void set_mask(Mask const& ) {}

    CUTLASS_DEVICE bool valid() { return state_ == 0; }
};


template <typename ThreadMap_,
          typename Element_,
          int ElementsPerAccess>
class PerChannelBiasPredicatedTileIterator<ThreadMap_, layout::TensorNHWC,
                                           Element_, ElementsPerAccess> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::TensorNHWC;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::ColumnMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0,
                  "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0,
                  "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");

    using Fragment =
            Array<Element, ThreadMap::Iterations::kColumn * kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;


    struct Params {
        LongIndex stride;


        CUTLASS_HOST_DEVICE
        Status initialize(Index ) {
            stride = sizeof_bits<Element>::value / 8;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& ) { initialize(0); }
    };

    struct Mask {};

private:

    Params params_;

    uint8_t* byte_pointer_;

    Index extent_row_;

    Index thread_start_row_;

    Index state_;

private:

public:

    CUTLASS_DEVICE
    PerChannelBiasPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset_ = ThreadMap::initial_offset(thread_idx);
        Index channel_offset = thread_offset_.row() + threadblock_offset.row();

        extent_row_ = extent.row();
        thread_start_row_ = channel_offset;

        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        channel_offset * sizeof_bits<Element>::value / 8;

        state_ = 0;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int column = 0; column < ThreadMap::Iterations::kColumn;
             ++column) {
            int row_offset = column * ThreadMap::Delta::kColumn;
            bool guard = ((row_offset + thread_start_row_) < extent_row_);

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[column],
                    (void*)&memory_pointer[row_offset / kElementsPerAccess],
                    guard);
        }

        if (state_ == 0)
            state_++;
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);
        CUTLASS_PRAGMA_UNROLL
        for (int column = 0; column < ThreadMap::Iterations::kColumn;
             ++column) {
            int row_offset = column * ThreadMap::Delta::kColumn;

            bool guard = ((row_offset + thread_start_row_) < extent_row_);
            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                    frag_ptr[column],
                    (void*)&memory_pointer[row_offset / kElementsPerAccess],
                    guard);
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    PerChannelBiasPredicatedTileIterator& operator++() { return *this; }

    CUTLASS_DEVICE void clear_mask() {}

    CUTLASS_DEVICE void enable_mask() {}

    CUTLASS_DEVICE void get_mask(Mask& ) {}

    CUTLASS_DEVICE void set_mask(Mask const& ) {}

    CUTLASS_DEVICE bool valid() { return state_ == 0; }
};


template <typename ThreadMap_,
          typename Layout_,
          typename Element_,
          int ElementsPerAccess = 1,
          bool IsRow = true>
class PerChannelBiasPredicatedTileIteratorTensorOp;

template <typename ThreadMap_,
          typename Layout_,
          typename Element_,
          int ElementsPerAccess
          >
class PerChannelBiasPredicatedTileIteratorTensorOp<
        ThreadMap_, Layout_, Element_, ElementsPerAccess, true> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = Layout_;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kCount;

    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");
    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");

    using Fragment =
            Array<Element, ThreadMap::Iterations::kRow * kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;


    struct Params {
        LongIndex stride;

        LongIndex advance_row;


        CUTLASS_HOST_DEVICE
        Status initialize(Index ) {
            stride = sizeof_bits<Element>::value / 8;

            advance_row = stride * ThreadMap::Shape::kRow;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& ) { initialize(0); }
    };

    struct Mask {};

private:

    Params params_;

    uint8_t* byte_pointer_;

    Index extent_row_;

    Index thread_start_row_;

    int state_;

private:

public:

    CUTLASS_DEVICE
    PerChannelBiasPredicatedTileIteratorTensorOp(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset_ = ThreadMap::initial_offset(thread_idx);
        Index channel_offset = thread_offset_.row() + threadblock_offset.row();

        extent_row_ = extent.row();
        thread_start_row_ = channel_offset;

        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        channel_offset * sizeof_bits<Element>::value / 8;

        state_ = 0;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            int row_offset = row * ThreadMap::Delta::kRow;
            bool guard = (row_offset + thread_start_row_) < extent_row_;

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[row],
                    (void*)&memory_pointer[row_offset / kElementsPerAccess],
                    guard);
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            int row_offset = row * ThreadMap::Delta::kRow;
            bool guard = (row_offset + thread_start_row_) < extent_row_;

            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                    frag_ptr[row],
                    (void*)&memory_pointer[row_offset / kElementsPerAccess],
                    guard);
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    PerChannelBiasPredicatedTileIteratorTensorOp& operator++() {
        ++state_;
        if (state_ == ThreadMap::Count::kColumn) {
            state_ = 0;
            byte_pointer_ += params_.advance_row;
            thread_start_row_ += ThreadMap::Shape::kRow;
        }

        return *this;
    }

    CUTLASS_DEVICE void clear_mask() {}

    CUTLASS_DEVICE void enable_mask() {}

    CUTLASS_DEVICE void get_mask(Mask& ) {}

    CUTLASS_DEVICE void set_mask(Mask const& ) {}

    CUTLASS_DEVICE bool valid() { return state_ == 0; }
};

template <typename ThreadMap_,
          typename Layout_,
          typename Element_,
          int ElementsPerAccess
          >
class PerChannelBiasPredicatedTileIteratorTensorOp<
        ThreadMap_, Layout_, Element_, ElementsPerAccess, false> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = Layout_;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kCount;

    static_assert(ThreadMap::Iterations::kColumn > 0,
                  "ThreadMap::Iterations::kColumn must be > 0");
    static_assert(ThreadMap::Iterations::kRow > 0,
                  "ThreadMap::Iterations::kRow must be > 0");

    using Fragment =
            Array<Element, ThreadMap::Iterations::kColumn * kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;


    struct Params {
        LongIndex stride;

        LongIndex advance_col;


        CUTLASS_HOST_DEVICE
        Status initialize(Index ) {
            stride = sizeof_bits<Element>::value / 8;

            advance_col = stride * ThreadMap::Shape::kColumn;

            return Status::kSuccess;
        }

        CUTLASS_HOST_DEVICE
        Params() { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& ) { initialize(0); }
    };

    struct Mask {};

private:

    Params params_;

    uint8_t* byte_pointer_;

    Index extent_col_;

    Index thread_start_col_;

    int state_;

private:

public:

    CUTLASS_DEVICE
    PerChannelBiasPredicatedTileIteratorTensorOp(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset_ = ThreadMap::initial_offset(thread_idx);
        Index channel_offset =
                thread_offset_.column() + threadblock_offset.column();

        extent_col_ = extent.column();
        thread_start_col_ = channel_offset;

        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        channel_offset * sizeof_bits<Element>::value / 8;

        state_ = 0;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int col = 0; col < ThreadMap::Iterations::kColumn; ++col) {
            int col_offset = col * ThreadMap::Delta::kColumn;
            bool guard = (col_offset + thread_start_col_) < extent_col_;

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[col],
                    (void*)&memory_pointer[col_offset / kElementsPerAccess],
                    guard);
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int col = 0; col < ThreadMap::Iterations::kColumn; ++col) {
            int col_offset = col * ThreadMap::Delta::kColumn;
            bool guard = (col_offset + thread_start_col_) < extent_col_;

            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                    frag_ptr[col],
                    (void*)&memory_pointer[col_offset / kElementsPerAccess],
                    guard);
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    PerChannelBiasPredicatedTileIteratorTensorOp& operator++() {
        ++state_;
        if (state_ == ThreadMap::Count::kRow) {
            state_ = 0;
            byte_pointer_ += params_.advance_col;
            thread_start_col_ += ThreadMap::Shape::kColumn;
        }

        return *this;
    }

    CUTLASS_DEVICE void clear_mask() {}

    CUTLASS_DEVICE void enable_mask() {}

    CUTLASS_DEVICE void get_mask(Mask& ) {}

    CUTLASS_DEVICE void set_mask(Mask const& ) {}

    CUTLASS_DEVICE bool valid() { return state_ == 0; }
};


template <typename Layout_,
          typename Element_,
          int ElementsPerAccess = 1
          >
class Dwconv2dBiasTileIterator {
public:
    using Element = Element_;

    using Layout = Layout_;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    static int const kElementsPerAccess = ElementsPerAccess;
    static_assert(kElementsPerAccess == 1,
                  "Elements per access of dwconv2d bias iterator must be 1");

    using Fragment = Array<Element, kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;


    struct Params {

        Layout layout_;


        CUTLASS_HOST_DEVICE
        Status initialize(Index ) { return Status::kSuccess; }

        CUTLASS_HOST_DEVICE
        Params() { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout) : layout_(layout) { initialize(0); }
    };

    struct Mask {};

private:

    Params params_;

    uint8_t* byte_pointer_;

private:

public:

    CUTLASS_DEVICE
    Dwconv2dBiasTileIterator(Params const& params, Element* pointer,
                             LogicalCoord , int ,
                             LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer);
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(Fragment& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                frag_ptr[0], reinterpret_cast<void*>(memory_pointer), true);
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
        AccessType* memory_pointer =
                reinterpret_cast<AccessType*>(byte_pointer + byte_offset);
        cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                frag_ptr[0], reinterpret_cast<void*>(memory_pointer), true);
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    Dwconv2dBiasTileIterator& operator++() { return *this; }

    CUTLASS_DEVICE void clear_mask() {}

    CUTLASS_DEVICE void enable_mask() {}

    CUTLASS_DEVICE void get_mask(Mask& ) {}

    CUTLASS_DEVICE void set_mask(Mask const& ) {}

    CUTLASS_DEVICE bool valid() { return true; }

    CUTLASS_DEVICE
    Dwconv2dBiasTileIterator& add_coord_offset(
            TensorCoord const& coord_offset) {
        add_pointer_offset(params_.layout_(coord_offset));
        return *this;
    }
};

}
}
}

