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
 * \file
 * include/cutlass/epilogue/threadblock/dwconv2d_predicated_tile_iterator.h
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
#include "cutlass/conv/conv2d_problem_size.h"
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
          typename Element_
          >
class Dwconv2dPredicatedTileIterator;


template <typename ThreadMap_, typename Element_>
class Dwconv2dPredicatedTileIterator<ThreadMap_, layout::TensorNCHW, Element_> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::TensorNCHW;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using ConvProblemSize = typename conv::Conv2dProblemSize;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
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

    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;


    struct Params {

        LongIndex stride;

        LongIndex increment_row;
        LongIndex increment_group;
        LongIndex
                increment_cluster;

        LongIndex advance_row;
        LongIndex advance_group;
        LongIndex advance_cluster;
        LongIndex advance_tile;

        Layout layout_;


        CUTLASS_HOST_DEVICE
        Status initialize(Index stride_) {
            stride = LongIndex(stride_);

            increment_row = stride * ThreadMap::Delta::kRow;

            increment_group = stride * ThreadMap::Delta::kGroup -
                              stride * ThreadMap::Delta::kRow *
                                      (ThreadMap::Iterations::kRow - 1);

            increment_cluster = stride * ThreadMap::Delta::kCluster -
                                stride * ThreadMap::Delta::kGroup *
                                        (ThreadMap::Iterations::kGroup - 1) -
                                stride * ThreadMap::Delta::kRow *
                                        (ThreadMap::Iterations::kRow - 1);

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
        Params() : layout_(Layout()) { initialize(0); }

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, conv::Operator conv_operator,
               ConvProblemSize const& problem_size)
                : layout_(layout) {
            initialize(layout.stride()[2] * sizeof_bits<Element>::value / 8);
        }
    };

    struct Mask {
        static int const kCount = ThreadMap::Iterations::kColumn;

        bool predicates[kCount];

        CUTLASS_HOST_DEVICE
        Mask() { enable(); }

        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = false;
            }
        }

        CUTLASS_DEVICE void enable() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                predicates[i] = true;
            }
        }
    };

private:

    Params const& params_;

    uint8_t* byte_pointer_;

    Mask mask_;

    Index extent_row_;

    Index thread_start_row_;

    Index thread_start_col_;

    int state_[3];

public:

    CUTLASS_DEVICE
    Dwconv2dPredicatedTileIterator(
            Params const& params, Element* pointer, LogicalCoord extent,
            int thread_idx, LogicalCoord threadblock_offset = LogicalCoord())
            : params_(params) {
        MatrixCoord thread_offset =
                ThreadMap::initial_offset(thread_idx) + threadblock_offset;
        extent_row_ = extent.row();
        thread_start_row_ = thread_offset.row();
        thread_start_col_ = thread_offset.column();

        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
            mask_.predicates[c] =
                    ((thread_offset.column() + ThreadMap::Delta::kColumn * c) <
                     extent.column());
        }

        byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
                        thread_start_row_ * params_.stride;

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

                    bool row_guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];
                        int column_offset = thread_start_col_ +
                                            column * ThreadMap::Delta::kColumn;
                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        column_offset *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_load<AccessType,
                                                   sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)(memory_pointer), guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) {
        uint8_t* byte_pointer = byte_pointer_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

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

                    bool row_guard =
                            ((row_offset + thread_start_row_) < extent_row_);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        bool guard = row_guard && mask_.predicates[column];
                        int column_offset = thread_start_col_ +
                                            column * ThreadMap::Delta::kColumn;
                        AccessType* memory_pointer =
                                reinterpret_cast<AccessType*>(
                                        byte_pointer + byte_offset +
                                        column_offset *
                                                sizeof_bits<Element>::value /
                                                8);

                        cutlass::arch::global_store<AccessType,
                                                    sizeof(AccessType)>(
                                frag_ptr[frag_row_idx * ThreadMap::Iterations::
                                                                kColumn +
                                         column],
                                (void*)(memory_pointer), guard);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        byte_pointer += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    byte_pointer += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                byte_pointer += params_.increment_cluster;
            }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag) { store_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    Dwconv2dPredicatedTileIterator& operator++() {
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

    CUTLASS_DEVICE void clear_mask() { mask_.clear(); }

    CUTLASS_DEVICE void enable_mask() { mask_.enable(); }

    CUTLASS_DEVICE void get_mask(Mask& mask) { return mask_; }

    CUTLASS_DEVICE void set_mask(Mask const& mask) { mask_ = mask; }

    CUTLASS_DEVICE
    Dwconv2dPredicatedTileIterator& add_coord_offset(
            TensorCoord const& coord_offset) {
        add_pointer_offset(params_.layout_(coord_offset));
        return *this;
    }
};


template <typename Shape_,
          typename Operator_,
          typename Element_,
          typename Layout_,
          typename Detail
          >
class Dwconv2dPredicatedAccessTileIterator {
public:
    using Shape = Shape_;
    using Operator = Operator_;
    using Element = Element_;
    using Layout = Layout_;

    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    using LogicalLayout = layout::RowMajor;

    using LogicalCoord = typename LogicalLayout::TensorCoord;

    using ConvProblemSize = typename conv::Conv2dProblemSize;

    static int const kElementsPerAccess = 1;

    using WarpCount = gemm::GemmShape<Shape::kM / Operator::Shape::kM,
                                      Shape::kN / Operator::Shape::kN,
                                      Shape::kK / Operator::Shape::kK>;
    static int const kThreads = WarpCount::kCount;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;

    using TileMap = conv::threadblock::TileMap<
            Layout, conv::threadblock::TileMapType::kRow2OHW_Col2IHW>;


    struct Params {
    public:
    private:
        friend Dwconv2dPredicatedAccessTileIterator;


        Layout layout_;

        TileMap tile_map_;

        Index fh_;
        Index fw_;

    public:

        CUTLASS_HOST_DEVICE
        Params() : layout_(Layout()), fh_(0), fw_(0) {}

        CUTLASS_HOST_DEVICE
        Params(Layout const& layout, ConvProblemSize const& problem_size)
                : layout_(layout),
                  tile_map_(problem_size.W, problem_size.Q,
                            problem_size.stride_h, problem_size.stride_w,
                            problem_size.pad_h, problem_size.pad_w),
                  fh_(problem_size.R),
                  fw_(problem_size.S) {}
    };

    using Mask = platform::none_type;

    using Pointer = Element*;

    using BytePointer = char*;

private:

    Params const& params_;

    BytePointer pointer_;

    LogicalCoord thread_origin_;

    LogicalCoord extent_;

private:

public:

    CUTLASS_DEVICE
    Dwconv2dPredicatedAccessTileIterator(
            Params const& params,
            Element* pointer, LogicalCoord extent,
            int thread_idx,
            int warp_idx,
            int lane_idx,
            LogicalCoord const& threadblock_offset = LogicalCoord())
            : params_(params) {
        int warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
        int warp_m = warp_mn % WarpCount::kM;
        int warp_n = warp_mn / WarpCount::kM;

        LogicalCoord warp_offset = LogicalCoord{warp_m * Operator::Shape::kM,
                                                warp_n * Operator::Shape::kN};

        LogicalCoord lane_offset = Detail::get_lane_offset(lane_idx);

        thread_origin_ = threadblock_offset + warp_offset + lane_offset;

        extent_ = extent - thread_origin_;

        pointer_ = reinterpret_cast<BytePointer>(pointer);
    }

    CUTLASS_DEVICE
    bool early_stop(LogicalCoord const& threadblock_offset) {
        auto filter_ranges = params_.tile_map_(
                make_Coord(threadblock_offset.row(),
                           threadblock_offset.row() + Shape::kM),
                make_Coord(threadblock_offset.column(),
                           threadblock_offset.column() + Shape::kN));
        return filter_ranges.at(0) >= params_.fh_ || filter_ranges.at(1) < 0;
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    }

    CUTLASS_HOST_DEVICE
    AccessType* get(TensorCoord const& coord = TensorCoord()) const {
        return reinterpret_cast<AccessType*>(
                pointer_ +
                params_.layout_(coord) * sizeof_bits<Element>::value / 8);
    }

    CUTLASS_HOST_DEVICE
    bool valid(TensorCoord& tensor_coord,
               LogicalCoord const& coord = LogicalCoord()) {
        bool guard = coord.row() < extent_.row() &&
                     coord.column() < extent_.column();
        auto coord_ = thread_origin_ + coord;
        auto filter = params_.tile_map_(coord_);
        tensor_coord = TensorCoord(0, filter.row(), filter.column(), 0);
        return guard && filter.row() >= 0 && filter.row() < params_.fh_ &&
               filter.column() >= 0 && filter.column() < params_.fw_;
    }

    CUTLASS_DEVICE
    Dwconv2dPredicatedAccessTileIterator& add_coord_offset(
            TensorCoord const& coord_offset) {
        add_pointer_offset(params_.layout_(coord_offset));
        return *this;
    }
};


}
}
}

