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
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/fast_math.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <int Column, int Row, int Group, int Cluster, int Tile>
struct OutputTileShape {
    static int const kColumn = Column;
    static int const kRow = Row;
    static int const kGroup = Group;
    static int const kCluster = Cluster;
    static int const kTile = Tile;

    static int const kCount = kColumn * kRow * kGroup * kCluster * kTile;
};


template <typename ThreadMap_, typename Shape_, typename Iterations_,
          typename Delta_, typename Count_>
struct OutputTileThreadMap {
    using ThreadMap = ThreadMap_;

    static int const kThreads = ThreadMap::kThreads;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    using Shape = Shape_;

    using Iterations = Iterations_;

    using Delta = Delta_;

    using Count = Count_;

    CUTLASS_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
        using Index = typename layout::PitchLinearCoord::Index;

        layout::PitchLinearCoord coord = ThreadMap::initial_offset(thread_idx);

        Index cluster = coord.strided() / (Shape::kGroup * Shape::kRow);
        Index cluster_residual =
                coord.strided() % (Shape::kGroup * Shape::kRow);

        Index group = cluster_residual / (Shape::kRow);
        Index row = cluster_residual % (Shape::kRow);

        return MatrixCoord{row + group * Shape::kRow * Count::kRow +
                                   cluster * Shape::kGroup * Count::kGroup *
                                           Shape::kRow * Count::kRow,
                           coord.contiguous()};
    }
};


namespace detail {

template <typename Shape, int WarpsRemaining, int ElementsPerAccess,
          int ElementSize, bool Is2dTile>
struct RowArrangement;

template <typename Shape, int WarpsRemaining, int ElementsPerAccess,
          int ElementSize>
struct RowArrangement<Shape, WarpsRemaining, ElementsPerAccess, ElementSize,
                      false> {
    static int const kWarpSize = 32;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;

    static int const kIterationsRow = 1;
    static int const kDeltaRow = 1;
    static int const kIterationsColumn =
            Shape::kColumn / kElementsPerAccess / kWarpSize;
    static int const kDeltaColumn = kWarpSize * kElementsPerAccess;

    static int const kAccessWidth = kWarpSize;
    static int const kAccessRows = 1;
    static int const kWarpPartitionsRow = 1;
    static int const kWarpPartitionsColumn = WarpsRemaining;
};

template <typename Shape, int WarpsRemaining, int ElementsPerAccess,
          int ElementSize>
struct RowArrangement<Shape, WarpsRemaining, ElementsPerAccess, ElementSize,
                      true> {
    static int const kMemoryAccessSize = 128;
    static int const kWarpSize = 32;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;

    struct Detail {
        static int const kShapeRow = Shape::kRow / WarpsRemaining;
        static int const kShapeWidth = Shape::kColumn / kElementsPerAccess;

        static int const kTargetMemoryAccessWidth =
                kMemoryAccessSize / (kElementsPerAccess * kElementSize / 8);

        static int const kTargetAccessRows =
                kWarpSize / kTargetMemoryAccessWidth;
    };

    static int const kAccessWidth =
            (Detail::kTargetAccessRows > Detail::kShapeRow
                     ? kWarpSize / Detail::kShapeRow
                     : const_min(Detail::kShapeWidth,
                                 const_min(kWarpSize,
                                           kMemoryAccessSize /
                                                   (kElementsPerAccess *
                                                    kElementSize / 8))));

    static int const kAccessRows =
            (Detail::kTargetAccessRows > Detail::kShapeRow
                     ? Detail::kShapeRow
                     : const_min(Shape::kRow, kWarpSize / kAccessWidth));

    static int const kIterationsRow = Detail::kShapeRow / kAccessRows;
    static int const kDeltaRow = kAccessRows;

    static int const kIterationsColumn = Detail::kShapeWidth / kAccessWidth;
    static int const kDeltaColumn = kAccessWidth * kElementsPerAccess;

    static_assert(kAccessWidth * kElementsPerAccess <= Shape::kColumn,
                  "Accessing too many elements per access");
    static_assert(kIterationsColumn > 0, "Iteration Count Column must be > 0");
    static_assert(kIterationsRow > 0, "Iteration Count Row must be > 0");

    static int const kWarpPartitionsRow = 1;
    static int const kWarpPartitionsColumn = 1;
};

}


template <typename Shape_, typename Count_, int Threads, int ElementsPerAccess,
          int ElementSize>
struct OutputTileOptimalThreadMap {
    using Shape = Shape_;
    using Count = Count_;

    static int const kWarpSize = 32;
    static int const kThreads = Threads;
    static int const kWarpCount = kThreads / kWarpSize;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;


    struct Detail {
        static int const kIterationsCluster =
                ((Shape::kCluster > kWarpCount) ? Shape::kCluster / kWarpCount
                                                : 1);

        static int const kDeltaCluster =
                ((Shape::kCluster > kWarpCount)
                         ? Shape::kRow * Count::kRow * Shape::kGroup *
                                   Count::kGroup * Shape::kCluster /
                                   kIterationsCluster
                         : 1);

        static int const kCompactedDeltaCluster =
                ((Shape::kCluster > kWarpCount)
                         ? Shape::kRow * Shape::kGroup * Shape::kCluster /
                                   kIterationsCluster
                         : 1);

        static int const kWarpPartitionsCluster =
                ((Shape::kCluster > kWarpCount) ? kWarpCount
                                                : kWarpCount / Shape::kCluster);

        static int const kWarpsRemainingForGroups =
                ((Shape::kCluster > kWarpCount) ? 1
                                                : kWarpCount / Shape::kCluster);

        static int const kIterationsGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kGroup / kWarpsRemainingForGroups
                         : 1);

        static int const kDeltaGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kRow * Count::kRow * Shape::kGroup /
                                   kIterationsGroup
                         : 1);

        static int const kCompactedDeltaGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? Shape::kRow * Shape::kGroup / kIterationsGroup
                         : 1);

        static int const kWarpPartitionsGroup =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? 1
                         : kWarpsRemainingForGroups / Shape::kGroup);

        static int const kWarpsRemainingForRows =
                ((Shape::kGroup > kWarpsRemainingForGroups)
                         ? 1
                         : kWarpsRemainingForGroups / Shape::kGroup);

        using RowArrangement =
                detail::RowArrangement<Shape, kWarpsRemainingForRows,
                                       kElementsPerAccess, kElementSize,
                                       (Shape::kRow > kWarpsRemainingForRows)>;

        using WarpPartitions =
                OutputTileShape<RowArrangement::kWarpPartitionsColumn,
                                RowArrangement::kWarpPartitionsRow,
                                kWarpPartitionsGroup, kWarpPartitionsCluster,
                                1>;

        static int const kAccessWidth = RowArrangement::kAccessWidth;
        static int const kAccessRows = RowArrangement::kAccessRows;
    };


    using Iterations =
            OutputTileShape<Detail::RowArrangement::kIterationsColumn,
                            Detail::RowArrangement::kIterationsRow,
                            Detail::kIterationsGroup,
                            Detail::kIterationsCluster, 1>;

    using Delta =
            OutputTileShape<Detail::RowArrangement::kDeltaColumn,
                            Detail::RowArrangement::kDeltaRow,
                            Detail::kDeltaGroup, Detail::kDeltaCluster, 1>;

    CUTLASS_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
        int warp_idx = thread_idx / kWarpSize;
        int lane_idx = thread_idx % kWarpSize;

        int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
        int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

        int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
        int residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

        int row_idx = residual_group / Detail::WarpPartitions::kRow;
        int col_idx = residual_group % Detail::WarpPartitions::kRow;

        int lane_row_offset = lane_idx / Detail::kAccessWidth;
        int lane_col_offset = lane_idx % Detail::kAccessWidth;

        int cluster_offset = cluster_idx * Shape::kRow * Count::kRow *
                             Shape::kGroup * Count::kGroup;
        int group_offset = group_idx * Shape::kRow * Count::kRow;
        int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
        int column_offset = col_idx * Iterations::kColumn *
                            Detail::kAccessWidth * kElementsPerAccess;

        return MatrixCoord(
                cluster_offset + group_offset + row_offset + lane_row_offset,
                (column_offset + lane_col_offset) * kElementsPerAccess);
    }

    struct CompactedThreadMap {
        using Shape = Shape_;

        using Iterations =
                OutputTileShape<Detail::RowArrangement::kIterationsColumn,
                                Detail::RowArrangement::kIterationsRow,
                                Detail::kIterationsGroup,
                                Detail::kIterationsCluster, 1>;

        using Delta = OutputTileShape<Detail::RowArrangement::kDeltaColumn,
                                      Detail::RowArrangement::kDeltaRow,
                                      Detail::kCompactedDeltaGroup,
                                      Detail::kCompactedDeltaCluster, 1>;

        static int const kElementsPerAccess = ElementsPerAccess;

        static int const kThreads = Threads;

        CUTLASS_HOST_DEVICE
        static MatrixCoord initial_offset(int thread_idx) {
            int warp_idx = thread_idx / kWarpSize;
            int lane_idx = thread_idx % kWarpSize;

            int cluster_idx = warp_idx / Detail::WarpPartitions::kCluster;
            int residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

            int group_idx = residual_cluster / Detail::WarpPartitions::kGroup;
            int residual_group =
                    residual_cluster % Detail::WarpPartitions::kGroup;

            int row_idx = residual_group / Detail::WarpPartitions::kRow;
            int col_idx = residual_group % Detail::WarpPartitions::kRow;

            int lane_row_offset = lane_idx / Detail::kAccessWidth;
            int lane_col_offset = lane_idx % Detail::kAccessWidth;

            int cluster_offset = cluster_idx * Shape::kRow * Shape::kGroup;
            int group_offset = group_idx * Shape::kRow;
            int row_offset = row_idx * Iterations::kRow * Detail::kAccessRows;
            int column_offset = col_idx * Iterations::kColumn *
                                Detail::kAccessWidth * kElementsPerAccess;

            MatrixCoord coord(
                    cluster_offset + group_offset + row_offset +
                            lane_row_offset,
                    (column_offset + lane_col_offset) * kElementsPerAccess);

            return coord;
        }
    };
};


template <typename WarpCount_, typename Iterations_, int Threads,
          int ElementsPerAccess, int ElementSize>
struct InterleavedOutputTileThreadMap {
    using WarpCount = WarpCount_;

    static int const kWarpSize = 32;
    static int const kThreads = Threads;
    static int const kWarpCount = kThreads / kWarpSize;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;


    struct Detail {};


    using Iterations = Iterations_;

    using Delta = layout::PitchLinearShape<kWarpSize * kElementsPerAccess, 1>;

    CUTLASS_HOST_DEVICE
    static layout::PitchLinearCoord initial_offset(int thread_idx) {
        int warp_idx = thread_idx / kWarpSize;
        int lane_idx = thread_idx % kWarpSize;

        layout::PitchLinearCoord warp_footprint{
                Delta::kContiguous * Iterations::kContiguous,
                Delta::kStrided * Iterations::kStrided};

        layout::PitchLinearCoord warp_offset{warp_idx % WarpCount::kContiguous,
                                             warp_idx / WarpCount::kContiguous};

        layout::PitchLinearCoord thread_offset_in_warp{
                lane_idx * kElementsPerAccess, 0};

        layout::PitchLinearCoord thread_offset_in_threadblock_tile =
                warp_footprint * warp_offset + thread_offset_in_warp;

        return thread_offset_in_threadblock_tile;
    }
};


template <typename WarpCount_, typename Iterations_, int Threads,
          int ElementsPerAccess, int ElementSize>
struct InterleavedConvOutputTileThreadMap {
    using WarpCount = WarpCount_;

    static int const kWarpSize = 32;
    static int const kThreads = Threads;
    static int const kWarpCount = kThreads / kWarpSize;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kElementSize = ElementSize;


    struct Detail {};


    using Iterations = Iterations_;

    using Delta = MatrixShape<kWarpSize / 4, 4 * kElementsPerAccess>;

    CUTLASS_HOST_DEVICE
    static MatrixCoord initial_offset(int thread_idx) {
        int warp_idx = thread_idx / kWarpSize;
        int lane_idx = thread_idx % kWarpSize;

        MatrixCoord warp_footprint{
                Delta::kRow * Iterations::kRow,
                Delta::kColumn * Iterations::kColumn,
        };

        MatrixCoord warp_offset{warp_idx % WarpCount::kRow,
                                warp_idx / WarpCount::kRow};

        MatrixCoord thread_offset_in_warp{lane_idx / 4,
                                          (lane_idx % 4) * kElementsPerAccess};

        MatrixCoord thread_offset_in_threadblock_tile =
                warp_footprint * warp_offset + thread_offset_in_warp;

        return thread_offset_in_threadblock_tile;
    }
};


}
}
}
