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


namespace cutlass {
namespace epilogue {
namespace threadblock {


struct OutputTileShapeDesc {
    int column;
    int row;
    int group;
    int cluster;
    int tile;


    CUTLASS_HOST_DEVICE
    OutputTileShapeDesc() : column(0), row(0), group(0), cluster(0), tile(0) {}

    CUTLASS_HOST_DEVICE
    OutputTileShapeDesc(int column_, int row_, int group_, int cluster_,
                        int tile_)
            : column(column_),
              row(row_),
              group(group_),
              cluster(cluster_),
              tile(tile_) {}

    CUTLASS_HOST_DEVICE
    int count() const { return column * row * group * cluster * tile; }
};

template <typename Shape>
CUTLASS_HOST_DEVICE OutputTileShapeDesc make_OutputTileShapeDesc() {
    return OutputTileShapeDesc(Shape::kColumn, Shape::kRow, Shape::kGroup,
                               Shape::kCluster, Shape::kTile);
}


struct OutputTileThreadMapDesc {
    int threads;
    int elements_per_access;
    OutputTileShapeDesc shape;
    OutputTileShapeDesc iterations;
    OutputTileShapeDesc delta;
    OutputTileShapeDesc count;


    CUTLASS_HOST_DEVICE
    OutputTileThreadMapDesc() {}

    CUTLASS_HOST_DEVICE
    OutputTileThreadMapDesc(int threads_, int elements_per_access_,
                            OutputTileShapeDesc shape_,
                            OutputTileShapeDesc iterations_,
                            OutputTileShapeDesc delta_,
                            OutputTileShapeDesc count_)
            : threads(threads_),
              elements_per_access(elements_per_access_),
              shape(shape_),
              iterations(iterations_),
              delta(delta_),
              count(count_) {}
};

template <typename ThreadMap>
CUTLASS_HOST_DEVICE OutputTileThreadMapDesc make_OutputTileThreadMapDesc() {
    return OutputTileThreadMapDesc(
            ThreadMap::kThreads, ThreadMap::kElementsPerAccess,
            make_OutputTileShapeDesc<typename ThreadMap::Shape>(),
            make_OutputTileShapeDesc<typename ThreadMap::Iterations>(),
            make_OutputTileShapeDesc<typename ThreadMap::Delta>(),
            make_OutputTileShapeDesc<typename ThreadMap::Count>());
}



struct PredicatedTileIteratorParams {
    using Index = int32_t;
    using LongIndex = int64_t;


    LongIndex stride;

    LongIndex increment_row;
    LongIndex increment_group;
    LongIndex increment_cluster;

    LongIndex
            advance_row;
    LongIndex advance_group;
    LongIndex advance_cluster;
    LongIndex advance_tile;


    CUTLASS_HOST_DEVICE
    Status initialize(Index stride_, OutputTileThreadMapDesc thread_map) {
        stride = LongIndex(stride_);

        increment_row = stride * thread_map.delta.row;

        increment_group =
                stride * thread_map.delta.group -
                stride * thread_map.delta.row * (thread_map.iterations.row - 1);

        increment_cluster =
                stride * thread_map.delta.cluster -
                stride * thread_map.delta.group *
                        (thread_map.iterations.group - 1) -
                stride * thread_map.delta.row * (thread_map.iterations.row - 1);

        advance_row = stride * thread_map.shape.row;

        advance_group = stride * (thread_map.shape.group - 1) *
                        thread_map.shape.row * thread_map.count.row;

        advance_cluster = stride * thread_map.count.group *
                          thread_map.shape.group * thread_map.count.row *
                          thread_map.shape.row;

        advance_tile = stride * thread_map.shape.group * thread_map.shape.row *
                       thread_map.shape.cluster * thread_map.shape.tile;

        return Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    PredicatedTileIteratorParams() { initialize(0, OutputTileThreadMapDesc()); }

    CUTLASS_HOST_DEVICE
    PredicatedTileIteratorParams(Index stride,
                                 OutputTileThreadMapDesc thread_map) {
        initialize(stride, thread_map);
    }
};


}
}
}

