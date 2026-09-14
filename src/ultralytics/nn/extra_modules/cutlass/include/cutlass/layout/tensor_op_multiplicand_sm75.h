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
#include "cutlass/coord.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/layout/pitch_linear.h"


namespace cutlass {
namespace layout {


template <int ElementSize, int Crosswise>
struct TensorOpMultiplicand {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kAccessSize = 128;

    static int const kElementSize = ElementSize;
    static int const kElementsPerAccess = kAccessSize / kElementSize;
    static int const kCrosswise = Crosswise;

    static int const kTileShapeContiguous = 128 / (kAccessSize / 8);

    static int const kFactor =
            kTileShapeContiguous * kElementsPerAccess / kCrosswise;

    static_assert(
            (kFactor > 0),
            "kCrosswise should be no large than one shared memory cache line.");

    static int const kTileShapeStride =
            ((kTileShapeContiguous / kFactor) > (32 / kTileShapeContiguous))
                    ? (kTileShapeContiguous / kFactor)
                    : (32 / kTileShapeContiguous);

    using TileShape = PitchLinearShape<kTileShapeContiguous, kTileShapeStride>;

    using PartitionShape = PitchLinearShape<4, 4>;

    using PartitionCount =
            PitchLinearShape<TileShape::kContiguous /
                                     PartitionShape::kContiguous,
                             TileShape::kStrided / PartitionShape::kStrided>;

    using AccessCount = PitchLinearShape<PartitionShape::kContiguous,
                                         PartitionShape::kStrided>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicand(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicand(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicand packed(TensorCoord const& extent) {
        return TensorOpMultiplicand(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {

        int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
        int vec_strided_idx = coord.strided() / kFactor;

        int tile_contiguous_idx =
                vec_contiguous_idx / (TileShape::kContiguous / kFactor);

        int tile_contiguous_residual =
                vec_contiguous_idx % (TileShape::kContiguous / kFactor) +
                ((coord.strided() % kFactor) *
                 (TileShape::kContiguous / kFactor));
        int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

        int partition_contiguous_idx =
                tile_contiguous_residual / PartitionShape::kContiguous;
        int partition_strided_idx =
                tile_strided_residual / PartitionShape::kStrided;

        int partition_contiguous_residual =
                tile_contiguous_residual % PartitionShape::kContiguous;
        int partition_strided_residual =
                tile_strided_residual % PartitionShape::kStrided;


        int permuted_vec_contiguous_within_partition =
                partition_contiguous_residual ^
                (partition_strided_residual % 4);

        int permuted_partition_contiguous_within_tile =
                partition_contiguous_idx ^ (partition_strided_idx % 2);


        int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous +
                                  permuted_partition_contiguous_within_tile *
                                          PartitionShape::kContiguous +
                                  permuted_vec_contiguous_within_partition) *
                                         kElementsPerAccess +
                                 (coord.contiguous() % kElementsPerAccess);

        int element_strided = vec_strided_idx;

        return element_contiguous + element_strided * stride_[0] * kFactor;
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return extent[1] * stride_[0];
    }
};


template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicand<ElementSize, Crosswise>;

    static int const kAccessSize = Base::kAccessSize;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;


    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicandCongruous packed(TensorCoord const& extent) {
        return TensorOpMultiplicandCongruous(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(coord);
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return coord;
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return layout_.capacity(extent);
    }
};


template <int Crosswise>
struct TensorOpMultiplicandCongruous<32, Crosswise> {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kAccessSize = 128;

    using TileShape = PitchLinearShape<8, 4>;

    using PartitionShape = PitchLinearShape<8, 4>;

    using PartitionCount =
            PitchLinearShape<TileShape::kContiguous /
                                     PartitionShape::kContiguous,
                             TileShape::kStrided / PartitionShape::kStrided>;

    using AccessCount = PitchLinearShape<PartitionShape::kContiguous,
                                         PartitionShape::kStrided>;

    static int const kElementSize = 32;
    static int const kElementsPerAccess = kAccessSize / kElementSize;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCongruous(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCongruous(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicandCongruous packed(TensorCoord const& extent) {
        return TensorOpMultiplicandCongruous(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        int tc = coord.contiguous() / 32;
        int ts = coord.strided() / 4;

        int c = (coord.contiguous() % 32) / kElementsPerAccess;
        int s = coord.strided() % 4;

        LongIndex offset = (c ^ (2 * s)) * kElementsPerAccess + s * stride_[0] +
                           tc * 32 + ts * stride_[0] * 4 +
                           coord.contiguous() % 4;

        return offset;
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return extent[1] * stride_[0];
    }
};


template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCongruous<ElementSize, Crosswise>;

    static int const kAccessSize = Base::kAccessSize;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;


    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorTensorOpMultiplicandCongruous packed(
            TensorCoord const& extent) {
        return ColumnMajorTensorOpMultiplicandCongruous(extent.row());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(PitchLinearCoord(coord.row(), coord.column()));
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return MatrixCoord(coord.contiguous(), coord.strided());
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return layout_.capacity(
                PitchLinearCoord(extent.row(), extent.column()));
    }
};


template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCongruous<ElementSize, Crosswise>;

    static int const kAccessSize = Base::kAccessSize;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;


    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorTensorOpMultiplicandCongruous packed(
            TensorCoord const& extent) {
        return RowMajorTensorOpMultiplicandCongruous(extent.column());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(PitchLinearCoord(coord.column(), coord.row()));
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return MatrixCoord(coord.strided(), coord.contiguous());
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return layout_.capacity(
                PitchLinearCoord(extent.column(), extent.row()));
    }
};


template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicand<ElementSize, Crosswise>;

    static int const kAccessSize = Base::kAccessSize;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;


    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    static int const kCrosswise = Base::kCrosswise;
    static int const kFactor = Base::kFactor;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicandCrosswise packed(TensorCoord const& extent) {
        return TensorOpMultiplicandCrosswise(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(coord);
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return coord;
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return layout_.capacity(extent);
    }
};


template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCrosswise<ElementSize, Crosswise>;

    static int const kAccessSize = Base::kAccessSize;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;


    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorTensorOpMultiplicandCrosswise packed(
            TensorCoord const& extent) {
        return ColumnMajorTensorOpMultiplicandCrosswise(extent.row());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(PitchLinearCoord(coord.row(), coord.column()));
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return MatrixCoord(coord.contiguous(), coord.strided());
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return layout_.capacity(
                PitchLinearCoord(extent.row(), extent.column()));
    }
};


template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCrosswise<ElementSize, Crosswise>;

    static int const kAccessSize = Base::kAccessSize;
    using TileShape = typename Base::TileShape;
    using PartitionShape = typename Base::PartitionShape;


    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;
    using PartitionCount = typename Base::PartitionCount;
    using AccessCount = typename Base::AccessCount;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorTensorOpMultiplicandCrosswise packed(
            TensorCoord const& extent) {
        return RowMajorTensorOpMultiplicandCrosswise(extent.column());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(PitchLinearCoord(coord.column(), coord.row()));
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex offset) const {
        PitchLinearCoord coord = layout_.inverse(offset);
        return MatrixCoord(coord.strided(), coord.contiguous());
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return layout_.capacity(
                PitchLinearCoord(extent.column(), extent.row()));
    }
};


template <int ElementSize, int InterleavedK>
struct TensorOpMultiplicandColumnMajorInterleaved {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kAccessSize = 128;


    static int const kElementSize = ElementSize;
    static int const kElementsPerAccess = kAccessSize / kElementSize;

    static int const kInterleavedK = InterleavedK;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandColumnMajorInterleaved(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandColumnMajorInterleaved(Stride stride)
            : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicandColumnMajorInterleaved packed(
            TensorCoord const& extent) {
        return TensorOpMultiplicandColumnMajorInterleaved(extent[0] *
                                                          kInterleavedK);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        int const rows_per_smem_cache_line = 128 / kInterleavedK;

        int row_id = coord.strided() / rows_per_smem_cache_line;
        int col_id =
                (coord.strided() % rows_per_smem_cache_line) * kInterleavedK +
                coord.contiguous();

        int access_block_id = col_id >> 4;
        int swizzle_access_block_id = access_block_id ^ (row_id & 1);

        int swizzle_col_id = swizzle_access_block_id << 4;

        return row_id * 128 + swizzle_col_id;
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return (extent[1] / kInterleavedK) * stride_[0];
    }
};


template <int ElementSize, int InterleavedK>
struct TensorOpMultiplicandRowMajorInterleaved {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kAccessSize = 128;


    static int const kElementSize = ElementSize;
    static int const kElementsPerAccess = kAccessSize / kElementSize;

    static int const kInterleavedK = InterleavedK;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandRowMajorInterleaved(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandRowMajorInterleaved(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicandRowMajorInterleaved packed(
            TensorCoord const& extent) {
        return TensorOpMultiplicandRowMajorInterleaved(extent[1] *
                                                       kInterleavedK);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        int const rows_per_smem_cache_line = 128 / kInterleavedK;

        int row_id = coord.strided() / rows_per_smem_cache_line;
        int col_id =
                (coord.strided() % rows_per_smem_cache_line) * kInterleavedK +
                coord.contiguous();

        int access_block_id = col_id >> 4;
        int swizzle_access_block_id = access_block_id ^ (row_id & 1);

        int swizzle_col_id = swizzle_access_block_id << 4;

        return row_id * 128 + swizzle_col_id;
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return (extent[0] / kInterleavedK) * stride_[0];
    }
};


}
}

