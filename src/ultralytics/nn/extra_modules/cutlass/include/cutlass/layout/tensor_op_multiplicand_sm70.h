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
#include "cutlass/layout/pitch_linear.h"


namespace cutlass {
namespace layout {



template <int ElementSize>
struct VoltaTensorOpMultiplicandCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kAccessSize = 128;

    using TileShape = PitchLinearShape<8, 4>;

    using PartitionShape = PitchLinearShape<8, 2>;


    static int const kElementSize = ElementSize;
    static int const kElementsPerAccess = kAccessSize / kElementSize;

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
    VoltaTensorOpMultiplicandCongruous(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    VoltaTensorOpMultiplicandCongruous(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static VoltaTensorOpMultiplicandCongruous packed(
            TensorCoord const& extent) {
        return VoltaTensorOpMultiplicandCongruous(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
        int vec_strided_idx = coord.strided();

        int tile_contiguous_idx = vec_contiguous_idx / TileShape::kContiguous;
        int tile_strided_idx = vec_strided_idx / TileShape::kStrided;

        int tile_contiguous_residual =
                vec_contiguous_idx % TileShape::kContiguous;
        int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

        int permuted_strided_within_tile = (tile_contiguous_residual >> 1);
        int permuted_contiguous_within_tile =
                (tile_strided_residual ^ permuted_strided_within_tile) |
                ((tile_contiguous_residual & 1) << 2);
        int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous +
                                  permuted_contiguous_within_tile) *
                                         kElementsPerAccess +
                                 (coord.contiguous() % kElementsPerAccess);

        int element_strided = tile_strided_idx * TileShape::kStrided +
                              permuted_strided_within_tile;

        return element_contiguous + element_strided * stride_[0];
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


template <int ElementSize>
struct ColumnMajorVoltaTensorOpMultiplicandCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = VoltaTensorOpMultiplicandCongruous<ElementSize>;

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
    ColumnMajorVoltaTensorOpMultiplicandCongruous(Index ldm = 0)
            : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorVoltaTensorOpMultiplicandCongruous(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorVoltaTensorOpMultiplicandCongruous packed(
            TensorCoord const& extent) {
        return ColumnMajorVoltaTensorOpMultiplicandCongruous(extent.row());
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

template <int ElementSize>
struct RowMajorVoltaTensorOpMultiplicandCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = VoltaTensorOpMultiplicandCongruous<ElementSize>;

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
    RowMajorVoltaTensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorVoltaTensorOpMultiplicandCongruous(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorVoltaTensorOpMultiplicandCongruous packed(
            TensorCoord const& extent) {
        return RowMajorVoltaTensorOpMultiplicandCongruous(extent.column());
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

template <int ElementSize>
struct VoltaTensorOpMultiplicandBCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kAccessSize = 128;

    using TileShape = PitchLinearShape<8, 4>;

    using PartitionShape = PitchLinearShape<4, 4>;


    static int const kElementSize = ElementSize;
    static int const kElementsPerAccess = kAccessSize / kElementSize;

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
    VoltaTensorOpMultiplicandBCongruous(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    VoltaTensorOpMultiplicandBCongruous(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static VoltaTensorOpMultiplicandBCongruous packed(
            TensorCoord const& extent) {
        return VoltaTensorOpMultiplicandBCongruous(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
        int vec_strided_idx = coord.strided();

        int tile_contiguous_idx = vec_contiguous_idx / TileShape::kContiguous;
        int tile_strided_idx = vec_strided_idx / TileShape::kStrided;

        int tile_contiguous_residual =
                vec_contiguous_idx % TileShape::kContiguous;
        int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

        int permuted_strided_within_tile = (tile_contiguous_residual & 0x3);
        int permuted_contiguous_within_tile =
                (tile_strided_residual ^ permuted_strided_within_tile) |
                (tile_contiguous_residual & 0x4);

        int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous +
                                  permuted_contiguous_within_tile) *
                                         kElementsPerAccess +
                                 (coord.contiguous() % kElementsPerAccess);

        int element_strided = tile_strided_idx * TileShape::kStrided +
                              permuted_strided_within_tile;

        return element_contiguous + element_strided * stride_[0];
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


template <int ElementSize>
struct ColumnMajorVoltaTensorOpMultiplicandBCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = VoltaTensorOpMultiplicandBCongruous<ElementSize>;

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
    ColumnMajorVoltaTensorOpMultiplicandBCongruous(Index ldm = 0)
            : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorVoltaTensorOpMultiplicandBCongruous(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorVoltaTensorOpMultiplicandBCongruous packed(
            TensorCoord const& extent) {
        return ColumnMajorVoltaTensorOpMultiplicandBCongruous(extent.row());
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

template <int ElementSize>
struct RowMajorVoltaTensorOpMultiplicandBCongruous {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = VoltaTensorOpMultiplicandBCongruous<ElementSize>;

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
    RowMajorVoltaTensorOpMultiplicandBCongruous(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorVoltaTensorOpMultiplicandBCongruous(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorVoltaTensorOpMultiplicandBCongruous packed(
            TensorCoord const& extent) {
        return RowMajorVoltaTensorOpMultiplicandBCongruous(extent.column());
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

template <int ElementSize, int KBlock>
struct VoltaTensorOpMultiplicandCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kAccessSize = 64;


    static int const kElementSize = ElementSize;
    static int const kElementsPerAccess = kAccessSize / kElementSize;
    static int const kKBlock = KBlock;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    VoltaTensorOpMultiplicandCrosswise(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    VoltaTensorOpMultiplicandCrosswise(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static VoltaTensorOpMultiplicandCrosswise packed(
            TensorCoord const& extent) {
        return VoltaTensorOpMultiplicandCrosswise(extent[1]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
        int vec_strided_idx = coord.strided();


        int vec_strided_within_tile = vec_contiguous_idx & 0x7;
        int permuted_vec_contiguous =
                (vec_strided_idx & (~0xF)) + (vec_strided_idx & 0x3) * 4 +
                (((vec_strided_idx >> 2) ^ ((vec_strided_idx & 0x10) >> 3)) &
                 0x3);

        permuted_vec_contiguous ^= ((vec_strided_within_tile >> 1) & 0x3);

        int permuted_vec_strided = vec_contiguous_idx;


        int element_contiguous = permuted_vec_contiguous * kElementsPerAccess +
                                 (coord.contiguous() % kElementsPerAccess);

        return element_contiguous +
               permuted_vec_strided * (stride_[0] * kElementsPerAccess);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return extent[0] * stride_[0];
    }
};

template <int ElementSize, int KBlock>
struct ColumnMajorVoltaTensorOpMultiplicandCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = VoltaTensorOpMultiplicandCrosswise<ElementSize, KBlock>;

    static int const kAccessSize = Base::kAccessSize;


    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorVoltaTensorOpMultiplicandCrosswise(Index ldm = 0)
            : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorVoltaTensorOpMultiplicandCrosswise(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorVoltaTensorOpMultiplicandCrosswise packed(
            TensorCoord const& extent) {
        return ColumnMajorVoltaTensorOpMultiplicandCrosswise(extent.column());
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

template <int ElementSize, int KBlock>
struct RowMajorVoltaTensorOpMultiplicandCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = VoltaTensorOpMultiplicandCrosswise<ElementSize, KBlock>;

    static int const kAccessSize = Base::kAccessSize;


    static int const kElementSize = Base::kElementSize;
    static int const kElementsPerAccess = Base::kElementsPerAccess;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorVoltaTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorVoltaTensorOpMultiplicandCrosswise(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorVoltaTensorOpMultiplicandCrosswise packed(
            TensorCoord const& extent) {
        return RowMajorVoltaTensorOpMultiplicandCrosswise(extent.row());
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

}
}

