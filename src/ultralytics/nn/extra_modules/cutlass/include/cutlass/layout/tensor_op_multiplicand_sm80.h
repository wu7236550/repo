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
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"


namespace cutlass {
namespace layout {


struct TensorOpMultiplicandCongruous64b {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kElementSize = 64;
    static int const kElementsPerAccess = 1;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCongruous64b(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCongruous64b(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicandCongruous64b packed(TensorCoord const& extent) {
        return TensorOpMultiplicandCongruous64b(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        int tc = coord.contiguous() / 16;
        int ts = coord.strided() / 4;

        int c = coord.contiguous() % 16;
        int s = coord.strided() % 4;

        int bank = ((((c & 1) * 4 + (c & 6) / 2)) ^ (s & 1)) * 2 + (c / 8);
        int row = (c & 6) / 2;

        bank ^= ((s & 2) * 2);

        LongIndex offset = tc * 16 + bank + (ts * 4 + row) * stride_[0];

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

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex offset) const { return TensorCoord(); }
};


struct ColumnMajorTensorOpMultiplicandCongruous64b {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCongruous64b;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCongruous64b(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCongruous64b(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorTensorOpMultiplicandCongruous64b packed(
            TensorCoord const& extent) {
        return ColumnMajorTensorOpMultiplicandCongruous64b(extent.row());
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


struct RowMajorTensorOpMultiplicandCongruous64b {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCongruous64b;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCongruous64b(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCongruous64b(Stride stride) : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorTensorOpMultiplicandCongruous64b packed(
            TensorCoord const& extent) {
        return RowMajorTensorOpMultiplicandCongruous64b(extent.column());
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


struct TensorOpMultiplicand64bCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kElementSize = 64;
    static int const kElementsPerAccess = 1;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicand64bCrosswise(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicand64bCrosswise(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicand64bCrosswise packed(TensorCoord const& extent) {
        return TensorOpMultiplicand64bCrosswise(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        int tc = coord.contiguous() / 16;
        int ts = coord.strided() / 16;

        int c = coord.contiguous() % 16;
        int s = coord.strided() % 16;

        int k_group = c / 4;
        int access_s = s / 2;

        int row = access_s % 4;
        int bank = ((k_group & 2) << 2) ^
                   ((s % 2) << 3) + (c % 4) * 2 + (access_s / 4) ^
                   (k_group & 1);

        int smem_row = (k_group * 4 + row) + tc * 16;
        int smem_col = ts * 16 + bank;

        LongIndex offset = smem_row * stride_[0] + smem_col;

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


struct ColumnMajorTensorOpMultiplicand64bCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicand64bCrosswise;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicand64bCrosswise(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicand64bCrosswise(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorTensorOpMultiplicand64bCrosswise packed(
            TensorCoord const& extent) {
        return ColumnMajorTensorOpMultiplicand64bCrosswise(extent.column());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(PitchLinearCoord(coord.row(), coord.column()));
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


struct RowMajorTensorOpMultiplicand64bCrosswise {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicand64bCrosswise;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicand64bCrosswise(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicand64bCrosswise(Stride stride) : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorTensorOpMultiplicand64bCrosswise packed(
            TensorCoord const& extent) {
        return RowMajorTensorOpMultiplicand64bCrosswise(extent.row());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(PitchLinearCoord(coord.column(), coord.row()));
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


struct TensorOpMultiplicandCongruous128b {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kElementSize = 128;
    static int const kElementsPerAccess = 1;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCongruous128b(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCongruous128b(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicandCongruous128b packed(TensorCoord const& extent) {
        return TensorOpMultiplicandCongruous128b(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        Index tc = coord.contiguous() / 8;
        Index ts = coord.strided() / 4;

        Index c = coord.contiguous() % 8;
        Index s = coord.strided() % 4;

        Index k_index = (c / 2);

        Index bank = (((c & 1) * 4) | (s ^ k_index));

        LongIndex offset = tc * 8 + bank + (ts * 4 + k_index) * stride_[0];

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

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex offset) const { return TensorCoord(); }
};


struct ColumnMajorTensorOpMultiplicandCongruous128b {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCongruous128b;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCongruous128b(Index ldm = 0)
            : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCongruous128b(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorTensorOpMultiplicandCongruous128b packed(
            TensorCoord const& extent) {
        return ColumnMajorTensorOpMultiplicandCongruous128b(extent.row());
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


struct RowMajorTensorOpMultiplicandCongruous128b {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCongruous128b;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCongruous128b(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCongruous128b(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorTensorOpMultiplicandCongruous128b packed(
            TensorCoord const& extent) {
        return RowMajorTensorOpMultiplicandCongruous128b(extent.column());
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


struct TensorOpMultiplicandCrosswise128x4 {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    static int const kElementSize = 128;
    static int const kElementsPerAccess = 1;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCrosswise128x4(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    TensorOpMultiplicandCrosswise128x4(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorOpMultiplicandCrosswise128x4 packed(
            TensorCoord const& extent) {
        return TensorOpMultiplicandCrosswise128x4(extent[0]);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        Index tc = coord.contiguous() / 8;
        Index ts = coord.strided() / 8;

        Index c = coord.contiguous() % 8;
        Index s = coord.strided() % 8;

        Index liq = c % 4;

        Index bank = liq + ((s & 1) * 4) ^ (c & 4);

        Index k_index = (c & 4) + (s / 4) * 2 + ((s & 2) / 2);

        LongIndex offset = (tc * 8 + k_index) * stride_[0] + ts * 8 + bank;

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


struct ColumnMajorTensorOpMultiplicandCrosswise128x4 {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCrosswise128x4;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCrosswise128x4(Index ldm = 0)
            : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorTensorOpMultiplicandCrosswise128x4(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorTensorOpMultiplicandCrosswise128x4 packed(
            TensorCoord const& extent) {
        return ColumnMajorTensorOpMultiplicandCrosswise128x4(extent.column());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(PitchLinearCoord(coord.row(), coord.column()));
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


struct RowMajorTensorOpMultiplicandCrosswise128x4 {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index, LongIndex>;


    using Base = TensorOpMultiplicandCrosswise128x4;

private:

    Base layout_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCrosswise128x4(Index ldm = 0) : layout_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorTensorOpMultiplicandCrosswise128x4(Stride stride)
            : layout_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorTensorOpMultiplicandCrosswise128x4 packed(
            TensorCoord const& extent) {
        return RowMajorTensorOpMultiplicandCrosswise128x4(extent.row());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return layout_(PitchLinearCoord(coord.column(), coord.row()));
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

