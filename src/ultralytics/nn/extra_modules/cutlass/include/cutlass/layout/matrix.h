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
#include "cutlass/matrix_coord.h"

namespace cutlass {
namespace layout {


class RowMajor {
public:
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    RowMajor(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajor(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajor packed(MatrixCoord const& extent) {
        return RowMajor(extent.column());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(MatrixCoord const& coord) const {
        return LongIndex(coord.row()) * LongIndex(stride_[0]) + coord.column();
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord inverse(LongIndex offset) const {
        return MatrixCoord(Index(offset / stride_[0]),
                           Index(offset % stride_[0]));
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Index stride(int idx) const { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int idx) { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(MatrixCoord const& extent) const {
        return LongIndex(extent.row()) * LongIndex(stride_[0]);
    }
};

class ColumnMajor {
public:
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajor(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajor(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajor packed(MatrixCoord const& extent) {
        return ColumnMajor(extent.row());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(MatrixCoord const& coord) const {
        return LongIndex(coord.column()) * LongIndex(stride_[0]) + coord.row();
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord inverse(LongIndex offset) const {
        return MatrixCoord(Index(offset % stride_[0]),
                           Index(offset / stride_[0]));
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Index stride(int idx) const { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int idx) { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(MatrixCoord const& extent) const {
        return LongIndex(extent.column()) * LongIndex(stride_[0]);
    }
};

template <int Interleave>
struct RowMajorInterleaved {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index>;

    static int const kInterleave = Interleave;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorInterleaved(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    RowMajorInterleaved(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static RowMajorInterleaved packed(MatrixCoord const& extent) {
        return RowMajorInterleaved(extent.column() * kInterleave);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(MatrixCoord const& coord) const {
        Index row_major = coord.row() / kInterleave;
        Index row_minor = coord.row() % kInterleave;
        return LongIndex(row_major) * LongIndex(stride_[0]) +
               LongIndex(coord.column()) * kInterleave + row_minor;
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord inverse(LongIndex offset) const {
        Index row_major = Index(offset / stride_[0]);
        Index residual = Index(offset % stride_[0]);

        Index column = residual / kInterleave;
        Index row_minor = residual % kInterleave;

        return MatrixCoord(row_major * kInterleave + row_minor, column);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Index stride(int idx) const { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int idx) { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(MatrixCoord const& extent) const {
        return (extent.row() + kInterleave - 1) / kInterleave * stride_[0];
    }
};

template <int Interleave>
struct ColumnMajorInterleaved {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index>;

    static int const kInterleave = Interleave;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorInterleaved(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    ColumnMajorInterleaved(Stride stride) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorInterleaved packed(MatrixCoord const& extent) {
        return ColumnMajorInterleaved(extent.row() * kInterleave);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(MatrixCoord const& coord) const {
        Index column_major = coord.column() / kInterleave;
        Index column_minor = coord.column() % kInterleave;
        return LongIndex(column_major) * LongIndex(stride_[0]) +
               LongIndex(coord.row()) * kInterleave + column_minor;
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord inverse(LongIndex offset) const {
        Index column_major = Index(offset / stride_[0]);
        Index residual = Index(offset % stride_[0]);

        Index row = residual / kInterleave;
        Index column_minor = residual % kInterleave;

        return MatrixCoord(row, column_major * kInterleave + column_minor);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Index stride(int idx) const { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int idx) { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(MatrixCoord const& extent) const {
        return (extent.column() + kInterleave - 1) / kInterleave * stride_[0];
    }
};

enum class Matrix {
    kColumnMajor,
    kRowMajor
};

struct ContiguousMatrix {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index>;

private:

    Stride stride_;

    Matrix layout_;

public:

    CUTLASS_HOST_DEVICE
    ContiguousMatrix(Index ldm = 0, Matrix layout = Matrix::kColumnMajor)
            : stride_(ldm), layout_(layout) {}

    CUTLASS_HOST_DEVICE
    static ContiguousMatrix packed(MatrixCoord const& extent,
                                   Matrix layout = Matrix::kColumnMajor) {
        Index ldm = 0;
        if (layout == Matrix::kColumnMajor) {
            ldm = extent.row();
        } else if (layout == Matrix::kRowMajor) {
            ldm = extent.column();
        }
        return ContiguousMatrix(ldm, layout);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(MatrixCoord const& coord) const {
        if (layout_ == Matrix::kColumnMajor) {
            return coord.row() + coord.column() * stride_[0];
        } else if (layout_ == Matrix::kRowMajor) {
            return coord.row() * stride_[0] + coord.column();
        } else {
            return 0;
        }
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord inverse(LongIndex offset) const {
        return MatrixCoord(0, 0);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Index stride(int idx) const { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int idx) { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(MatrixCoord const& extent) const {
        if (layout_ == Matrix::kColumnMajor) {
            return stride_[0] * extent.column();
        } else if (layout_ == Matrix::kRowMajor) {
            return stride_[0] * extent.row();
        } else {
            return 0;
        }
    }
};

template <int BlockRows, int BlockColumns>
struct ColumnMajorBlockLinear {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index>;

    static int const kBlockRows = BlockRows;

    static int const kBlockColumns = BlockColumns;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    ColumnMajorBlockLinear(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    static ColumnMajorBlockLinear packed(MatrixCoord const& extent) {
        return ColumnMajorBlockLinear(extent.row() * kBlockRows *
                                      kBlockColumns);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(MatrixCoord const& coord) const {
        return (coord.row() % kBlockRows) +
               (coord.column() % kBlockColumns) * kBlockRows +
               (coord.row() / kBlockRows) * kBlockRows * kBlockColumns +
               (coord.column() / kBlockColumns) * stride_[0];
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord inverse(LongIndex offset) const {
        return MatrixCoord(0, 0);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Index stride(int idx) const { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int idx) { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(MatrixCoord const& extent) const {
        return (extent.column() + kBlockColumns - 1) / kBlockColumns *
               stride_[0];
    }
};

template <int BlockRows, int BlockColumns>
struct RowMajorBlockLinear {
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index>;

    static int const kBlockRows = BlockRows;

    static int const kBlockColumns = BlockColumns;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    RowMajorBlockLinear(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    static RowMajorBlockLinear packed(MatrixCoord const& extent) {
        return RowMajorBlockLinear(extent.column() * kBlockRows *
                                   kBlockColumns);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(MatrixCoord const& coord) const {
        return (coord.column() % kBlockColumns) +
               (coord.row() % kBlockRows) * kBlockColumns +
               (coord.column() / kBlockColumns) * kBlockRows * kBlockColumns +
               (coord.row() / kBlockRows) * stride_[0];
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord inverse(LongIndex offset) const {
        return MatrixCoord(0, 0);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Index stride(int idx) const { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int idx) { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(MatrixCoord const& extent) const {
        return (extent.row() + kBlockRows - 1) / kBlockRows * stride_[0];
    }
};


struct GeneralMatrix {
    static int const kRank = 2;

    static int const kStrideRank = 2;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = MatrixCoord;

    using Stride = Coord<kStrideRank, Index>;

private:

    Matrix layout_id_;

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    GeneralMatrix()
            : layout_id_(Matrix::kColumnMajor), stride_(make_Coord(0, 1)) {}

    CUTLASS_HOST_DEVICE
    GeneralMatrix(Matrix layout_id, Index ldm, Index interleave)
            : layout_id_(layout_id), stride_(make_Coord(ldm, interleave)) {}

    CUTLASS_HOST_DEVICE
    static GeneralMatrix packed(MatrixCoord const& extent,
                                Matrix layout_id = Matrix::kColumnMajor,
                                Index interleave = 1) {
        Index c;
        if (layout_id == Matrix::kRowMajor) {
            c = extent.column();
        } else {
            c = extent.row();
        }

        Index ldm = c * interleave;

        return GeneralMatrix(layout_id, ldm, interleave);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(MatrixCoord const& coord) const {
        Index c, s;
        if (layout_id_ == Matrix::kRowMajor) {
            c = coord.column();
            s = coord.row();
        } else {
            s = coord.column();
            c = coord.row();
        }

        Index v = s / stride_[1];
        Index residual = (s % stride_[1]);

        return LongIndex(c) * LongIndex(stride_[1]) +
               LongIndex(v) * LongIndex(stride_[0]) + residual;
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Matrix layout_id() const { return layout_id_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Matrix& layout_id() { return layout_id_; }

    CUTLASS_HOST_DEVICE
    Index stride(int idx) const { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int idx) { return stride_[idx]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(MatrixCoord const& extent) const {
        Index s;
        if (layout_id_ == Matrix::kRowMajor) {
            s = extent.row();
        } else {
            s = extent.column();
        }

        Index v = Index((s + stride_[1] - 1) / stride_[1]);
        return LongIndex(v) * LongIndex(stride_[0]);
    }
};


template <typename Layout>
struct LayoutTranspose;

template <>
struct LayoutTranspose<layout::RowMajor> {
    using type = layout::ColumnMajor;
};

template <>
struct LayoutTranspose<layout::ColumnMajor> {
    using type = layout::RowMajor;
};


}
}
