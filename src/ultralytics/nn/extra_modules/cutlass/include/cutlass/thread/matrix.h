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
#include "cutlass/array.h"
#include "cutlass/matrix_coord.h"

namespace cutlass {
namespace thread {


template <typename Element, int Rows, int Columns,
          typename Layout = layout::RowMajor>
class Matrix : public Array<Element, Rows * Columns> {
public:
    static_assert(Layout::kRank == 2,
                  "Layout type must refer to a rank=2 matrix");

    using Base = Array<Element, Rows * Columns>;

    using Element = Element_;

    static int const kRows = Rows;

    static int const kColumns = Columns;

    using Layout = Layout_;

    using Reference = Element&;

    static int const kRank = 2;

    using Index = typename Layout::Index;

    using LongIndex = typename Layout::LongIndex;

    using TensorCoord = typename Layout::TensorCoord;

    using Stride = typename Layout::Stride;

    using TensorRef = TensorRef<Element, kRank, Layout>;

    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using TensorView = TensorView<Element, kRank, Layout>;

    using ConstTensorView = typename TensorView::ConstTensorView;

    using Diagonal = Vector<Element, __NV_STD_MIN(kRows, kColumns)>;

private:
public:

    CUTLASS_HOST_DEVICE
    static MatrixCoord extent() { return make_Coord(kRows, kColumns); }

    CUTLASS_HOST_DEVICE
    static Layout layout() { return Layout::packed(extent()); }

    CUTLASS_HOST_DEVICE
    Matrix() {}

    CUTLASS_HOST_DEVICE
    Matrix(Diagonal const& diag) {
    }

    CUTLASS_HOST_DEVICE
    TensorRef ref() { return TensorRef(this->data(), layout()); }

    CUTLASS_HOST_DEVICE
    ConstTensorRef const_ref() const {
        return ConstTensorRef(this->data(), layout());
    }

    CUTLASS_HOST_DEVICE
    TensorView view() { return TensorView(ref(), extent()); }

    CUTLASS_HOST_DEVICE
    ConstTensorView const_view() const {
        return ConstTensorView(const_ref(), extent());
    }

    CUTLASS_HOST_DEVICE
    Reference at(MatrixCoord const& coord) const {
        typename Base::size_type offset_(layout().offset(coord));
        return Base::at(offset_);
    }

    CUTLASS_HOST_DEVICE
    LongIndex capacity() const { return LongIndex(Base::size()); }
};


template <typename Element, int Rows, typename Layout = layout::ColumnMajor>
using ColumnVector = Matrix<Element, Rows, 1, Layout>;

template <typename Element, int Columns, typename Layout = layout::RowMajor>
using RowVector = Matrix<Element, 1, Columns, Layout>;


}
}
