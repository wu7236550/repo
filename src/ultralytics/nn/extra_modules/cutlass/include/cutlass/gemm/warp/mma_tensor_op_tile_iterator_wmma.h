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
#include "cutlass/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include "cutlass/wmma_array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"


namespace cutlass {
namespace gemm {
namespace warp {

template <
        typename Shape_,
        Operand Operand,
        typename Element_,
        typename Layout_,
        int OpDelta_,
        int Threads,
        typename Policy_>
class MmaTensorOpWmmaMultiplicandTileIterator;

template <
        typename Shape_,
        typename Element_,
        typename Layout_,
        int OpDelta_,
        typename Policy_>
class MmaTensorOpWmmaMultiplicandTileIterator<Shape_, Operand::kA, Element_,
                                              Layout_, OpDelta_, 32, Policy_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kA;

    using Element = Element_;

    using Layout = Layout_;

    static int const kOpDelta = OpDelta_;

    using Policy = Policy_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using WmmaShape = MatrixShape<Policy::Operator::Shape::kM,
                                  Policy::Operator::Shape::kK>;

    using WmmaDataType =
            typename cutlass::arch::CutlassToWmmaDataType<Element>::Type;

    using Iterations = MatrixShape<Shape::kRow / WmmaShape::kRow, 1>;

    using Fragment = WmmaFragmentArray<typename Policy::Operator::FragmentA,
                                       Iterations::kCount>;

    static_assert(kOperand == Operand::kA,
                  "MmaTensorOpWmmaMultiplicandTileIterator may only be "
                  "instantiated for A operands to warp-level Mma.");

    static_assert(platform::is_same<cutlass::layout::RowMajor, Layout>::value ||
                          platform::is_same<cutlass::layout::ColumnMajor,
                                            Layout>::value,
                  "Supported list of memory layouts for WMMA are: RowMajor, "
                  "ColumnMajor");

    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");


private:
    char const* pointer_;

    Index byte_offset_;

    Index stride_;

    Layout layout_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator() {}

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : pointer_(reinterpret_cast<char const*>(ref.data())),
              byte_offset_(0),
              stride_(ref.stride(0)),
              layout_(ref.stride(0)) {}

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        byte_offset_ += (offset * sizeof_bits<Element>::value) / 8;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        Index elements_offset =
                layout_({tile_offset.row() * Shape::kRow,
                         tile_offset.column() * WmmaShape::kColumn});

        byte_offset_ += (elements_offset * sizeof_bits<Element>::value) / 8;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& operator++() {
        Index elements_offset = layout_({0, WmmaShape::kColumn});

        byte_offset_ += (elements_offset * sizeof_bits<Element>::value) / 8;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& operator--() {
        Index elements_offset = layout_({0, WmmaShape::kColumn});

        byte_offset_ -= (elements_offset * sizeof_bits<Element>::value) / 8;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_byte_offset(Fragment& frag, Index byte_offset) const {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kColumn; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Iterations::kRow; ++m) {
                Index load_byte_offset =
                        layout_({m * WmmaShape::kRow, k * WmmaShape::kColumn}) *
                        sizeof_bits<Element>::value / 8;

                const WmmaDataType* ptr = reinterpret_cast<const WmmaDataType*>(
                        pointer_ + byte_offset_ + load_byte_offset +
                        byte_offset);

                nvcuda::wmma::load_matrix_sync(frag[m], ptr, stride_);
            }
        }
    }
    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_byte_offset(Fragment const& frag, Index byte_offset) const {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kColumn; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Iterations::kRow; ++m) {
                Index store_byte_offset =
                        layout_({m * WmmaShape::kRow, k * WmmaShape::kColumn}) *
                        sizeof_bits<Element>::value / 8;

                WmmaDataType* ptr = reinterpret_cast<WmmaDataType*>(
                        pointer_ + byte_offset_ + store_byte_offset +
                        byte_offset);

                nvcuda::wmma::store_matrix_sync(ptr, frag[m], stride_);
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) const { store_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};


template <
        typename Shape_,
        typename Element_,
        typename Layout_,
        int OpDelta_,
        typename Policy_>
class MmaTensorOpWmmaMultiplicandTileIterator<Shape_, Operand::kB, Element_,
                                              Layout_, OpDelta_, 32, Policy_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kB;

    using Element = Element_;

    using Layout = Layout_;

    static int const kOpDelta = OpDelta_;

    using Policy = Policy_;


    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using WmmaShape = MatrixShape<Policy::Operator::Shape::kK,
                                  Policy::Operator::Shape::kN>;

    using WmmaDataType =
            typename cutlass::arch::CutlassToWmmaDataType<Element>::Type;

    using Iterations = MatrixShape<1, Shape::kColumn / WmmaShape::kColumn>;

    using Fragment = WmmaFragmentArray<typename Policy::Operator::FragmentB,
                                       Iterations::kCount>;

    static_assert(kOperand == Operand::kB,
                  "MmaTensorOpWmmaMultiplicandTileIterator may only be "
                  "instantiated for B operands to warp-level Mma.");

    static_assert(platform::is_same<cutlass::layout::RowMajor, Layout>::value ||
                          platform::is_same<cutlass::layout::ColumnMajor,
                                            Layout>::value,
                  "Supported list of memory layouts for WMMA are: RowMajor, "
                  "ColumnMajor");

    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");


private:
    char const* pointer_;

    Index byte_offset_;

    Index stride_;

    Layout layout_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator() {}

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : pointer_(reinterpret_cast<char const*>(ref.data())),
              byte_offset_(0),
              stride_(ref.stride(0)),
              layout_(ref.stride(0)) {}

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        byte_offset_ += (offset * sizeof_bits<Element>::value) / 8;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        Index elements_offset =
                layout_({tile_offset.row() * WmmaShape::kRow,
                         tile_offset.column() * Shape::kColumn});

        byte_offset_ += (elements_offset * sizeof_bits<Element>::value) / 8;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& operator++() {
        Index elements_offset = layout_({WmmaShape::kRow, 0});

        byte_offset_ += (elements_offset * sizeof_bits<Element>::value) / 8;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& operator--() {
        Index elements_offset = layout_({WmmaShape::kRow, 0});

        byte_offset_ -= (elements_offset + sizeof_bits<Element>::value) / 8;
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_byte_offset(Fragment& frag, Index byte_offset) const {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kRow; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Iterations::kColumn; ++n) {
                Index load_byte_offset =
                        layout_({k * WmmaShape::kRow, n * WmmaShape::kColumn}) *
                        sizeof_bits<Element>::value / 8;

                const WmmaDataType* ptr = reinterpret_cast<const WmmaDataType*>(
                        pointer_ + byte_offset_ + load_byte_offset +
                        byte_offset);

                nvcuda::wmma::load_matrix_sync(frag[n], ptr, stride_);
            }
        }
    }
    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_byte_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_byte_offset(Fragment const& frag, Index byte_offset) const {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kRow; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Iterations::kColumn; ++n) {
                Index store_byte_offset =
                        layout_({k * WmmaShape::kRow, n * WmmaShape::kColumn}) *
                        sizeof_bits<Element>::value / 8;

                WmmaDataType* ptr = reinterpret_cast<WmmaDataType*>(
                        pointer_ + byte_offset_ + store_byte_offset +
                        byte_offset);

                nvcuda::wmma::store_matrix_sync(ptr, frag[n], stride_);
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) const { store_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};

template <
        typename Shape_,
        typename Element_,
        typename Layout_,
        typename OpDelta_,
        typename Policy_>
class MmaTensorOpWmmaAccumulatorTileIterator;


template <
        typename Shape_,
        typename Element_,
        typename Layout_,
        typename OpDelta_,
        typename Policy_>
class MmaTensorOpWmmaAccumulatorTileIterator {
public:
    using Shape = Shape_;

    using Element = Element_;

    using Layout = Layout_;

    using OpDelta = OpDelta_;

    static int const kThreads = 32;

    using Policy = Policy_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using WmmaShape = MatrixShape<Policy::Operator::Shape::kM,
                                  Policy::Operator::Shape::kN>;

    using WmmaDataType =
            typename cutlass::arch::CutlassToWmmaDataType<Element>::Type;

    static nvcuda::wmma::layout_t const WmmaLayout =
            cutlass::arch::CutlassToWmmaLayout<Layout>::value;

    using Iterations = MatrixShape<Shape::kRow / WmmaShape::kRow,
                                   Shape::kColumn / WmmaShape::kColumn>;

    using Fragment = WmmaFragmentArray<typename Policy::Operator::FragmentC,
                                       Iterations::kCount>;

    static_assert(platform::is_same<cutlass::layout::RowMajor, Layout>::value ||
                          platform::is_same<cutlass::layout::ColumnMajor,
                                            Layout>::value,
                  "Supported list of memory layouts for WMMA are: RowMajor, "
                  "ColumnMajor");

private:
    cutlass::TensorRef<Element, Layout> ref_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaAccumulatorTileIterator() {}

    CUTLASS_DEVICE
    MmaTensorOpWmmaAccumulatorTileIterator(TensorRef const& ref, int lane_id)
            : ref_(ref) {}

    CUTLASS_DEVICE
    MmaTensorOpWmmaAccumulatorTileIterator& add_pointer_offset(
            LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaAccumulatorTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        ref_.add_coord_offset({tile_offset.row() * Shape::kRow,
                               tile_offset.column() * Shape::kColumn});
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaAccumulatorTileIterator& operator++() {
        ref_.add_coord_offset({Shape::kRow, 0});
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpWmmaAccumulatorTileIterator& operator--() {
        ref_.add_coord_offset({-Shape::kRow, 0});
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaAccumulatorTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpWmmaAccumulatorTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Iterations::kRow; ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Iterations::kColumn; ++n) {
                const WmmaDataType* ptr = reinterpret_cast<const WmmaDataType*>(
                        ref_.data() +
                        ref_.offset(
                                {m * WmmaShape::kRow, n * WmmaShape::kColumn}) +
                        pointer_offset);

                nvcuda::wmma::load_matrix_sync(
                        frag[m * Iterations::kColumn + n], ptr,
                        ref_.stride()[0], WmmaLayout);
            }
        }
    }
    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag,
                                   Index pointer_offset) const {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Iterations::kRow; ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Iterations::kColumn; ++n) {
                WmmaDataType* ptr = reinterpret_cast<WmmaDataType*>(
                        ref_.data() +
                        ref_.offset(
                                {m * WmmaShape::kRow, n * WmmaShape::kColumn}) +
                        pointer_offset);

                nvcuda::wmma::store_matrix_sync(
                        ptr, frag[m * Iterations::kColumn + n],
                        ref_.stride()[0], WmmaLayout);
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) const {
        store_with_pointer_offset(frag, 0);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};

}
}
}


#endif
