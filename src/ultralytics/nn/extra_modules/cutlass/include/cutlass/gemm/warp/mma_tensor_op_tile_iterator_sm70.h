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
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"

#include "cutlass/platform/platform.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename Shape_,
        Operand Operand,
        typename Element_,
        typename Layout_,
        typename InstructionShape_,
        int OpDelta_,
        int Threads>
class MmaVoltaTensorOpMultiplicandTileIterator;


template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
        Shape_, Operand::kA, Element_,
        cutlass::layout::VoltaTensorOpMultiplicandCongruous<
                sizeof_bits<Element_>::value>,
        InstructionShape_, OpDelta_, 32> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kA;

    using Element = Element_;

    using Layout = cutlass::layout::VoltaTensorOpMultiplicandCongruous<
            sizeof_bits<Element_>::value>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static_assert(
                !(Shape::kContiguous % InstructionShape::kContiguous),
                "Shape of warp-level Mma must be divisible by operator shape.");

        using LdsShape = layout::PitchLinearShape<32, 4>;

        using LdsIterations = layout::PitchLinearShape<
                InstructionShape::kStrided / LdsShape::kStrided,
                Shape::kContiguous / LdsShape::kContiguous>;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    static int const kPointerCount = 2;

    using AccessType = AlignedArray<Element, Layout::kElementsPerAccess>;

public:

    using Fragment =
            Array<Element, Shape::kContiguous * InstructionShape::kStrided /
                                   kThreads * 2>;

private:
    Index stride_;

    AccessType const* pointer_[kPointerCount];

    Index byte_offset_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : stride_(ref.stride(0) / Layout::kElementsPerAccess),
              byte_offset_(0) {

        int vec_row = (lane_id >> 4);
        int vec_col = ((lane_id & 4) >> 2);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount; ++i) {
            if (i == 1) {
                vec_row |= 2;
            }
            int access_contiguous_idx =
                    (vec_col << 2) | ((lane_id & 3) ^ vec_row);
            int access_contiguous = access_contiguous_idx;

            int access_strided = vec_row;
            pointer_[i] = reinterpret_cast<AccessType const*>(ref.data()) +
                          access_contiguous + access_strided * stride_;
        }
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        byte_offset_ += offset * sizeof(Element);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int contiguous_offset = tile_offset.contiguous();
        int strided_offset = tile_offset.strided();

        if (Shape::kContiguous == Policy::LdsShape::kContiguous) {
            if (contiguous_offset % 2) {
                AccessType const* tmp_pointer = pointer_[0];
                pointer_[0] = pointer_[1];
                pointer_[1] = tmp_pointer;
            }
            contiguous_offset = contiguous_offset / 2;
        }

        int offset = (strided_offset * InstructionShape::kStrided) * stride_ *
                             Layout::kElementsPerAccess +
                     contiguous_offset * Shape::kContiguous;

        add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator++() {
        byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element) *
                        Layout::kElementsPerAccess;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator--() {
        byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                        Layout::kElementsPerAccess;

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        AccessType* fetch_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
                int access_idx = c + s * Policy::LdsIterations::kContiguous;

                AccessType const* source_ptr =
                        pointer_[s & 1] + Policy::LdsShape::kContiguous * c +
                        Policy::LdsShape::kStrided * (s / 2) * stride_;

                char const* source_byte_ptr =
                        reinterpret_cast<char const*>(source_ptr) +
                        byte_offset + byte_offset_;
                fetch_ptr[access_idx] =
                        *(reinterpret_cast<AccessType const*>(source_byte_ptr));
            }
        }
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        load_with_byte_offset(frag, pointer_offset * sizeof(Element));
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
        load_with_byte_offset(frag, tile_offset, 0);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        load_with_byte_offset(frag, tile_offset,
                              pointer_offset * sizeof(Element));
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        Index pointer_offset =
                tile_offset.contiguous() * Shape::kContiguous /
                        Layout::kElementsPerAccess +
                tile_offset.strided() * InstructionShape::kStrided * stride_;

        byte_offset += sizeof(AccessType) * pointer_offset;

        load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};


template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_>

class MmaVoltaTensorOpMultiplicandTileIterator<
        Shape_, Operand::kB, Element_,
        cutlass::layout::VoltaTensorOpMultiplicandBCongruous<
                sizeof_bits<Element_>::value>,
        InstructionShape_, OpDelta_, 32> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kB;

    using Element = Element_;

    using Layout = cutlass::layout::VoltaTensorOpMultiplicandBCongruous<
            sizeof_bits<Element_>::value>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static_assert(
                !(Shape::kContiguous % InstructionShape::kContiguous),
                "Shape of warp-level Mma must be divisible by operator shape.");

        using LdsShape = layout::PitchLinearShape<32, 4>;

        using LdsIterations = layout::PitchLinearShape<
                Shape::kContiguous / LdsShape::kContiguous,
                InstructionShape::kStrided / LdsShape::kStrided>;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    using AccessType = AlignedArray<Element, Layout::kElementsPerAccess>;

public:

    using Fragment =
            Array<Element, Shape::kContiguous * InstructionShape::kStrided /
                                   kThreads * 2>;

private:
    Index stride_;

    AccessType const* pointer_;

    Index byte_offset_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : stride_(ref.stride(0) / Layout::kElementsPerAccess),
              byte_offset_(0) {
        int access_strided = (lane_id >> 3) & 0x3;
        int access_contiguous = ((lane_id ^ (lane_id >> 3)) & 0x3);

        pointer_ = reinterpret_cast<AccessType const*>(ref.data()) +
                   access_contiguous + access_strided * stride_;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        byte_offset_ += offset * sizeof(Element);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int contiguous_offset = tile_offset.contiguous();
        int strided_offset = tile_offset.strided();

        int offset = (strided_offset * InstructionShape::kStrided) * stride_ *
                             Layout::kElementsPerAccess +
                     contiguous_offset * Shape::kContiguous;

        add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator++() {
        byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element) *
                        Layout::kElementsPerAccess;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator--() {
        byte_offset_ += stride_ * InstructionShape::kStrided * sizeof(Element) *
                        Layout::kElementsPerAccess;

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        AccessType* fetch_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
                int access_idx = c + s * Policy::LdsIterations::kContiguous;

                AccessType const* source_ptr =
                        pointer_ +
                        Policy::LdsShape::kContiguous /
                                Layout::kElementsPerAccess * c +
                        Policy::LdsShape::kStrided * s * stride_;

                char const* source_byte_ptr =
                        reinterpret_cast<char const*>(source_ptr) +
                        byte_offset + byte_offset_;
                fetch_ptr[access_idx] =
                        *(reinterpret_cast<AccessType const*>(source_byte_ptr));
            }
        }
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        load_with_byte_offset(frag, pointer_offset * sizeof(Element));
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
        load_with_byte_offset(frag, tile_offset, 0);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        load_with_byte_offset(frag, tile_offset,
                              pointer_offset * sizeof(Element));
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        Index pointer_offset =
                tile_offset.contiguous() * Shape::kContiguous /
                        Layout::kElementsPerAccess +
                tile_offset.strided() * InstructionShape::kStrided * stride_;

        byte_offset += sizeof(AccessType) * pointer_offset;

        load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};


template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
        Shape_, Operand::kA, Element_,
        cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<
                sizeof_bits<Element_>::value>,
        InstructionShape_, OpDelta_, 32> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kA;

    using Element = Element_;

    using Layout =
            cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<
                    sizeof_bits<Element_>::value>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaVoltaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand,
            Element,
            layout::VoltaTensorOpMultiplicandCongruous<
                    sizeof_bits<Element_>::value>,
            layout::PitchLinearShape<InstructionShape::kRow,
                                     InstructionShape::kColumn>,
            kOpDelta, kThreads>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.row(), tile_offset.column()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                -PitchLinearCoord(tile_offset.row(), tile_offset.column()));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { iterator_.load(frag); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        iterator_.load_with_pointer_offset(frag, pointer_offset);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        iterator_.load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        iterator_.load_with_byte_offset(
                frag, {tile_offset.contiguous(), tile_offset.strided()},
                byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) { iterator_.set_kgroup_index(k_group); }
};


template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<
        Shape_, Operand::kB, Element_,
        cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
                sizeof_bits<Element_>::value>,
        InstructionShape_, OpDelta_, 32> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kB;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
            sizeof_bits<Element_>::value>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaVoltaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand,
            Element,
            layout::VoltaTensorOpMultiplicandBCongruous<
                    sizeof_bits<Element_>::value>,
            layout::PitchLinearShape<InstructionShape::kColumn,
                                     InstructionShape::kRow>,
            kOpDelta, kThreads>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.column(), tile_offset.row()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                -PitchLinearCoord(tile_offset.column(), tile_offset.row()));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { iterator_.load(frag); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        iterator_.load_with_pointer_offset(frag, pointer_offset);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        iterator_.load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        iterator_.load_with_byte_offset(
                frag, {tile_offset.strided(), tile_offset.contiguous()},
                byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) { iterator_.set_kgroup_index(k_group); }
};


template <
        typename Shape_,
        typename Element_,
        typename Layout_,
        typename InstructionShape_,
        typename OpDelta_>
class MmaVoltaTensorOpAccumulatorTileIterator {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kC;

    using Element = Element_;

    using Layout = Layout_;

    using InstructionShape = InstructionShape_;

    using OpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        using InterleavedTile = MatrixShape<32, 32>;

        static_assert(
                !(Shape::kRow % InterleavedTile::kRow) &&
                        !(Shape::kColumn % InterleavedTile::kColumn),
                "Shape of warp-level Mma must be divisible by operator shape.");

        static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
                      "Layouts must be defined for logical MatrixCoord "
                      "coordinate space.");

        using TileIterations =
                MatrixShape<Shape::kRow / InterleavedTile::kRow,
                            Shape::kColumn / InterleavedTile::kColumn>;

        using MmaIterations =
                MatrixShape<InterleavedTile::kRow / InstructionShape::kM,
                            InterleavedTile::kColumn / InstructionShape::kN>;
    };

private:
    static int const kElementsPerPartial = 4;
    using EleShapePerPatial = typename platform::conditional<
            platform::is_same<Element, float>::value, MatrixShape<2, 2>,
            MatrixShape<1, 4> >::type;
    static int const kElementsPerMma = 8;
    static int const kAccumulatorPatials = 2;
    using QuadShapePerPatialMma = MatrixShape<4, 4>;

public:

    using Fragment = Array<Element, Shape::kCount / kThreads>;

private:
    TensorRef ref_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpAccumulatorTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpAccumulatorTileIterator(TensorRef const& ref, int lane_id)
            : ref_(ref) {
        int quad = (lane_id >> 2);
        int lane_in_quad = (lane_id & 3);
        int accum_m, accum_n;

        if (platform::is_same<Element, float>::value) {
            accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 +
                      (lane_in_quad & 1);
            accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial *
                              kAccumulatorPatials +
                      (lane_in_quad & 2);
        } else {
            accum_m = (((quad & 0x4) >> 1) + (quad & 0x1)) * 8 +
                      lane_in_quad;
            accum_n = ((quad >> 1) & 0x1) * kElementsPerPartial *
                      kAccumulatorPatials;
        }
        MatrixCoord lane_offset(accum_m, accum_n);

        ref_.add_coord_offset(lane_offset);
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpAccumulatorTileIterator& add_pointer_offset(
            LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpAccumulatorTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        ref_.add_coord_offset(tile_offset *
                              make_Coord(Shape::kRow, Shape::kColumn));

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpAccumulatorTileIterator& operator++() {
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpAccumulatorTileIterator& operator--() {
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpAccumulatorTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpAccumulatorTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset)
            const {

        TensorRef offset_ref(ref_);
        offset_ref.add_pointer_offset(pointer_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn;
             ++tile_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int tile_m = 0; tile_m < Policy::TileIterations::kRow;
                 ++tile_m) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn;
                     ++mma_n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow;
                         ++mma_m) {
                        int mma_accum_start =
                                (((tile_n * Policy::TileIterations::kRow +
                                   tile_m) *
                                          Policy::MmaIterations::kColumn +
                                  mma_n) *
                                         Policy::MmaIterations::kRow +
                                 mma_m) *
                                kElementsPerMma;

                        CUTLASS_PRAGMA_UNROLL
                        for (int p = 0; p < kAccumulatorPatials; ++p) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
                                CUTLASS_PRAGMA_UNROLL
                                for (int n = 0; n < EleShapePerPatial::kColumn;
                                     ++n) {
                                    int accum_m =
                                            tile_m * Policy::InterleavedTile::
                                                             kRow +
                                            mma_m * QuadShapePerPatialMma::
                                                            kRow +
                                            m * 2;
                                    int accum_n =
                                            tile_n * Policy::InterleavedTile::
                                                             kColumn +
                                            mma_n * QuadShapePerPatialMma::
                                                            kColumn +
                                            p *
                                                    Policy::InterleavedTile::
                                                            kColumn /
                                                    2 +
                                            n;
                                    int idx = mma_accum_start +
                                              p * kElementsPerPartial +
                                              m * EleShapePerPatial::kColumn +
                                              n;
                                    frag[idx] =
                                            offset_ref.at({accum_m, accum_n});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {

        load_with_pointer_offset(byte_offset / sizeof(Element));
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag,
              TensorCoord const& tile_offset)
            const {

        load(frag, tile_offset, 0);
    }

    CUTLASS_HOST_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {

        load_with_pointer_offset(frag,
                                 ref_.offset(tile_offset) + pointer_offset);
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) const {
        store_with_pointer_offset(frag, 0);
    }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(
            Fragment const& frag,
            Index pointer_offset)
            const {

        TensorRef offset_ref(ref_);
        offset_ref.add_pointer_offset(pointer_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn;
             ++tile_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int tile_m = 0; tile_m < Policy::TileIterations::kRow;
                 ++tile_m) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn;
                     ++mma_n) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow;
                         ++mma_m) {
                        int mma_accum_start =
                                (((tile_n * Policy::TileIterations::kRow +
                                   tile_m) *
                                          Policy::MmaIterations::kColumn +
                                  mma_n) *
                                         Policy::MmaIterations::kRow +
                                 mma_m) *
                                kElementsPerMma;

                        CUTLASS_PRAGMA_UNROLL
                        for (int p = 0; p < kAccumulatorPatials; ++p) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int m = 0; m < EleShapePerPatial::kRow; ++m) {
                                CUTLASS_PRAGMA_UNROLL
                                for (int n = 0; n < EleShapePerPatial::kColumn;
                                     ++n) {
                                    int accum_m =
                                            tile_m * Policy::InterleavedTile::
                                                             kRow +
                                            mma_m * QuadShapePerPatialMma::
                                                            kRow +
                                            m * 2;
                                    int accum_n =
                                            tile_n * Policy::InterleavedTile::
                                                             kColumn +
                                            mma_n * QuadShapePerPatialMma::
                                                            kColumn +
                                            p *
                                                    Policy::InterleavedTile::
                                                            kColumn /
                                                    2 +
                                            n;
                                    int idx = mma_accum_start +
                                              p * kElementsPerPartial +
                                              m * EleShapePerPatial::kColumn +
                                              n;
                                    offset_ref.at({accum_m, accum_n}) =
                                            frag[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void store_with_byte_offset(
            Fragment const& frag,
            Index byte_offset) const {

        store_with_pointer_offset(byte_offset / sizeof(Element));
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment& frag,
               TensorCoord const& tile_offset)
            const {

        store(frag, tile_offset, 0);
    }

    CUTLASS_HOST_DEVICE
    void store(
            Fragment const& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        store_with_pointer_offset(frag,
                                  ref_.offset(tile_offset) + pointer_offset);
    }
};

template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int KBlock>
class MmaVoltaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::VoltaTensorOpMultiplicandCrosswise<
                sizeof_bits<Element_>::value, KBlock>,
        InstructionShape_, OpDelta_, 32> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(
            kOperand == Operand::kA || kOperand == Operand::kB,
            "MmaVoltaTensorOpMultiplicandIterator may only be instantiated for "
            "A or B operands to warp-level Mma.");

    using Element = Element_;

    static int const kKBlock = KBlock;

    using Layout = cutlass::layout::VoltaTensorOpMultiplicandCrosswise<
            sizeof_bits<Element_>::value, kKBlock>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        using LdsShape = layout::PitchLinearShape<1, 32>;

        using LdsIterations = layout::PitchLinearShape<1, Shape::kStrided / 32>;

        static int const kElementsPerAccess = 8;

        static int const kContiguousElementsPerLine = 4;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

public:

    using Fragment =
            Array<Element, Shape::kStrided * InstructionShape::kContiguous /
                                   kThreads * 2>;

private:
    Index stride_;

    AccessType const* pointer_;

    Index byte_offset_;

    Index line_size;

    int k_group_idx_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator()
            : pointer_(nullptr),
              stride_(0),
              line_size(0),
              byte_offset_(0),
              k_group_idx_(0) {}

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : pointer_(reinterpret_cast<AccessType const*>(ref.data())),
              stride_(ref.stride(0) * Policy::kElementsPerAccess),
              line_size((ref.stride(0) * Policy::kContiguousElementsPerLine) /
                        Policy::kElementsPerAccess),
              k_group_idx_(0),
              byte_offset_(0) {
        int quad = (lane_id / 4);
        int lane_in_quad = (lane_id % 4);
        int access_contiguous;

        if (kOperand == Operand::kA) {
            access_contiguous = ((quad & 0x4) << 1) + ((lane_in_quad) << 1) +
                                ((quad & 0x1) ^ ((quad & 0x4) >> 2));
        } else {
            access_contiguous = ((quad & 0x4) << 1) + (lane_in_quad << 1) +
                                ((quad & 0x2) >> 1 ^ ((quad & 0x4) >> 2));
        }

        byte_offset_ = access_contiguous * sizeof(Element) *
                       Policy::kElementsPerAccess;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        byte_offset_ += offset * sizeof(Element);

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int contiguous_offset = tile_offset.contiguous();
        int strided_offset = tile_offset.strided();
        k_group_idx_ = 0;

        pointer_ += contiguous_offset *
                            (InstructionShape::kContiguous /
                             Policy::kContiguousElementsPerLine) *
                            line_size +
                    strided_offset * Shape::kStrided / 2;
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator++() {
        k_group_idx_ = (k_group_idx_ + 1) % 8;

        if (k_group_idx_ == 4 || k_group_idx_ == 0) {
            byte_offset_ ^= 1 * sizeof(Element) * Policy::kElementsPerAccess;
        }

        pointer_ += line_size;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator--() { assert(0); }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_byte_offset(frag, 0); }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        AccessType* fetch_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
                int access_idx = c + s * Policy::LdsIterations::kContiguous;

                AccessType const* source_ptr =
                        pointer_ +
                        Policy::LdsShape::kContiguous * c * line_size +
                        Policy::LdsShape::kStrided * s / 2;

                char const* source_byte_ptr =
                        reinterpret_cast<char const*>(source_ptr) +
                        byte_offset + byte_offset_;
                fetch_ptr[access_idx] =
                        *(reinterpret_cast<AccessType const*>(source_byte_ptr));

                if (k_group_idx_ & 0x2) {
                    uint64_t* low =
                            reinterpret_cast<uint64_t*>(&frag) + access_idx * 2;
                    uint64_t* high = reinterpret_cast<uint64_t*>(&frag) +
                                     access_idx * 2 + 1;
                    uint64_t tmp = *low;
                    *low = *high;
                    *high = tmp;
                }
            }
        }
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        load_with_byte_offset(frag, pointer_offset * sizeof(Element));
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
        load_with_byte_offset(frag, tile_offset, 0);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        load_with_byte_offset(frag, tile_offset,
                              pointer_offset * sizeof(Element));
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        Index pointer_offset =
                tile_offset.contiguous() * InstructionShape::kContiguous /
                        Policy::kElementsPerAccess +
                tile_offset.strided() * Shape::kStrided * stride_;

        byte_offset += sizeof(AccessType) * pointer_offset;

        load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) { k_group_idx_ = k_group; }
};

template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int KBlock>
class MmaVoltaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
                sizeof_bits<Element_>::value, KBlock>,
        InstructionShape_, OpDelta_, 32> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(
            kOperand == Operand::kA || kOperand == Operand::kB,
            "MmaTensorOpMultiplicandIterator may only be instantiated for "
            "A or B operands to warp-level Mma.");

    using Element = Element_;

    static int const kKBlock = KBlock;

    using Layout =
            cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<
                    sizeof_bits<Element_>::value, kKBlock>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaVoltaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand,
            Element,
            layout::VoltaTensorOpMultiplicandCrosswise<
                    sizeof_bits<Element_>::value, kKBlock>,
            layout::PitchLinearShape<InstructionShape::kRow,
                                     InstructionShape::kColumn>,
            kOpDelta, kThreads>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.row(), tile_offset.column()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                -PitchLinearCoord(tile_offset.row(), tile_offset.column()));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { iterator_.load(frag); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        iterator_.load_with_pointer_offset(frag, pointer_offset);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        iterator_.load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
        assert(0);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        assert(0);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        iterator_.load_with_byte_offset(
                frag, {tile_offset.contiguous(), tile_offset.strided()},
                byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) { iterator_.set_kgroup_index(k_group); }
};


template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int KBlock>
class MmaVoltaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
                sizeof_bits<Element_>::value, KBlock>,
        InstructionShape_, OpDelta_, 32> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(
            kOperand == Operand::kA || kOperand == Operand::kB,
            "MmaTensorOpMultiplicandIterator may only be instantiated for "
            "A or B operands to warp-level Mma.");

    using Element = Element_;

    static int const kKBlock = KBlock;

    using Layout = cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
            sizeof_bits<Element_>::value, kKBlock>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaVoltaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand,
            Element,
            layout::VoltaTensorOpMultiplicandCrosswise<
                    sizeof_bits<Element_>::value, kKBlock>,
            layout::PitchLinearShape<InstructionShape::kColumn,
                                     InstructionShape::kRow>,
            kOpDelta, kThreads>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_pointer_offset(
            LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.column(), tile_offset.row()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                -PitchLinearCoord(tile_offset.column(), tile_offset.row()));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { iterator_.load(frag); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        iterator_.load_with_pointer_offset(frag, pointer_offset);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        iterator_.load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
        assert(0);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        assert(0);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        iterator_.load_with_byte_offset(
                frag, {tile_offset.strided(), tile_offset.contiguous()},
                byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) { iterator_.set_kgroup_index(k_group); }
};


template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename Layout_,
        typename InstructionShape_,
        int OpDelta_,
        int Threads = 32,
        int PartitionsK_ = 1>
class MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaVoltaTensorOpMultiplicandIterator may only be "
                  "instantiated for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = Layout_;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static int const kElementsPerAccess = 4;

private:
    static int const kInterleavedTileRows = 32;
    static int const kInterleavedTileColumns = 32;
    static int const kInstructionsPerTile = 2;

    using TileCount = MatrixShape<Shape::kRow / kInterleavedTileRows,
                                  Shape::kColumn / kInterleavedTileColumns>;

    using FragmentCount =
            MatrixShape<TileCount::kRow * kInstructionsPerTile,
                        TileCount::kColumn * kInstructionsPerTile>;

public:

    using Fragment =
            Array<Element, (kOperand == Operand::kA ? FragmentCount::kRow
                                                    : FragmentCount::kColumn) *
                                   kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;

private:
    TensorRef ref_;

    MatrixCoord extent_;

    MatrixCoord origin_;

    bool divisible_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner()
            : divisible_(true) {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner(TensorRef const& ref,
                                                           int lane_id)
            : ref_(ref),
              extent_(Shape::kRow, Shape::kColumn),
              divisible_(true) {
        int quad_id = lane_id / 4;
        int lane_in_quad = (lane_id % 4);

        if (kOperand == Operand::kA) {
            int row_idx = ((quad_id & 1) + ((quad_id & 4) / 2)) * 4 *
                                  kInstructionsPerTile +
                          lane_in_quad;
            int col_idx = 0;

            origin_ = MatrixCoord(row_idx, col_idx);
        } else {
            int row_idx = 0;
            int col_idx =
                    (quad_id / 2) * 4 * kInstructionsPerTile + lane_in_quad;

            origin_ = MatrixCoord(row_idx, col_idx);
        }

        ref_.add_coord_offset(origin_);
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner(TensorRef const& ref,
                                                           TensorCoord extent,
                                                           int lane_id)
            : ref_(ref), extent_(extent), divisible_(false) {
        int quad_id = lane_id / 4;
        int lane_in_quad = (lane_id % 4);

        if (kOperand == Operand::kA) {
            int row_idx = ((quad_id & 1) + ((quad_id & 4) / 2)) * 4 *
                                  kInstructionsPerTile +
                          lane_in_quad;
            int col_idx = 0;

            origin_ = MatrixCoord(row_idx, col_idx);
        } else {
            int row_idx = 0;
            int col_idx =
                    (quad_id / 2) * 4 * kInstructionsPerTile + lane_in_quad;

            origin_ = MatrixCoord(row_idx, col_idx);
        }

#if defined(__CUDA_ARCH__)
        __syncthreads();
#endif

        ref_.add_coord_offset(origin_);
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner& add_pointer_offset(
            LongIndex offset) {
        ref_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner& add_tile_offset(
            TensorCoord const& tile_offset) {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);
        origin_ += coord_offset;

        ref_.add_coord_offset(coord_offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner& operator++() {
        if (kOperand == Operand::kA) {
            add_tile_offset({0, 1});
        } else {
            add_tile_offset({1, 0});
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner& operator--() {
        if (kOperand == Operand::kA) {
            add_tile_offset({0, -1});
        } else {
            add_tile_offset({-1, 0});
        }

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        AccessType const* access_ptr =
                reinterpret_cast<AccessType const*>(ref_.data());
        int ldm = ref_.stride()[0];

        if (kOperand == Operand::kA) {
            CUTLASS_PRAGMA_UNROLL
            for (int idx = 0; idx < FragmentCount::kRow; ++idx) {
                int tile_idx = idx / 2;
                int quad_idx = idx % 2;

                int row_offset = tile_idx * kInterleavedTileRows + quad_idx * 4;
                frag_ptr[idx] =
                        access_ptr[row_offset * ldm / kElementsPerAccess];
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int idx = 0; idx < FragmentCount::kColumn; ++idx) {
                int tile_idx = idx / 2;
                int quad_idx = idx % 2;

                int col_offset =
                        tile_idx * kInterleavedTileColumns + quad_idx * 4;
                frag_ptr[idx] =
                        access_ptr[col_offset * ldm / kElementsPerAccess];
            }
        }
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        load_with_pointer_offset(frag,
                                 byte_offset * 8 / sizeof_bits<Element>::value);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);

        load_with_pointer_offset(frag, ref_.offset(coord_offset));
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);

        load_with_pointer_offset(frag,
                                 ref_.offset(coord_offset) + pointer_offset);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);

        load_with_pointer_offset(
                frag, ref_.offset(coord_offset) +
                              byte_offset * 8 / sizeof_bits<Element>::value);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};

template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename Layout_,
        typename InstructionShape_,
        int OpDelta_,
        int Threads = 32,
        int PartitionsK_ = 1>
class MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaVoltaTensorOpMultiplicandIterator may only be "
                  "instantiated for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = Layout_;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static int const kElementsPerAccess = 4;

private:
    static int const kInterleavedTileRows = 32;
    static int const kInterleavedTileColumns = 32;
    static int const kInstructionsPerTile = 2;

    using TileCount = MatrixShape<Shape::kRow / kInterleavedTileRows,
                                  Shape::kColumn / kInterleavedTileColumns>;

    using FragmentCount =
            MatrixShape<TileCount::kRow * kInstructionsPerTile,
                        TileCount::kColumn * kInstructionsPerTile>;

public:

    using Fragment =
            Array<Element, (kOperand == Operand::kA ? FragmentCount::kRow
                                                    : FragmentCount::kColumn) *
                                   kElementsPerAccess>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;

private:
    TensorRef ref_;

    MatrixCoord extent_;

    MatrixCoord origin_;

    bool divisible_;

public:
    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter()
            : divisible_(true) {}

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter(TensorRef const& ref,
                                                           int lane_id)
            : ref_(ref),
              extent_(Shape::kRow, Shape::kColumn),
              divisible_(true) {
        int quad_id = lane_id / 4;
        int lane_in_quad = (lane_id % 4);

        if (kOperand == Operand::kA) {
            int row_idx = ((quad_id & 1) + ((quad_id & 4) / 2)) * 4 *
                          kInstructionsPerTile;
            int col_idx = lane_in_quad;

            origin_ = MatrixCoord(row_idx, col_idx);
        } else {
            int row_idx = lane_in_quad;
            int col_idx = (quad_id / 2) * 4 * kInstructionsPerTile;

            origin_ = MatrixCoord(row_idx, col_idx);
        }

        ref_.add_coord_offset(origin_);
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter(TensorRef const& ref,
                                                           TensorCoord extent,
                                                           int lane_id)
            : ref_(ref), extent_(extent), divisible_(false) {
        int quad_id = lane_id / 4;
        int lane_in_quad = (lane_id % 4);

        if (kOperand == Operand::kA) {
            int row_idx = ((quad_id & 1) + ((quad_id & 4) / 2)) * 4 *
                          kInstructionsPerTile;
            int col_idx = lane_in_quad;

            origin_ = MatrixCoord(row_idx, col_idx);
        } else {
            int row_idx = lane_in_quad;
            int col_idx = (quad_id / 2) * 4 * kInstructionsPerTile;

            origin_ = MatrixCoord(row_idx, col_idx);
        }

#if defined(__CUDA_ARCH__)
        __syncthreads();
#endif

        ref_.add_coord_offset(origin_);
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter& add_pointer_offset(
            LongIndex offset) {
        ref_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter& add_tile_offset(
            TensorCoord const& tile_offset) {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);
        origin_ += coord_offset;

        ref_.add_coord_offset(coord_offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter& operator++() {
        if (kOperand == Operand::kA) {
            add_tile_offset({0, 1});
        } else {
            add_tile_offset({1, 0});
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter& operator--() {
        if (kOperand == Operand::kA) {
            add_tile_offset({0, -1});
        } else {
            add_tile_offset({-1, 0});
        }

        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);
        AccessType const* access_ptr =
                reinterpret_cast<AccessType const*>(ref_.data());
        int ldm = ref_.stride()[0];

        if (kOperand == Operand::kA) {
            CUTLASS_PRAGMA_UNROLL
            for (int idx = 0; idx < FragmentCount::kRow; ++idx) {
                int tile_idx = idx / 2;
                int quad_idx = idx % 2;

                int row_offset = tile_idx * kInterleavedTileRows;
                frag_ptr[idx] =
                        access_ptr[row_offset / kElementsPerAccess + quad_idx];
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int idx = 0; idx < FragmentCount::kColumn; ++idx) {
                int tile_idx = idx / 2;
                int quad_idx = idx % 2;

                int col_offset = tile_idx * kInterleavedTileColumns;
                frag_ptr[idx] =
                        access_ptr[col_offset / kElementsPerAccess + quad_idx];
            }
        }
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            Index byte_offset) const {
        load_with_pointer_offset(frag,
                                 byte_offset * 8 / sizeof_bits<Element>::value);
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset) const {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);

        load_with_pointer_offset(frag, ref_.offset(coord_offset));
    }

    CUTLASS_DEVICE
    void load(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index pointer_offset) const {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);

        load_with_pointer_offset(frag,
                                 ref_.offset(coord_offset) + pointer_offset);
    }

    CUTLASS_DEVICE
    void load_with_byte_offset(
            Fragment& frag,
            TensorCoord const& tile_offset,
            Index byte_offset) const {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);

        load_with_pointer_offset(
                frag, ref_.offset(coord_offset) +
                              byte_offset * 8 / sizeof_bits<Element>::value);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};


template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<Shape_, Operand::kA, Element_,
                                               cutlass::layout::RowMajor,
                                               InstructionShape_, OpDelta_, 32>
        : public MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner<
                  Shape_, Operand::kA, Element_, cutlass::layout::RowMajor,
                  InstructionShape_, OpDelta_> {
public:
    using Base = MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner<
            Shape_, Operand::kA, Element_, cutlass::layout::RowMajor,
            InstructionShape_, OpDelta_>;

    using TensorRef = typename Base::TensorRef;

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : Base(ref, lane_id) {}
};

template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<Shape_, Operand::kA, Element_,
                                               cutlass::layout::ColumnMajor,
                                               InstructionShape_, OpDelta_, 32>
        : public MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter<
                  Shape_, Operand::kA, Element_, cutlass::layout::ColumnMajor,
                  InstructionShape_, OpDelta_> {
public:
    using Base = MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter<
            Shape_, Operand::kA, Element_, cutlass::layout::ColumnMajor,
            InstructionShape_, OpDelta_>;

    using TensorRef = typename Base::TensorRef;

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : Base(ref, lane_id) {}
};

template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<Shape_, Operand::kB, Element_,
                                               cutlass::layout::ColumnMajor,
                                               InstructionShape_, OpDelta_, 32>
        : public MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner<
                  Shape_, Operand::kB, Element_, cutlass::layout::ColumnMajor,
                  InstructionShape_, OpDelta_> {
public:
    using Base = MmaVoltaTensorOpMultiplicandTileIteratorCanonicalInner<
            Shape_, Operand::kB, Element_, cutlass::layout::ColumnMajor,
            InstructionShape_, OpDelta_>;

    using TensorRef = typename Base::TensorRef;

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : Base(ref, lane_id) {}
};

template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_>
class MmaVoltaTensorOpMultiplicandTileIterator<Shape_, Operand::kB, Element_,
                                               cutlass::layout::RowMajor,
                                               InstructionShape_, OpDelta_, 32>
        : public MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter<
                  Shape_, Operand::kB, Element_, cutlass::layout::RowMajor,
                  InstructionShape_, OpDelta_> {
public:
    using Base = MmaVoltaTensorOpMultiplicandTileIteratorCanonicalOuter<
            Shape_, Operand::kB, Element_, cutlass::layout::RowMajor,
            InstructionShape_, OpDelta_>;

    using TensorRef = typename Base::TensorRef;

    CUTLASS_HOST_DEVICE
    MmaVoltaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : Base(ref, lane_id) {}
};


}
}
}

