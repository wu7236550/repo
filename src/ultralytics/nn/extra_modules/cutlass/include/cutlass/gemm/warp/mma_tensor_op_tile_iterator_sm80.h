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

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::TensorOpMultiplicandCongruous64b, InstructionShape_,
        OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    static_assert(!(Shape::kContiguous % 16) && !(Shape::kStrided % 4),
                  "Divisibility.");

    static_assert(sizeof_bits<Element_>::value == 64,
                  "This is specialized for 64b accesses.");

    using Element = Element_;

    using Layout = cutlass::layout::TensorOpMultiplicandCongruous64b;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    static int const kPartitionsK = PartitionsK_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static int const kElementsPerAccess = 2;

    struct Policy {
        using Delta = layout::PitchLinearShape<8, 4>;

        using Iterations = layout::PitchLinearShape<
                Shape::kContiguous / kElementsPerAccess / Delta::kContiguous,
                InstructionShape::kStrided / Delta::kStrided>;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    using AccessType = AlignedArray<Element, kElementsPerAccess, 16>;

    int k_group_idx_;

public:

    using Fragment =
            Array<Element,
                  Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:
    Index stride_;

    AccessType const* pointer_;

    Index byte_offset_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : stride_(ref.stride(0) / kElementsPerAccess),
              byte_offset_(0),
              k_group_idx_(0) {
        int access_strided = lane_id / Policy::Delta::kContiguous;
        int access_contiguous =
                (lane_id % Policy::Delta::kContiguous) ^ access_strided;

        pointer_ = reinterpret_cast<AccessType const*>(ref.data()) +
                   access_contiguous + access_strided * stride_;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        byte_offset_ += offset * sizeof(Element);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int offset = (tile_offset.strided() * InstructionShape::kStrided) *
                             stride_ * kElementsPerAccess +
                     tile_offset.contiguous() * Shape::kContiguous;

        add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        add_tile_offset({0, 1});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        add_tile_offset({0, -1});

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
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
        for (int s = 0; s < Policy::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < Policy::Iterations::kContiguous; ++c) {
                int access_idx = c + s * Policy::Iterations::kContiguous;

                AccessType const* source_ptr =
                        pointer_ + Policy::Delta::kContiguous * c +
                        Policy::Delta::kStrided * s * stride_;

                char const* source_byte_ptr =
                        reinterpret_cast<char const*>(source_ptr) +
                        byte_offset + byte_offset_;

                AccessType const* source =
                        reinterpret_cast<AccessType const*>(source_byte_ptr);

                fetch_ptr[access_idx] = *source;
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
    void set_kgroup_index(int k_group) {}
};


template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand,
            Element, layout::TensorOpMultiplicandCongruous64b,
            layout::PitchLinearShape<InstructionShape::kColumn,
                                     InstructionShape::kRow>,
            kOpDelta, kThreads, PartitionsK_>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.column(), tile_offset.row()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
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
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand,
            Element, layout::TensorOpMultiplicandCongruous64b,
            layout::PitchLinearShape<InstructionShape::kRow,
                                     InstructionShape::kColumn>,
            kOpDelta, kThreads, PartitionsK_>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.row(), tile_offset.column()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
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
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::TensorOpMultiplicand64bCrosswise, InstructionShape_,
        OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    static_assert(!(Shape::kContiguous % 4) && !(Shape::kStrided % 16),
                  "Divisibility.");

    static_assert(sizeof_bits<Element_>::value == 64,
                  "This is specialized for 64b accesses.");

    using Element = Element_;

    using Layout = cutlass::layout::TensorOpMultiplicand64bCrosswise;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    static int const kPartitionsK = PartitionsK_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static int const kElementsPerAccess = 2;

    struct Policy {
        using Delta = layout::PitchLinearShape<4, 16>;

        using Iterations =
                layout::PitchLinearShape<InstructionShape::kContiguous /
                                                 Delta::kContiguous,
                                         Shape::kStrided / Delta::kStrided>;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    using AccessType = AlignedArray<Element, kElementsPerAccess, 16>;

public:

    using Fragment =
            Array<Element,
                  Shape::kStrided * InstructionShape::kContiguous / kThreads>;

private:
    Index stride_;

    AccessType const* pointer_;

    Index byte_offset_;

    Index k_group_idx_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : stride_(ref.stride(0) / kElementsPerAccess),
              byte_offset_(0),
              k_group_idx_(0) {
        int access_strided = lane_id / 8;
        int access_contiguous = (lane_id % 8);

        byte_offset_ = (access_contiguous + access_strided * stride_) *
                       sizeof(AccessType);

        pointer_ = reinterpret_cast<AccessType const*>(ref.data());
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        pointer_ += offset / kElementsPerAccess;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int offset =
                (tile_offset.contiguous() * InstructionShape::kContiguous) *
                        stride_ * kElementsPerAccess +
                tile_offset.strided() * Shape::kStrided;

        add_pointer_offset(offset);

        int old_k_group_idx = k_group_idx_;

        k_group_idx_ += tile_offset.contiguous();

        if ((k_group_idx_ & 2) ^ (old_k_group_idx & 2)) {
            byte_offset_ ^= 0x40;
        }

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        pointer_ += stride_ * InstructionShape::kContiguous;

        if (k_group_idx_ & 0x1) {
            byte_offset_ ^= 0x40;
        }

        ++k_group_idx_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
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
        for (int c = 0; c < Policy::Iterations::kContiguous; ++c) {
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < Policy::Iterations::kStrided; ++s) {
                int access_idx = c + s * Policy::Iterations::kContiguous;

                AccessType const* source_ptr =
                        pointer_ + Policy::Delta::kContiguous * c * stride_ +
                        Policy::Delta::kStrided * s / kElementsPerAccess;

                char const* source_byte_ptr =
                        reinterpret_cast<char const*>(source_ptr) +
                        byte_offset + byte_offset_;

                AccessType const* source =
                        reinterpret_cast<AccessType const*>(source_byte_ptr);

                fetch_ptr[access_idx] = *source;
            }
        }

        Element* exchange_ptr = reinterpret_cast<Element*>(&frag);

        if (k_group_idx_ & 1) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Fragment::kElements; i += 2) {
                Element tmp = exchange_ptr[i];
                exchange_ptr[i] = exchange_ptr[i + 1];
                exchange_ptr[i + 1] = tmp;
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
                        Layout::kElementsPerAccess +
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
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand,
            Element, layout::TensorOpMultiplicand64bCrosswise,
            layout::PitchLinearShape<InstructionShape::kColumn,
                                     InstructionShape::kRow>,
            kOpDelta, kThreads, PartitionsK_>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.column(), tile_offset.row()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
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
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand,
            Element, layout::TensorOpMultiplicand64bCrosswise,
            layout::PitchLinearShape<InstructionShape::kRow,
                                     InstructionShape::kColumn>,
            kOpDelta, kThreads, PartitionsK_>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.row(), tile_offset.column()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
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
        Operand Operand_,
        typename Element_,
        typename Layout_,
        typename InstructionShape_,
        int OpDelta_,
        int Threads = 32,
        int PartitionsK_ = 1>
class MmaTensorOpMultiplicandTileIteratorCanonical {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = Layout_;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static int const kElementsPerAccess =
            (sizeof_bits<Element>::value >= 32
                     ? 1
                     : 32 / sizeof_bits<Element>::value);

private:
    static int const kWarpShapeOuter =
            (kOperand == Operand::kA ? Shape::kRow : Shape::kColumn);

    static int const kWarpShapeInner =
            (kOperand == Operand::kA ? Shape::kColumn : Shape::kRow);

    using InstructionCount =
            MatrixShape<Shape::kRow / InstructionShape::kRow,
                        Shape::kColumn / InstructionShape::kColumn>;

    using WarpShapeDivisible =
            MatrixShape<InstructionCount::kRow * InstructionShape::kRow,
                        InstructionCount::kColumn * InstructionShape::kColumn>;

public:

    using Fragment =
            Array<Element, WarpShapeDivisible::kRow *
                                   WarpShapeDivisible::kColumn / kThreads>;

    using AccessType = AlignedArray<Element, kElementsPerAccess>;

private:
    TensorRef ref_;

    MatrixCoord extent_;

    MatrixCoord origin_;

    bool divisible_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical() : divisible_(true) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical(TensorRef const& ref,
                                                 int lane_id)
            : ref_(ref),
              extent_(Shape::kRow, Shape::kColumn),
              divisible_(true) {
        if (kOperand == Operand::kA) {
            origin_ = MatrixCoord(lane_id / 4,
                                  (lane_id % 4) * kElementsPerAccess);
        } else {
            origin_ = MatrixCoord((lane_id % 4) * kElementsPerAccess,
                                  lane_id / 4);
        }

        ref_.add_coord_offset(origin_);
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical(TensorRef const& ref,
                                                 TensorCoord extent,
                                                 int lane_id)
            : ref_(ref), extent_(extent), divisible_(false) {
        if (kOperand == Operand::kA) {
            origin_ = MatrixCoord(lane_id / 4,
                                  (lane_id % 4) * kElementsPerAccess);
        } else {
            origin_ = MatrixCoord((lane_id % 4) * kElementsPerAccess,
                                  lane_id / 4);
        }

        ref_.add_coord_offset(origin_);
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical& add_pointer_offset(
            LongIndex offset) {
        ref_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical& add_tile_offset(
            TensorCoord const& tile_offset) {
        TensorCoord coord_offset(tile_offset.row() * Shape::kRow,
                                 tile_offset.column() * Shape::kColumn);
        origin_ += coord_offset;

        ref_.add_coord_offset(coord_offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical& operator++() {
        if (kOperand == Operand::kA) {
            add_tile_offset({0, 1});
        } else {
            add_tile_offset({1, 0});
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical& operator--() {
        if (kOperand == Operand::kA) {
            add_tile_offset({0, -1});
        } else {
            add_tile_offset({-1, 0});
        }

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIteratorCanonical& operator-=(
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
        int const kWarpShapeDivisibleInner =
                (kOperand == Operand::kA ? WarpShapeDivisible::kColumn
                                         : WarpShapeDivisible::kRow);

        int const kAccessesInner =
                (kWarpShapeDivisibleInner / kElementsPerAccess) / 4;

        AccessType* access_ptr = reinterpret_cast<AccessType*>(&frag);

        if (kOperand == Operand::kA) {
            int const kTilesPerInstruction = InstructionShape::kRow / 8;

            CUTLASS_PRAGMA_UNROLL
            for (int inst_m_idx = 0; inst_m_idx < InstructionCount::kRow;
                 ++inst_m_idx) {
                CUTLASS_PRAGMA_UNROLL
                for (int inner_idx = 0; inner_idx < kAccessesInner;
                     ++inner_idx) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int access_m_idx = 0;
                         access_m_idx < kTilesPerInstruction; ++access_m_idx) {
                        int access_idx = access_m_idx +
                                         kTilesPerInstruction *
                                                 (inner_idx +
                                                  kAccessesInner * inst_m_idx);

                        MatrixCoord offset(
                                access_m_idx * 8 +
                                        inst_m_idx * InstructionShape::kRow,
                                inner_idx * 4 * kElementsPerAccess);

                        MatrixCoord access_coord = origin_ + offset;

                        if (divisible_ ||
                            (access_coord.row() < extent_.row() &&
                             access_coord.column() < extent_.column())) {
                            access_ptr[access_idx] =
                                    *reinterpret_cast<AccessType const*>(
                                            ref_.data() + ref_.offset(offset));
                        } else {
                            AccessType zero;
                            zero.clear();
                            access_ptr[access_idx] = zero;
                        }
                    }
                }
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int inst_n_idx = 0; inst_n_idx < InstructionCount::kColumn;
                 ++inst_n_idx) {
                CUTLASS_PRAGMA_UNROLL
                for (int inner_idx = 0; inner_idx < kAccessesInner;
                     ++inner_idx) {
                    int access_idx = inner_idx + kAccessesInner * inst_n_idx;

                    MatrixCoord offset(inner_idx * 4 * kElementsPerAccess,
                                       inst_n_idx * 8);

                    MatrixCoord access_coord = origin_ + offset;

                    if (divisible_ ||
                        (access_coord.row() < extent_.row() &&
                         access_coord.column() < extent_.column())) {
                        access_ptr[access_idx] =
                                *reinterpret_cast<AccessType const*>(
                                        ref_.data() + ref_.offset(offset));
                    } else {
                        AccessType zero;
                        zero.clear();
                        access_ptr[access_idx] = zero;
                    }
                }
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
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_, cutlass::layout::ColumnMajor,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::ColumnMajor;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIteratorCanonical<
            Shape, kOperand, Element, layout::ColumnMajor, InstructionShape,
            kOpDelta, kThreads, PartitionsK_>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref,
                                        TensorCoord const& extent, int lane_id)
            : iterator_({ref.data(), ref.stride()}, extent, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.row(), tile_offset.column()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
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
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_, cutlass::layout::RowMajor,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::RowMajor;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIteratorCanonical<
            Shape, kOperand, Element, layout::RowMajor, InstructionShape,
            kOpDelta, kThreads, PartitionsK_>;

public:

    using Fragment = typename Base::Fragment;

private:
    Base iterator_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : iterator_({ref.data(), ref.stride()}, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref,
                                        TensorCoord const& extent, int lane_id)
            : iterator_({ref.data(), ref.stride()}, extent, lane_id) {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        iterator_.add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        ++iterator_;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        --iterator_;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(
                PitchLinearCoord(tile_offset.row(), tile_offset.column()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
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


}
}
}

