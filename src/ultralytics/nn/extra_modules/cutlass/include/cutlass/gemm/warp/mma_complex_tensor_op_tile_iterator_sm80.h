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
        cutlass::layout::TensorOpMultiplicandCongruous128b, InstructionShape_,
        OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    static_assert(!(Shape::kContiguous % 8) && !(Shape::kStrided % 4),
                  "Divisibility.");

    static_assert(sizeof_bits<Element_>::value == 128,
                  "This is specialized for 128b accesses.");

    using Element = Element_;

    using Layout = cutlass::layout::TensorOpMultiplicandCongruous128b;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    static int const kPartitionsK = PartitionsK_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static int const kElementsPerAccess = 1;

    struct Policy {
        using Delta = layout::PitchLinearShape<8, 4>;

        using Iterations = layout::PitchLinearShape<
                Shape::kContiguous / Delta::kContiguous,
                InstructionShape::kStrided / Delta::kStrided>;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    using AccessType = AlignedArray<Element, kElementsPerAccess, 16>;

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
            : stride_(ref.stride(0) / kElementsPerAccess), byte_offset_(0) {
        int quad_pair = lane_id / 8;
        int quad = lane_id / 4;
        int lane = lane_id % 4;

        int row = (quad & 1) * 4 + (lane ^ quad_pair);

        byte_offset_ = (row + quad_pair * stride_) * sizeof(AccessType);

        pointer_ = reinterpret_cast<AccessType const*>(ref.data());
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        pointer_ += offset;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int offset =
                (tile_offset.contiguous() * Shape::kContiguous) +
                (tile_offset.strided() * InstructionShape::kStrided * stride_);

        add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        pointer_ += stride_ * InstructionShape::kStrided;

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
                tile_offset.contiguous() * Shape::kContiguous +
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
        cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand,
            Element, layout::TensorOpMultiplicandCongruous128b,
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
        add_tile_offset(layout::PitchLinearCoord(tile_offset.column(),
                                                 tile_offset.row()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(layout::PitchLinearCoord(-tile_offset.column(),
                                                 -tile_offset.row()));
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
        cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout =
            cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand,
            Element, layout::TensorOpMultiplicandCongruous128b,
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
        add_tile_offset(layout::PitchLinearCoord(tile_offset.row(),
                                                 tile_offset.column()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(layout::PitchLinearCoord(-tile_offset.row(),
                                                 -tile_offset.column()));
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
        typename RealElement,
        typename InstructionShape_,
        typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<Shape_, complex<RealElement>,
                                         cutlass::layout::RowMajor,
                                         InstructionShape_, OpDelta_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kC;

    using Element = complex<RealElement>;

    using Layout = cutlass::layout::RowMajor;

    using InstructionShape = InstructionShape_;

    using OpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static_assert(
                !(Shape::kRow % InstructionShape::kM) &&
                        !(Shape::kColumn % InstructionShape::kN),
                "Shape of warp-level Mma must be divisible by operator shape.");

        static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
                      "Layouts must be defined for logical MatrixCoord "
                      "coordinate space.");

        using MmaIterations =
                MatrixShape<Shape::kRow / InstructionShape::kM,
                            Shape::kColumn / InstructionShape::kN>;
    };

private:
    static int const kElementsPerAccess = InstructionShape::kN / 4;
    static int const kRowsPerTile = 8;
    static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

    using Fragment = Array<RealElement, Shape::kCount / kThreads * 2>;

    static int const kRealIndex = 0;
    static int const kImaginaryIndex = Shape::kCount / kThreads;

private:
    TensorRef ref_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator(TensorRef const& ref, int lane_id)
            : ref_(ref) {
        int quad = (lane_id >> 2);
        int lane_in_quad = (lane_id & 3);

        MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

        ref_.add_coord_offset(lane_offset);
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator& add_pointer_offset(LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        ref_.add_coord_offset(tile_offset *
                              make_Coord(Shape::kRow, Shape::kColumn));

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator& operator++() {
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator& operator--() {
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpAccumulatorTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpAccumulatorTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(-tile_offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset)
            const {

        TensorRef offset_ref(ref_);
        offset_ref.add_pointer_offset(pointer_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                int mma_accum_start =
                        kAccumulatorRows * kElementsPerAccess *
                        (mma_n * Policy::MmaIterations::kRow + mma_m);

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < kAccumulatorRows; ++row) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int col = 0; col < kElementsPerAccess; ++col) {
                        int accum_m =
                                mma_m * InstructionShape::kM * OpDelta::kRow +
                                row * kRowsPerTile;
                        int accum_n = mma_n * InstructionShape::kN *
                                              OpDelta::kColumn +
                                      col;

                        Element z = offset_ref.at({accum_m, accum_n});

                        frag[mma_accum_start + row * kElementsPerAccess + col +
                             kRealIndex] = z.real();
                        frag[mma_accum_start + row * kElementsPerAccess + col +
                             kImaginaryIndex] = z.imag();
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

    CUTLASS_DEVICE
    void load(Fragment& frag,
              TensorCoord const& tile_offset)
            const {

        load(frag, tile_offset, 0);
    }

    CUTLASS_DEVICE
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

    CUTLASS_DEVICE
    void store_with_pointer_offset(
            Fragment const& frag,
            Index pointer_offset)
            const {

        TensorRef offset_ref(ref_);
        offset_ref.add_pointer_offset(pointer_offset);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                int mma_accum_start =
                        kAccumulatorRows * kElementsPerAccess *
                        (mma_n * Policy::MmaIterations::kRow + mma_m);

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < kAccumulatorRows; ++row) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int col = 0; col < kElementsPerAccess; ++col) {
                        int accum_m =
                                mma_m * InstructionShape::kM * OpDelta::kRow +
                                row * kRowsPerTile;
                        int accum_n = mma_n * InstructionShape::kN *
                                              OpDelta::kColumn +
                                      col;
                        int idx = mma_accum_start + row * kElementsPerAccess +
                                  col;

                        Element z(frag[kRealIndex + idx],
                                  frag[kImaginaryIndex + idx]);

                        offset_ref.at({accum_m, accum_n}) = z;
                    }
                }
            }
        }
    }

    CUTLASS_DEVICE
    void store_with_byte_offset(
            Fragment const& frag,
            Index byte_offset) const {

        store_with_pointer_offset(byte_offset / sizeof(Element));
    }

    CUTLASS_DEVICE
    void store(Fragment& frag,
               TensorCoord const& tile_offset)
            const {

        store(frag, tile_offset, 0);
    }

    CUTLASS_DEVICE
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
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::TensorOpMultiplicandCrosswise128x4, InstructionShape_,
        OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    static_assert(!(Shape::kContiguous % 4) && !(Shape::kStrided % 8),
                  "Divisibility.");

    static_assert(sizeof_bits<Element_>::value == 128,
                  "This is specialized for 128b accesses.");

    using Element = Element_;

    using Layout = cutlass::layout::TensorOpMultiplicandCrosswise128x4;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    static int const kPartitionsK = PartitionsK_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static int const kElementsPerAccess = 1;

    struct Policy {
        using Delta = layout::PitchLinearShape<4, 8>;

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

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : stride_(ref.stride(0) / kElementsPerAccess), byte_offset_(0) {
        int quad = lane_id / 4;
        int liq = lane_id % 4;

        int c = liq + (quad & 1) * 4;
        int s = (quad / 2);

        byte_offset_ = (c + s * stride_) * sizeof(AccessType);

        pointer_ = reinterpret_cast<AccessType const*>(ref.data());
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        pointer_ += offset;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int offset =
                (tile_offset.contiguous() * InstructionShape::kContiguous) *
                        stride_ +
                (tile_offset.strided() * Shape::kStrided);

        add_pointer_offset(offset);

        byte_offset_ ^= (tile_offset.contiguous() & 1) * 4 * sizeof(AccessType);

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        pointer_ += stride_ * InstructionShape::kContiguous;

        byte_offset_ ^= 4 * sizeof(AccessType);

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
                int access_idx = s + c * Policy::Iterations::kStrided;

                AccessType const* source_ptr =
                        pointer_ + Policy::Delta::kContiguous * c * stride_ +
                        Policy::Delta::kStrided * s;

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
        Index pointer_offset = tile_offset.contiguous() *
                                       InstructionShape::kContiguous * stride_ +
                               tile_offset.strided() * Shape::kStrided;

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
        cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand,
            Element, layout::TensorOpMultiplicandCrosswise128x4,
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
        add_tile_offset(layout::PitchLinearCoord(tile_offset.column(),
                                                 tile_offset.row()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(layout::PitchLinearCoord(-tile_offset.column(),
                                                 -tile_offset.row()));
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
        cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout =
            cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand,
            Element, layout::TensorOpMultiplicandCrosswise128x4,
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
        add_tile_offset(layout::PitchLinearCoord(tile_offset.row(),
                                                 tile_offset.column()));
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator-=(
            TensorCoord const& tile_offset) {
        add_tile_offset(layout::PitchLinearCoord(-tile_offset.row(),
                                                 -tile_offset.column()));
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
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, cutlass::complex<float>,
        cutlass::layout::TensorOpMultiplicandCongruous64b, InstructionShape_,
        OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    static_assert(!(Shape::kContiguous % 16) && !(Shape::kStrided % 8),
                  "Divisibility.");

    using Element = cutlass::complex<float>;

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
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, complex<float>,
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

    static_assert(sizeof_bits<complex<float>>::value == 64,
                  "This is specialized for 64b accesses.");

    using Element = complex<float>;

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

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        pointer_ += stride_ * InstructionShape::kContiguous;

        byte_offset_ ^= 0x40;

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
                int access_idx = c * Policy::Iterations::kStrided + s;

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

        CUTLASS_PRAGMA_UNROLL
        for (int i = Fragment::kElements / 2; i < Fragment::kElements; i += 2) {
            Element tmp = exchange_ptr[i];
            exchange_ptr[i] = exchange_ptr[i + 1];
            exchange_ptr[i + 1] = tmp;
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

}
}
}

