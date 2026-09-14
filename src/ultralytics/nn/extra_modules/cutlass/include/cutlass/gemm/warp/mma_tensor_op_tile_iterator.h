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
        typename InstructionShape_,
        int OpDelta_,
        int Threads,
        int PartitionsK_ = 1>
class MmaTensorOpMultiplicandTileIterator;


template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::TensorOpMultiplicandCongruous<
                sizeof_bits<Element_>::value, 64>,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator may only be instantiated "
                  "for A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
            sizeof_bits<Element_>::value, 64>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    static int const kPartitionsK = PartitionsK_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static_assert(
                !(Shape::kContiguous % InstructionShape::kContiguous),
                "Shape of warp-level Mma must be divisible by operator shape.");

        static int const kLdsmOpOuter = Layout::kElementsPerAccess;
        static int const kLdsmOpInner = 8;

        static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                      "Shape of warp-level mma must be divisible by LDSM's "
                      "fundamental tile size.");

        static_assert(!(Shape::kStrided % kLdsmOpInner),
                      "Shape of warp-level mma must be divisible by LDSM's "
                      "fundamental tile size.");

        static int const LdsmShapeStrided =
                InstructionShape::kStrided / kLdsmOpInner;
        static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
        using LdsmShape =
                layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

        using LdsmIterations =
                layout::PitchLinearShape<Shape::kContiguous /
                                                 Layout::kElementsPerAccess /
                                                 LdsmShapeContiguous,
                                         1>;

        static int const kGroupsPerTile =
                Shape::kStrided / InstructionShape::kStrided;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    static int const kPointerCount =
            Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous;

    using AccessType = Array<Element, Layout::kElementsPerAccess>;

    int k_group_idx_;

public:

    using Fragment =
            Array<Element,
                  Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:
    Index stride_;

    AccessType const* pointer_[kPointerCount];

    Index byte_offset_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : stride_(ref.stride(0) / Layout::kElementsPerAccess),
              byte_offset_(0),
              k_group_idx_(0) {
        int quad_pair = (lane_id >> 3);
        int quad_quad = (lane_id >> 4);
        int lane_in_quad = (lane_id & 3);
        int lane_in_quad_pair = (lane_id & 7);
        int lane_in_quad_quad = (lane_id & 15);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount; ++i) {
            int partition_contiguous_idx = -1;
            int access_contiguous_idx = -1;
            int access_strided_idx = -1;

            if (Policy::LdsmShape::kContiguous == 4) {
                partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i);
                access_contiguous_idx = (quad_pair ^ lane_in_quad);
                access_strided_idx = lane_in_quad_pair;
            } else if (Policy::LdsmShape::kContiguous == 2 &&
                       kOperand == Operand::kA) {
                partition_contiguous_idx =
                        ((lane_in_quad_pair >> 2) ^ (i >> 1));
                access_contiguous_idx =
                        (((quad_pair & 1) + ((i & 1) << 1)) ^ lane_in_quad);
                access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
            } else if (Policy::LdsmShape::kContiguous == 2 &&
                       kOperand == Operand::kB) {
                partition_contiguous_idx =
                        ((lane_in_quad_pair >> 2) ^ (i >> 1));
                access_contiguous_idx =
                        ((quad_quad + ((i & 1) << 1)) ^ lane_in_quad);
                access_strided_idx = lane_in_quad_quad;
            } else if (Policy::LdsmShape::kContiguous == 1) {
                partition_contiguous_idx =
                        ((lane_in_quad_pair >> 2) ^ (i >> 2));
                access_contiguous_idx = ((i & 3) ^ lane_in_quad);
                access_strided_idx = lane_id;
            }

            int access_contiguous =
                    partition_contiguous_idx *
                            Layout::PartitionShape::kContiguous +
                    access_contiguous_idx;

            int access_strided = access_strided_idx;

            pointer_[i] = reinterpret_cast<AccessType const*>(ref.data()) +
                          access_contiguous + access_strided * stride_;
        }
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        byte_offset_ += offset * sizeof(Element);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int contiguous_offset = tile_offset.contiguous();
        if (Shape::kContiguous ==
            Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
            if (tile_offset.contiguous() % 2) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kPointerCount / 2; ++i) {
                    AccessType const* tmp_pointer = pointer_[i];
                    pointer_[i] = pointer_[i + kPointerCount / 2];
                    pointer_[i + kPointerCount / 2] = tmp_pointer;
                }
            }
            contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
        }

        int offset = (tile_offset.strided() * InstructionShape::kStrided) *
                             stride_ * Layout::kElementsPerAccess +
                     contiguous_offset * Shape::kContiguous;

        add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        add_tile_offset({0, 1});

        if (kPartitionsK > 1) {
            ++k_group_idx_;
            if (k_group_idx_ == Policy::kGroupsPerTile) {
                k_group_idx_ = 0;
                add_tile_offset(
                        {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
            }
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                        Layout::kElementsPerAccess;

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
        Array<unsigned, Policy::LdsmShape::kCount>* fetch_ptr =
                reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
                int access_idx = c + s * Policy::LdsmIterations::kContiguous;

                AccessType const* source_ptr =
                        pointer_[c % kPointerCount] +
                        Layout::TileShape::kContiguous * (c / kPointerCount) +
                        Policy::LdsmShape::kStrided * s * stride_;

                char const* source_byte_ptr =
                        reinterpret_cast<char const*>(source_ptr) +
                        byte_offset + byte_offset_;

                cutlass::arch::ldsm<layout::ColumnMajor,
                                    Policy::LdsmShape::kCount>(
                        fetch_ptr[access_idx], source_byte_ptr);
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
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::TensorOpMultiplicandCongruous<32, 32>,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(
            kOperand == Operand::kA || kOperand == Operand::kB,
            "MmaTensorOpMultiplicandIterator may only be instantiated for "
            "A or B operands to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::TensorOpMultiplicandCongruous<32, 32>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    static int const kPartitionsK = PartitionsK_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static_assert(
                !(Shape::kContiguous % InstructionShape::kContiguous),
                "Shape of warp-level Mma must be divisible by operator shape.");

        static int const kLdsOpInner = Layout::TileShape::kStrided;
        static int const kLdsOpOuter = kThreads / kLdsOpInner;

        static_assert(!(Shape::kContiguous % kLdsOpOuter),
                      "Shape of warp-level mma must be divisible by 32bit "
                      "fundamental tile size.");

        static_assert(!(Shape::kStrided % kLdsOpInner),
                      "Shape of warp-level mma must be divisible by 32bit "
                      "fundamental tile size.");

        static int const LdsShapeContiguous =
                InstructionShape::kContiguous / kLdsOpOuter;
        static int const LdsShapeStrided =
                InstructionShape::kStrided / kLdsOpInner;
        using LdsShape =
                layout::PitchLinearShape<LdsShapeContiguous, LdsShapeStrided>;

        using LdsIterations = layout::PitchLinearShape<
                Shape::kContiguous / LdsShapeContiguous / kLdsOpOuter, 1>;

        static int const kGroupsPerTile =
                Shape::kStrided / InstructionShape::kStrided;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    static int const kPointerCount = Layout::TileShape::kContiguous *
                                     Layout::kElementsPerAccess /
                                     Policy::kLdsOpOuter;

    static int const kElementsPerAccess = 1;

    using AccessType = Element;

    int k_group_idx_;

public:

    using Fragment =
            Array<Element,
                  Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:
    Index stride_;

    AccessType const* pointer_[kPointerCount];

    Index byte_offset_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : stride_(ref.stride(0)), byte_offset_(0), k_group_idx_(0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount; ++i) {
            int access_strided = lane_id % Policy::kLdsOpInner;
            int access_contiguous = (lane_id / Policy::kLdsOpInner) +
                                    (access_strided ^ i) * Policy::kLdsOpOuter;

            pointer_[i] = reinterpret_cast<AccessType const*>(ref.data()) +
                          access_contiguous + access_strided * stride_;
        }
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        byte_offset_ += offset * sizeof(Element);

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int contiguous_offset = tile_offset.contiguous();
        if (Shape::kContiguous ==
            Layout::TileShape::kContiguous * Layout::kElementsPerAccess / 2) {
            if (tile_offset.contiguous() % 2) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kPointerCount / 2; ++i) {
                    AccessType const* tmp_pointer = pointer_[i];
                    pointer_[i] = pointer_[i + kPointerCount / 2];
                    pointer_[i + kPointerCount / 2] = tmp_pointer;
                }
            }
            contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
        }

        int offset =
                (tile_offset.strided() * InstructionShape::kStrided) * stride_ +
                contiguous_offset * Shape::kContiguous;

        add_pointer_offset(offset);

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {
        add_tile_offset({0, 1});

        if (kPartitionsK > 1) {
            ++k_group_idx_;
            if (k_group_idx_ == Policy::kGroupsPerTile) {
                k_group_idx_ = 0;
                add_tile_offset(
                        {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
            }
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() {
        byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                        kElementsPerAccess;

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
        Element* fetch_ptr = reinterpret_cast<Element*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
                CUTLASS_PRAGMA_UNROLL
                for (int ss = 0; ss < Policy::LdsShape::kStrided; ++ss) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int cc = 0; cc < Policy::LdsShape::kContiguous; ++cc) {
                        int access_idx =
                                cc +
                                (ss +
                                 (c + s * Policy::LdsIterations::kContiguous) *
                                         Policy::LdsShape::kStrided) *
                                        Policy::LdsShape::kContiguous;
                        int access_idx_contiguous =
                                cc + c * Policy::LdsShape::kContiguous;
                        int access_idx_strided =
                                (ss + s * Policy::LdsShape::kStrided) *
                                Policy::kLdsOpInner;

                        AccessType const* source_ptr =
                                pointer_[access_idx_contiguous %
                                         kPointerCount] +
                                Layout::TileShape::kContiguous *
                                        Layout::kElementsPerAccess *
                                        (access_idx_contiguous /
                                         kPointerCount) +
                                access_idx_strided * stride_;

                        char const* source_byte_ptr =
                                reinterpret_cast<char const*>(source_ptr) +
                                byte_offset + byte_offset_;

                        fetch_ptr[access_idx] =
                                *reinterpret_cast<Element const*>(
                                        source_byte_ptr);
                    }
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
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
                sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(
            kOperand == Operand::kA,
            "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
            "only be instantiated for A operand to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
            sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand,
            Element,
            layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                  int(128 / sizeof(Element_))>,
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
        cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
                sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kB,
                  "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                  "only be instantiated for B operand to warp-level Mma.");

    using Element = Element_;

    using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
            sizeof_bits<Element_>::value, int(128 / sizeof(Element_))>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand,
            Element,
            layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                  int(128 / sizeof(Element_))>,
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
        int Crosswise,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::TensorOpMultiplicandCrosswise<
                sizeof_bits<Element_>::value, Crosswise>,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(
            kOperand == Operand::kA || kOperand == Operand::kB,
            "MmaTensorOpMultiplicandIterator may only be instantiated for "
            "A or B operands to warp-level Mma.");

    using Element = Element_;

    static int const kCrosswise = Crosswise;

    using Layout = cutlass::layout::TensorOpMultiplicandCrosswise<
            sizeof_bits<Element_>::value, kCrosswise>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    static int const kPartitionsK = PartitionsK_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static_assert(
                !(Shape::kContiguous % InstructionShape::kContiguous),
                "Shape of warp-level Mma must be divisible by operator shape.");

        static int const kLdsmOpOuter = Layout::kElementsPerAccess;
        static int const kLdsmOpInner = 8;

        static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                      "Shape of warp-level mma must be divisible by LDSM's "
                      "fundamental tile size.");

        static_assert(!(Shape::kStrided % kLdsmOpInner),
                      "Shape of warp-level mma must be divisible by LDSM's "
                      "fundamental tile size.");

        static int const LdsmShapeContiguous =
                InstructionShape::kContiguous / kLdsmOpOuter;
        static int const LdsmShapeStrided =
                ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
                        ? (Shape::kStrided / kLdsmOpInner)
                        : (4 / LdsmShapeContiguous);
        using LdsmShape =
                layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

        using LdsmIterations =
                layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                                    LdsmShape::kStrided>;

        static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                          Layout::kFactor /
                                          LdsmShape::kContiguous;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    using AccessType = Array<Element, Layout::kElementsPerAccess>;

public:

    using Fragment =
            Array<Element,
                  Shape::kStrided * InstructionShape::kContiguous / kThreads>;

private:
    int sections_;

    Index stride_;

    AccessType const* pointer_;

    Index byte_offset_;

    int k_group_idx_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator()
            : pointer_(nullptr),
              sections_(0),
              stride_(0),
              byte_offset_(0),
              k_group_idx_(0) {}

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator(TensorRef const& ref, int lane_id)
            : pointer_(reinterpret_cast<AccessType const*>(ref.data())),
              sections_(ref.stride(0) / kCrosswise),
              stride_(ref.stride(0) * Layout::kFactor /
                      Layout::kElementsPerAccess),
              byte_offset_(0),
              k_group_idx_(0) {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750))
        lane_id = lane_id % (Policy::LdsmShape::kCount * Policy::kLdsmOpInner);
#endif

        int quad_quad = (lane_id >> 4);
        int quad_pair = (lane_id >> 3);
        int lane_in_pair = (lane_id & 1);
        int lane_in_quad = (lane_id & 3);
        int lane_in_quad_pair = (lane_id & 7);
        int lane_in_quad_quad = (lane_id & 15);

        int partition_contiguous_idx = -1;
        int access_contiguous_idx = -1;
        int access_strided_idx = -1;

        if (Layout::kFactor == 4) {

            int factor_in_partition =
                    (Layout::PartitionShape::kContiguous * Layout::kFactor /
                     Layout::TileShape::kContiguous);

            if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
                partition_contiguous_idx = lane_in_quad / factor_in_partition;
                access_contiguous_idx = ((lane_in_pair * factor_in_partition) ^
                                         (lane_in_quad_quad / Layout::kFactor));
                access_strided_idx = lane_id / Layout::kFactor;
            } else if (Policy::LdsmShape::kStrided ==
                               (Policy::LdsmShape::kCount / 2) &&
                       kOperand == Operand::kA) {
                partition_contiguous_idx = lane_in_quad / factor_in_partition;
                access_strided_idx = lane_in_quad_quad / Layout::kFactor;
                access_contiguous_idx =
                        ((lane_in_pair * factor_in_partition + quad_quad) ^
                         access_strided_idx);
            } else if (Policy::LdsmShape::kStrided ==
                               (Policy::LdsmShape::kCount / 2) &&
                       kOperand == Operand::kB) {
                partition_contiguous_idx = lane_in_quad / factor_in_partition;
                access_strided_idx =
                        lane_in_quad_pair / Layout::kFactor + quad_quad * 2;
                access_contiguous_idx = ((lane_in_pair * factor_in_partition +
                                          ((lane_id & 8) >> 3)) ^
                                         access_strided_idx);
            }
        } else if (Layout::kFactor == 2) {
            if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
                partition_contiguous_idx = (lane_id % Layout::kFactor);
                access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor);
                access_strided_idx = lane_id / Layout::kFactor;
            } else if (Policy::LdsmShape::kStrided ==
                               (Policy::LdsmShape::kCount / 2) &&
                       kOperand == Operand::kA) {
                partition_contiguous_idx = (lane_id % Layout::kFactor);
                access_contiguous_idx =
                        (quad_quad ^ (lane_in_quad_pair / Layout::kFactor));
                access_strided_idx = (lane_in_quad_quad / Layout::kFactor);
            } else if (Policy::LdsmShape::kStrided ==
                               (Policy::LdsmShape::kCount / 2) &&
                       kOperand == Operand::kB) {
                partition_contiguous_idx = (lane_id % Layout::kFactor);
                access_contiguous_idx = ((quad_pair & 1) ^
                                         (lane_in_quad_pair / Layout::kFactor));
                access_strided_idx = (lane_in_quad_pair + (lane_id >> 4 << 3)) /
                                     Layout::kFactor;
            } else if (Policy::LdsmShape::kContiguous ==
                       Policy::LdsmShape::kCount) {
                partition_contiguous_idx = (lane_id % Layout::kFactor);
                access_contiguous_idx =
                        (quad_pair ^ (lane_in_quad_pair / Layout::kFactor));
                access_strided_idx = lane_in_quad_pair / Layout::kFactor;
            }
        } else if (Layout::kFactor == 1) {
            if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
                partition_contiguous_idx = (lane_in_quad_pair >> 2);
                access_contiguous_idx = lane_in_quad;
                access_strided_idx = lane_id;
            } else if (Policy::LdsmShape::kStrided ==
                               (Policy::LdsmShape::kCount / 2) &&
                       kOperand == Operand::kA) {
                partition_contiguous_idx = (lane_in_quad_pair >> 2);
                access_contiguous_idx = (quad_quad ^ lane_in_quad);
                access_strided_idx = lane_in_quad_quad;
            } else if (Policy::LdsmShape::kStrided ==
                               (Policy::LdsmShape::kCount / 2) &&
                       kOperand == Operand::kB) {
                partition_contiguous_idx = (lane_in_quad_pair >> 2);
                access_contiguous_idx = ((quad_pair & 1) ^ lane_in_quad);
                access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
            } else if (Policy::LdsmShape::kContiguous ==
                       Policy::LdsmShape::kCount) {
                partition_contiguous_idx = (lane_in_quad_pair >> 2);
                access_contiguous_idx = (quad_pair ^ lane_in_quad);
                access_strided_idx = lane_in_quad_pair;
            }
        }

        int access_contiguous =
                partition_contiguous_idx * Layout::PartitionShape::kContiguous +
                access_contiguous_idx;

        int access_strided = access_strided_idx;

        byte_offset_ = (access_contiguous + access_strided * stride_) *
                       sizeof_bits<Element>::value *
                       Layout::kElementsPerAccess / 8;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_pointer_offset(LongIndex offset) {
        byte_offset_ += offset * sizeof_bits<Element>::value / 8;

        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
        int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

        byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess *
                        Policy::LdsmShape::kContiguous / 8;
        pointer_ += tile_offset.strided() * stride_ * Shape::kStrided /
                            Layout::kFactor +
                    whole_tiles * stride_ / sections_;
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset_negative(
            TensorCoord const& tile_offset) {
        int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
        int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
        if (k_groups_delta < 0) {
            whole_tiles -= 1;
            k_groups_delta += Policy::kGroupsPerTile;
        }

        if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
            byte_offset_ ^= (k_groups_delta & 1) *
                            Policy::LdsmShape::kContiguous *
                            sizeof_bits<Element>::value *
                            Layout::kElementsPerAccess / 8;
        }
        if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
            byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) *
                            Policy::LdsmShape::kContiguous *
                            sizeof_bits<Element>::value *
                            Layout::kElementsPerAccess / 8;
        }
        if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
            byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) *
                            Policy::LdsmShape::kContiguous *
                            sizeof_bits<Element>::value *
                            Layout::kElementsPerAccess / 8;
        }

        k_group_idx_ += k_groups_delta;
        whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
        k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

        pointer_ += tile_offset.strided() * stride_ * Shape::kStrided /
                            Layout::kFactor +
                    whole_tiles * stride_ / sections_;
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator++() {



        if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
            int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                               ? 3
                               : (((Policy::kGroupsPerTile / kPartitionsK) == 4)
                                          ? 1
                                          : 0);

            if (((k_group_idx_ & mask) % 2) == 0)
                byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                                sizeof_bits<Element>::value *
                                Layout::kElementsPerAccess / 8;
            else if ((k_group_idx_ & mask) == 1)
                byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                                sizeof_bits<Element>::value *
                                Layout::kElementsPerAccess / 8;
            else if ((k_group_idx_ & mask) == 3)
                byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                                sizeof_bits<Element>::value *
                                Layout::kElementsPerAccess / 8;
        }

        k_group_idx_++;

        if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
            k_group_idx_ = 0;
            add_tile_offset({Policy::kGroupsPerTile, 0});
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpMultiplicandTileIterator& operator--() { assert(0); }

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
        Array<unsigned, Policy::LdsmShape::kCount>* fetch_ptr =
                reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
                int access_idx = c + s * Policy::LdsmIterations::kContiguous;

                AccessType const* source_ptr =
                        pointer_ + Policy::LdsmShape::kContiguous * c +
                        Policy::kLdsmOpInner / Layout::kFactor *
                                Policy::LdsmShape::kStrided * s * stride_;

                char const* source_byte_ptr =
                        reinterpret_cast<char const*>(source_ptr) +
                        byte_offset + byte_offset_;

                cutlass::arch::ldsm<layout::RowMajor,
                                    Policy::LdsmShape::kCount>(
                        fetch_ptr[access_idx], source_byte_ptr);
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

        byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

        load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
        k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
    }
};


template <
        typename Shape_,
        Operand Operand_,
        typename Element_,
        typename InstructionShape_,
        int OpDelta_,
        int Crosswise,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
                sizeof_bits<Element_>::value, Crosswise>,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(
            kOperand == Operand::kB,
            "MmaTensorOpMultiplicandIterator for ColumnMajor Crosswise may "
            "only be instantiated for B operand to warp-level Mma.");

    using Element = Element_;

    static int const kCrosswise = Crosswise;

    using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<Element_>::value, kCrosswise>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand,
            Element,
            layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                  kCrosswise>,
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

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset_negative(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset_negative(
                {tile_offset.row(), tile_offset.column()});

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
        int Crosswise,
        int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
        Shape_, Operand_, Element_,
        cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
                sizeof_bits<Element_>::value, Crosswise>,
        InstructionShape_, OpDelta_, 32, PartitionsK_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand_;

    static_assert(kOperand == Operand::kA,
                  "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                  "only be instantiated for A operand to warp-level Mma.");

    using Element = Element_;

    static int const kCrosswise = Crosswise;

    using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
            sizeof_bits<Element_>::value, kCrosswise>;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    using Base = MmaTensorOpMultiplicandTileIterator<
            layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand,
            Element,
            layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                  kCrosswise>,
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

    CUTLASS_DEVICE
    MmaTensorOpMultiplicandTileIterator& add_tile_offset_negative(
            TensorCoord const& tile_offset) {
        iterator_.add_tile_offset_negative(
                {tile_offset.column(), tile_offset.row()});

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
        typename Element_,
        typename Layout_,
        typename InstructionShape_,
        typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator;


template <
        typename Shape_,
        typename Element_,
        typename InstructionShape_,
        typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<Shape_, Element_,
                                         cutlass::layout::RowMajor,
                                         InstructionShape_, OpDelta_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kC;

    using Element = Element_;

    using Layout = cutlass::layout::RowMajor;

    using InstructionShape = InstructionShape_;

    using OpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static bool const kDivisible = !(Shape::kRow % InstructionShape::kM) &&
                                       !(Shape::kColumn % InstructionShape::kN);

        static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
                      "Layouts must be defined for logical MatrixCoord "
                      "coordinate space.");

        using MmaIterations =
                MatrixShape<(Shape::kRow + InstructionShape::kM - 1) /
                                    InstructionShape::kM,
                            (Shape::kColumn + InstructionShape::kN - 1) /
                                    InstructionShape::kN>;
    };

private:
    static int const kElementsPerAccess = InstructionShape::kN / 4;
    static int const kRowsPerTile = 8;
    static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

    using Fragment = Array<Element, Policy::MmaIterations::kCount *
                                            InstructionShape::kMN / kThreads>;

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

                        frag[mma_accum_start + row * kElementsPerAccess + col] =
                                offset_ref.at({accum_m, accum_n});
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

                        offset_ref.at({accum_m, accum_n}) = frag[idx];
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
        typename Element_,
        typename InstructionShape_,
        typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<Shape_, Element_,
                                         cutlass::layout::ColumnMajor,
                                         InstructionShape_, OpDelta_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kC;

    using Element = Element_;

    using Layout = cutlass::layout::ColumnMajor;

    using InstructionShape = InstructionShape_;

    using OpDelta = OpDelta_;

    static int const kThreads = 32;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static bool const kDivisible = !(Shape::kRow % InstructionShape::kM) &&
                                       !(Shape::kColumn % InstructionShape::kN);

        static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
                      "Layouts must be defined for logical MatrixCoord "
                      "coordinate space.");

        using MmaIterations =
                MatrixShape<(Shape::kRow + InstructionShape::kM - 1) /
                                    InstructionShape::kM,
                            (Shape::kColumn + InstructionShape::kN - 1) /
                                    InstructionShape::kN>;
    };

private:
    static int const kElementsPerAccess = InstructionShape::kN / 4;
    static int const kRowsPerTile = 8;
    static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

    using Fragment = Array<Element, Policy::MmaIterations::kCount *
                                            InstructionShape::kMN / kThreads>;

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
                        int idx = mma_accum_start + row * kElementsPerAccess +
                                  col;

                        frag[idx] = offset_ref.at({accum_m, accum_n});
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

                        offset_ref.at({accum_m, accum_n}) = frag[idx];
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
        typename Element_,
        typename InstructionShape_,
        typename OpDelta_,
        int InterleavedN>
class MmaTensorOpAccumulatorTileIterator<
        Shape_, Element_, cutlass::layout::ColumnMajorInterleaved<InterleavedN>,
        InstructionShape_, OpDelta_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kC;

    using Element = Element_;

    using Layout = cutlass::layout::ColumnMajorInterleaved<InterleavedN>;

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
    static int const kElementsPerAccess = 2;

public:

    using AccessType = Array<Element, kElementsPerAccess>;

    using Fragment = Array<Element, Shape::kCount / kThreads>;

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

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                int accum_m = mma_m * InstructionShape::kM;
                int accum_n = mma_n * InstructionShape::kN;

                int idx = mma_m + mma_n * Policy::MmaIterations::kRow;

                AccessType* access_ptr = reinterpret_cast<AccessType*>(
                        offset_ref.data() +
                        offset_ref.offset(TensorCoord(accum_m, accum_n)));

                frag_ptr[idx] = access_ptr[0];
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

        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                int accum_m = mma_m * InstructionShape::kM;
                int accum_n = mma_n * InstructionShape::kN;

                int idx = mma_m + mma_n * Policy::MmaIterations::kRow;

                AccessType* access_ptr = reinterpret_cast<AccessType*>(
                        offset_ref.data() +
                        offset_ref.offset(TensorCoord(accum_m, accum_n)));

                access_ptr[0] = frag_ptr[idx];
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
        typename Element_,
        typename InstructionShape_,
        typename OpDelta_,
        int InterleavedN>
class MmaTensorOpAccumulatorTileIterator<
        Shape_, Element_, cutlass::layout::TensorNCxHWx<InterleavedN>,
        InstructionShape_, OpDelta_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kC;

    using Element = int8_t;

    using Layout = cutlass::layout::TensorNCxHWx<InterleavedN>;

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

        static int const kStridedPerSTG = 8;

        static int const kPackedFactor = Shape::kColumn / 32;

        using MmaIterations = MatrixShape<Shape::kRow / kStridedPerSTG,
                                          Shape::kColumn / InterleavedN>;
    };

private:
    static int const kElementsPerAccess = InterleavedN / 4;

public:

    struct alignas((kElementsPerAccess * sizeof_bits<Element>::value /
                    8)) AccessType {
        Array<Element, kElementsPerAccess> storage;
    };

    using Fragment = Array<int32_t, Shape::kCount / kThreads>;

private:
    TensorRef ref_;

    LongIndex global_offset_row_;

    LongIndex global_offset_col_;

    TensorCoord extent_;

    float alpha_;

    float beta_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator(TensorRef const& ref, int const lane_id,
                                       TensorCoord extent, float alpha = 1.0f,
                                       float beta = 0.0f)
            : ref_(ref), extent_(extent), alpha_(alpha), beta_(beta) {
        int quad = (lane_id >> 2);
        int lane_in_quad = (lane_id & 3);

        global_offset_row_ = quad;

        global_offset_col_ = lane_in_quad * kElementsPerAccess;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator& add_pointer_offset(LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpAccumulatorTileIterator& add_tile_offset(
            MatrixCoord const& tile_offset) {
        global_offset_row_ += tile_offset.row() * Shape::kRow;

        global_offset_col_ += tile_offset.column() * Shape::kColumn;

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
    void load(Fragment& frag) const { load_with_pointer_offset(frag); }

    CUTLASS_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset)
            const {

        TensorRef offset_ref(ref_);
        offset_ref.add_pointer_offset(pointer_offset);

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kN; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kM; ++mma_m) {
                int accum_m = mma_m * InstructionShape::kM;
                int accum_n = mma_n * InstructionShape::kN;

                int idx = mma_m + mma_n * Policy::MmaIterations::kM;

                AccessType* access_ptr = reinterpret_cast<AccessType*>(
                        offset_ref.data() + accum_m * offset_ref.stride(0) +
                        accum_n);

                frag_ptr[idx] = access_ptr[0];
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

        Array<float, Shape::kCount / kThreads> output_frag_f;
        Array<Element, Shape::kCount / kThreads> output_frag;

        LongIndex pq = extent_.h() * extent_.w();

        LongIndex extent_row = extent_.n() * pq;
        LongIndex extent_col = extent_.c();

        LongIndex k_major = (global_offset_col_ / InterleavedN) * pq;
        Index k_minor = global_offset_col_ % InterleavedN;
        LongIndex k_offset = k_major * InterleavedN + k_minor;
        LongIndex k_offset_delta = pq * InterleavedN;

        LongIndex stride_n = pq * extent_.c();

        Index n;
        LongIndex pq_rem;

        unsigned int pq_mul, pq_shr;
        find_divisor(pq_mul, pq_shr, pq);

        if (beta_ == 0.0f) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < frag.size(); ++i) {
                output_frag_f[i] = frag[i];
            }

            if (InstructionShape::kM == Policy::kStridedPerSTG) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < frag.size(); ++i) {
                    output_frag[i] = (Element)(output_frag_f[i] * alpha_);
                }
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < frag.size(); ++i) {
                    int map_i = (i / (16 * Policy::kPackedFactor)) *
                                        (16 * Policy::kPackedFactor) +
                                (i % (8 * Policy::kPackedFactor)) / 2 * 4 +
                                (i % (8 * Policy::kPackedFactor)) % 2 +
                                (i / (8 * Policy::kPackedFactor)) % 2 * 2;
                    output_frag[i] = (Element)(output_frag_f[map_i] * alpha_);
                }
            }

            AccessType const* frag_ptr =
                    reinterpret_cast<AccessType const*>(&output_frag);

            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                int accum_m = mma_m * Policy::kStridedPerSTG;

                fast_divmod(n, pq_rem, global_offset_row_ + accum_m, pq, pq_mul,
                            pq_shr);
                LongIndex offset_m =
                        n * stride_n + k_offset + pq_rem * InterleavedN;

                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn;
                     ++mma_n) {
                    int accum_n = mma_n * InterleavedN;

                    int idx = mma_n + mma_m * Policy::MmaIterations::kColumn;

                    if ((global_offset_row_ + accum_m < extent_row) &&
                        (global_offset_col_ + accum_n < extent_col)) {
                        AccessType* access_ptr = reinterpret_cast<AccessType*>(
                                offset_ref.data() + offset_m +
                                mma_n * k_offset_delta);

                        access_ptr[0] = frag_ptr[idx];
                    }
                }
            }
        } else {
            if (InstructionShape::kM == Policy::kStridedPerSTG) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < frag.size(); ++i) {
                    output_frag_f[i] = frag[i];
                }
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < frag.size(); ++i) {
                    int map_i = (i / (16 * Policy::kPackedFactor)) *
                                        (16 * Policy::kPackedFactor) +
                                (i % (8 * Policy::kPackedFactor)) / 2 * 4 +
                                (i % (8 * Policy::kPackedFactor)) % 2 +
                                (i / (8 * Policy::kPackedFactor)) % 2 * 2;
                    output_frag_f[i] = frag[map_i];
                }
            }

            AccessType const* frag_ptr =
                    reinterpret_cast<AccessType const*>(&output_frag);

            Array<Element, kElementsPerAccess> ref_frag;
            AccessType* ref_frag_ptr = reinterpret_cast<AccessType*>(&ref_frag);

            CUTLASS_PRAGMA_UNROLL
            for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
                int accum_m = mma_m * Policy::kStridedPerSTG;

                fast_divmod(n, pq_rem, global_offset_row_ + accum_m, pq, pq_mul,
                            pq_shr);
                LongIndex offset_m =
                        n * stride_n + k_offset + pq_rem * InterleavedN;

                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn;
                     ++mma_n) {
                    int accum_n = mma_n * InterleavedN;

                    int idx = mma_n + mma_m * Policy::MmaIterations::kColumn;

                    if ((global_offset_row_ + accum_m < extent_row) &&
                        (global_offset_col_ + accum_n < extent_col)) {
                        AccessType* access_ptr = reinterpret_cast<AccessType*>(
                                offset_ref.data() + offset_m +
                                mma_n * k_offset_delta);

                        ref_frag_ptr[0] = access_ptr[0];

                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kElementsPerAccess; ++i) {
                            output_frag[idx * kElementsPerAccess + i] = Element(
                                    alpha_ * output_frag_f
                                                     [idx * kElementsPerAccess +
                                                      i] +
                                    beta_ * ref_frag[i]);
                        }

                        access_ptr[0] = frag_ptr[idx];
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


}
}
}

