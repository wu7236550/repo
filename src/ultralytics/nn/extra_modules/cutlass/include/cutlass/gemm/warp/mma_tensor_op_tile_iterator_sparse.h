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
        typename Element_,
        typename Layout_,
        typename InstructionShape_,
        int OpDelta_,
        int Threads,
        int PartitionsK_ = 1>
class SparseMmaTensorOpMetaTileIterator {
public:
    using Shape = Shape_;

    using Element = Element_;

    using Layout = Layout_;

    using InstructionShape = InstructionShape_;

    static int const kOpDelta = OpDelta_;

    static int const kThreads = 32;

    static int const kPartitionsK = PartitionsK_;

    static int const kSparse = 2;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    struct Policy {
        static_assert(
                !(Shape::kColumn % InstructionShape::kColumn),
                "Shape of warp-level Mma must be divisible by operator shape.");

        static int const kElementsPerAccess = 128 / sizeof_bits<Element>::value;

        static int const kLdsmOpOuter = InstructionShape::kColumn;
        static int const kLdsmOpInner = 8 * kElementsPerAccess / kLdsmOpOuter;

        static_assert(!(Shape::kColumn % kLdsmOpOuter),
                      "Shape of warp-level mma must be divisible by LDSM's "
                      "fundamental tile size.");

        static_assert(!(Shape::kRow % kLdsmOpInner),
                      "Shape of warp-level mma must be divisible by LDSM's "
                      "fundamental tile size.");

        static int const LdsmShapeColumn =
                InstructionShape::kColumn / kLdsmOpOuter;
        static int const LdsmShapeRow =
                ((4 / LdsmShapeColumn * kLdsmOpInner) > Shape::kRow)
                        ? (Shape::kRow / kLdsmOpInner)
                        : (4 / LdsmShapeColumn);
        using LdsmShape =
                layout::PitchLinearShape<LdsmShapeRow, LdsmShapeColumn>;

        using LdsmIterations = layout::PitchLinearShape<
                Shape::kRow / kLdsmOpInner / LdsmShapeRow, 1>;

        static int const kGroupsPerTile =
                Shape::kColumn / InstructionShape::kColumn;
    };

private:
    static_assert(kOpDelta == 1,
                  "Alternative arrangements not supported at present.");

    using AccessType = Array<Element, Policy::kElementsPerAccess>;

public:

    using Fragment =
            Array<Element, Shape::kRow * InstructionShape::kColumn / kThreads>;

private:
    Index stride_;

    AccessType const* pointer_;

    Index byte_offset_;

    int k_group_idx_;

public:
    CUTLASS_HOST_DEVICE
    SparseMmaTensorOpMetaTileIterator()
            : pointer_(nullptr), stride_(0), byte_offset_(0), k_group_idx_(0) {}

    CUTLASS_DEVICE
    SparseMmaTensorOpMetaTileIterator(TensorRef const& ref, int lane_id)
            : pointer_(reinterpret_cast<AccessType const*>(ref.data())),
              stride_(ref.stride(0) / Policy::kElementsPerAccess),
              byte_offset_(0),
              k_group_idx_(0) {
        int access_contiguous =
                (lane_id % (Shape::kRow / Policy::kElementsPerAccess));
        int access_strided =
                (lane_id / (Shape::kRow / Policy::kElementsPerAccess));

        byte_offset_ = (access_contiguous + access_strided * stride_) *
                       sizeof_bits<Element>::value *
                       Policy::kElementsPerAccess / 8;
    }

    CUTLASS_DEVICE
    SparseMmaTensorOpMetaTileIterator& add_pointer_offset(LongIndex offset) {
        byte_offset_ += offset * sizeof_bits<Element>::value / 8;

        return *this;
    }

    CUTLASS_DEVICE
    SparseMmaTensorOpMetaTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        int offset = tile_offset.row() * Shape::kRow +
                     tile_offset.column() * InstructionShape::kColumn *
                             stride_ * Policy::kElementsPerAccess;

        add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_DEVICE
    SparseMmaTensorOpMetaTileIterator& operator++() {
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
    SparseMmaTensorOpMetaTileIterator& operator--() {
        byte_offset_ -= stride_ * InstructionShape::kColumn *
                        sizeof_bits<Element>::value *
                        Policy::kElementsPerAccess / 8;
    }

    CUTLASS_DEVICE SparseMmaTensorOpMetaTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    SparseMmaTensorOpMetaTileIterator& operator-=(
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
                        pointer_ +
                        Policy::LdsmShape::kContiguous * Policy::kLdsmOpInner *
                                c +
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
                tile_offset.contiguous() * Shape::kRow /
                        Layout::kElementsPerAccess +
                tile_offset.strided() * InstructionShape::kColumn * stride_;

        byte_offset += sizeof(AccessType) * pointer_offset;

        load_with_byte_offset(frag, byte_offset);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};

}
}
}

