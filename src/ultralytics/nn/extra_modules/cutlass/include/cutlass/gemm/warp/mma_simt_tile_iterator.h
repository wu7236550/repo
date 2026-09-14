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
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename Shape_,
        Operand Operand,
        typename Element_,
        typename Layout_,
        typename Policy_,
        int PartitionsK = 1,
        int PartitionGroupSize = 1>
class MmaSimtTileIterator;


template <
        typename Shape_,
        typename Element_,
        typename Policy_,
        int PartitionsK,
        int PartitionGroupSize>
class MmaSimtTileIterator<Shape_, Operand::kA, Element_, layout::ColumnMajor,
                          Policy_, PartitionsK, PartitionGroupSize> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kA;

    using Element = Element_;

    using Layout = layout::ColumnMajor;

    using Policy = Policy_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;


    static_assert(!(Shape::kRow % Policy::WarpShape::kRow),
                  "The warp-level GEMM M size must be divisible by the number "
                  "of threads arranged along the M dimension.");

    static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
    static_assert(Shape::kColumn > 0,
                  "Shape::kColumn must be greater than zero.");
    static_assert(Policy::WarpShape::kRow > 0,
                  "Policy::WarpShape::kRow must be greater than zero.");
    static_assert(
            Shape::kRow / Policy::WarpShape::kRow > 0,
            "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

    using ThreadShape =
            MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

    static_assert(
            !(ThreadShape::kRow % Policy::LaneMmaShape::kM),
            "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

    using Iterations = MatrixShape<ThreadShape::kRow / Policy::LaneMmaShape::kM,
                                   ThreadShape::kColumn>;

    using Fragment = Array<Element, ThreadShape::kCount>;

private:
    cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kM>,
                       layout::ColumnMajor>
            ref_;

public:
    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator(TensorRef ref, int lane_id) {
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

        MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
                                  MatrixCoord(Policy::LaneMmaShape::kM, 0);

        ref.add_coord_offset(lane_offset);

        ref_.reset(reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM>*>(
                           ref.data()),
                   ref.stride(0) / Policy::LaneMmaShape::kM);
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_pointer_offset(LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_tile_offset(TensorCoord const& coord) {
        ref_.add_coord_offset(
                {coord.row() * Shape::kRow / Policy::LaneMmaShape::kM,
                 coord.column() * Shape::kColumn});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator++() {
        ref_.add_coord_offset({0, Shape::kColumn});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator--() {
        ref_.add_coord_offset({0, -Shape::kColumn});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kM>* dst_ptr =
                reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kColumn; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Iterations::kRow; ++m) {
                dst_ptr[m + k * Iterations::kRow] =
                        *(ref_.data() +
                          ref_.offset({m * Policy::WarpShape::kRow, k}) +
                          pointer_offset / Policy::LaneMmaShape::kM);
            }
        }
    }
    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag,
                                   Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kM> const* src_ptr =
                reinterpret_cast<Array<Element, Policy::LaneMmaShape::kM>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kN; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Iterations::kM; ++m) {
                *(ref_.data() + ref_.offset(m * Policy::WarpShape::kM, k) +
                  pointer_offset / Policy::LaneMmaShape::kM) =
                        src_ptr[m + k * Iterations::kM];
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


template <
        typename Shape_,
        typename Element_,
        typename Policy_,
        int PartitionsK,
        int PartitionGroupSize>
class MmaSimtTileIterator<Shape_, Operand::kB, Element_, layout::RowMajor,
                          Policy_, PartitionsK, PartitionGroupSize> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kB;

    using Element = Element_;

    using Layout = layout::RowMajor;

    using Policy = Policy_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;


    static_assert(!(Shape::kColumn % Policy::WarpShape::kColumn),
                  "The warp-level GEMM N size must be divisible by the number "
                  "of threads arranged along the N dimension.");

    static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
    static_assert(Shape::kColumn > 0,
                  "Shape::kColumn must be greater than zero.");
    static_assert(Policy::WarpShape::kColumn > 0,
                  "Policy::WarpShape::kColumn must be greater than zero.");
    static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0,
                  "Shape::kColumn / Policy::WarpShape::kColumn must be greater "
                  "than zero.");

    using ThreadShape =
            MatrixShape<Shape::kRow,
                        Shape::kColumn / Policy::WarpShape::kColumn>;

    static_assert(
            !(ThreadShape::kColumn % Policy::LaneMmaShape::kN),
            "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

    using Iterations =
            MatrixShape<ThreadShape::kRow,
                        ThreadShape::kColumn / Policy::LaneMmaShape::kN>;

    using Fragment = Array<Element, ThreadShape::kCount>;

private:
    cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kN>,
                       layout::RowMajor>
            ref_;

public:
    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator(TensorRef ref, int lane_id) {
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

        MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
                                  MatrixCoord(0, Policy::LaneMmaShape::kN);

        ref.add_coord_offset(lane_offset);

        ref_.reset(reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN>*>(
                           ref.data()),
                   ref.stride(0) / Policy::LaneMmaShape::kN);
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_pointer_offset(LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_tile_offset(TensorCoord const& coord) {
        ref_.add_coord_offset(
                {coord.row() * Shape::kRow,
                 coord.column() * Shape::kColumn / Policy::LaneMmaShape::kN});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator++() {
        ref_.add_coord_offset({Shape::kRow, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator--() {
        ref_.add_coord_offset({-Shape::kRow, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kN>* dst_ptr =
                reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kRow; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Iterations::kColumn; ++n) {
                dst_ptr[n + k * Iterations::kColumn] =
                        *(ref_.data() +
                          ref_.offset({k, n * Policy::WarpShape::kColumn}) +
                          pointer_offset / Policy::LaneMmaShape::kN);
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag,
                                   Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kN> const* src_ptr =
                reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kM; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Iterations::kN; ++n) {
                *(ref_.data() + ref_.offset({k, n * Policy::WarpShape::kN}) +
                  pointer_offset / Policy::LaneMmaShape::kN) =
                        src_ptr[n + k * Iterations::kN];
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag, Index pointer_offset) const {
        store_with_pointer_offset(frag, 0);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};


template <
        typename Shape_,
        typename Element_,
        typename Policy_>
class MmaSimtTileIterator<Shape_, Operand::kC, Element_, layout::ColumnMajor,
                          Policy_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kC;

    using Element = Element_;

    using Layout = layout::ColumnMajor;

    using Policy = Policy_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;


    static_assert((!(Shape::kRow % Policy::WarpShape::kRow)) &&
                          (!(Shape::kColumn % Policy::WarpShape::kColumn)),
                  "Warp-level GEMM shape must be divisible by the arrangement "
                  "of threads in the warp.");

    static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
    static_assert(Shape::kColumn > 0,
                  "Shape::kColumn must be greater than zero.");
    static_assert(Policy::WarpShape::kRow > 0,
                  "Policy::WarpShape::kRow must be greater than zero.");
    static_assert(Policy::WarpShape::kColumn > 0,
                  "Policy::WarpShape::kColumn must be greater than zero.");
    static_assert(
            Shape::kRow / Policy::WarpShape::kRow > 0,
            "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");
    static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0,
                  "Shape::kColumn / Policy::WarpShape::kColumn must be greater "
                  "than zero.");

    using ThreadShape =
            MatrixShape<Shape::kRow / Policy::WarpShape::kRow,
                        Shape::kColumn / Policy::WarpShape::kColumn>;

    static_assert((!(ThreadShape::kRow % Policy::LaneMmaShape::kM)) &&
                          (!(ThreadShape::kColumn % Policy::LaneMmaShape::kN)),
                  "Warp-level GEMM shape must be divisible by the arrangement "
                  "of threads in the warp.");

    using Iterations =
            MatrixShape<ThreadShape::kRow / Policy::LaneMmaShape::kM,
                        ThreadShape::kColumn / Policy::LaneMmaShape::kN>;

    using Delta =
            MatrixShape<Policy::WarpShape::kRow * Policy::LaneMmaShape::kM,
                        Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN>;

    using Fragment = Array<Element, ThreadShape::kCount>;

private:
    TensorRef ref_;

public:
    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator(TensorRef const& ref, int lane_id) : ref_(ref) {
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

        MatrixCoord lane_offset =
                lane_layout.inverse(lane_id) *
                MatrixCoord(Policy::LaneMmaShape::kM, Policy::LaneMmaShape::kN);

        ref_.add_coord_offset(lane_offset);
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_pointer_offset(LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_tile_offset(TensorCoord const& coord) {
        ref_.add_coord_offset(
                {coord.row() * Shape::kRow, coord.column() * Shape::kColumn});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator++() {
        ref_.add_coord_offset({Shape::kRow, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator--() {
        ref_.add_coord_offset({-Shape::kRow, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset)
            const {

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kN; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
                Array<Element, Policy::LaneMmaShape::kM> const* src_ptr =
                        reinterpret_cast<Array<
                                Element, Policy::LaneMmaShape::kM> const*>(
                                ref_.data() + pointer_offset +
                                ref_.offset({0, mma_n * Delta::kN + n}));

                CUTLASS_PRAGMA_UNROLL
                for (int mma_m = 0; mma_m < Iterations::kM; ++mma_m) {
                    Array<Element, Policy::LaneMmaShape::kM>* dst_ptr =
                            reinterpret_cast<
                                    Array<Element, Policy::LaneMmaShape::kM>*>(
                                    &frag) +
                            mma_m +
                            Iterations::kM *
                                    (n + mma_n * Policy::LaneMmaShape::kN);

                    *dst_ptr = src_ptr[mma_m * Policy::WarpShape::kM];
                }
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag,
                                   Index pointer_offset) const {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Policy::LaneMmaShape::kN; ++n) {
                Array<Element, Policy::LaneMmaShape::kM>* dst_ptr =
                        reinterpret_cast<
                                Array<Element, Policy::LaneMmaShape::kM>*>(
                                ref_.data() + pointer_offset +
                                ref_.offset({0, mma_n * Delta::kColumn + n}));

                CUTLASS_PRAGMA_UNROLL
                for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
                    Array<Element, Policy::LaneMmaShape::kM> const* src_ptr =
                            reinterpret_cast<Array<
                                    Element, Policy::LaneMmaShape::kM> const*>(
                                    &frag) +
                            mma_m +
                            Iterations::kRow *
                                    (n + mma_n * Policy::LaneMmaShape::kN);

                    dst_ptr[mma_m * Policy::WarpShape::kRow] = *src_ptr;
                }
            }
        }
    }
    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) const {
        store_with_pointer_offset(frag, 0);
    }
};


template <
        typename Shape_,
        typename Element_,
        typename Policy_>
class MmaSimtTileIterator<Shape_, Operand::kC, Element_, layout::RowMajor,
                          Policy_> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kC;

    using Element = Element_;

    using Layout = layout::RowMajor;

    using Policy = Policy_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;


    static_assert((!(Shape::kRow % Policy::WarpShape::kRow)) &&
                          (!(Shape::kColumn % Policy::WarpShape::kColumn)),
                  "Warp-level GEMM shape must be divisible by the arrangement "
                  "of threads in the warp.");

    static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
    static_assert(Shape::kColumn > 0,
                  "Shape::kColumn must be greater than zero.");
    static_assert(Policy::WarpShape::kRow > 0,
                  "Policy::WarpShape::kRow must be greater than zero.");
    static_assert(Policy::WarpShape::kColumn > 0,
                  "Policy::WarpShape::kColumn must be greater than zero.");
    static_assert(
            Shape::kRow / Policy::WarpShape::kRow > 0,
            "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");
    static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0,
                  "Shape::kColumn / Policy::WarpShape::kColumn must be greater "
                  "than zero.");

    using ThreadShape =
            MatrixShape<Shape::kRow / Policy::WarpShape::kRow,
                        Shape::kColumn / Policy::WarpShape::kColumn>;

    static_assert((!(ThreadShape::kRow % Policy::LaneMmaShape::kM)) &&
                          (!(ThreadShape::kColumn % Policy::LaneMmaShape::kN)),
                  "Warp-level GEMM shape must be divisible by the arrangement "
                  "of threads in the warp.");

    using Iterations =
            MatrixShape<ThreadShape::kRow / Policy::LaneMmaShape::kM,
                        ThreadShape::kColumn / Policy::LaneMmaShape::kN>;

    using Delta =
            MatrixShape<Policy::WarpShape::kRow * Policy::LaneMmaShape::kM,
                        Policy::WarpShape::kColumn * Policy::LaneMmaShape::kN>;

    using Fragment = Array<Element, ThreadShape::kCount>;

private:
    TensorRef ref_;

public:
    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator(TensorRef const& ref, int lane_id) : ref_(ref) {
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

        MatrixCoord lane_offset =
                lane_layout.inverse(lane_id) *
                MatrixCoord(Policy::LaneMmaShape::kM, Policy::LaneMmaShape::kN);

        ref_.add_coord_offset(lane_offset);
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_pointer_offset(LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_tile_offset(TensorCoord const& coord) {
        ref_.add_coord_offset(
                {coord.row() * Shape::kRow, coord.column() * Shape::kColumn});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator++() {
        ref_.add_coord_offset({Shape::kRow, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator--() {
        ref_.add_coord_offset({-Shape::kRow, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(
            Fragment& frag,
            Index pointer_offset)
            const {

        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
                Array<Element, Policy::LaneMmaShape::kN> const* src_ptr =
                        reinterpret_cast<Array<
                                Element, Policy::LaneMmaShape::kN> const*>(
                                ref_.data() + pointer_offset +
                                ref_.offset({mma_m * Delta::kRow + m, 0}));

                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
                    Array<Element, Policy::LaneMmaShape::kN>* dst_ptr =
                            reinterpret_cast<
                                    Array<Element, Policy::LaneMmaShape::kN>*>(
                                    &frag) +
                            mma_n +
                            Iterations::kColumn *
                                    (m + mma_m * Policy::LaneMmaShape::kM);

                    *dst_ptr = src_ptr[mma_n * Policy::WarpShape::kColumn];
                }
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag,
                                   Index pointer_offset) const {
        CUTLASS_PRAGMA_UNROLL
        for (int mma_m = 0; mma_m < Iterations::kRow; ++mma_m) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Policy::LaneMmaShape::kM; ++m) {
                Array<Element, Policy::LaneMmaShape::kN>* dst_ptr =
                        reinterpret_cast<
                                Array<Element, Policy::LaneMmaShape::kN>*>(
                                ref_.data() + pointer_offset +
                                ref_.offset({mma_m * Delta::kRow + m, 0}));

                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Iterations::kColumn; ++mma_n) {
                    Array<Element, Policy::LaneMmaShape::kN> const* src_ptr =
                            reinterpret_cast<Array<
                                    Element, Policy::LaneMmaShape::kN> const*>(
                                    &frag) +
                            mma_n +
                            Iterations::kColumn *
                                    (m + mma_m * Policy::LaneMmaShape::kM);

                    dst_ptr[mma_n * Policy::WarpShape::kColumn] = *src_ptr;
                }
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) const {
        store_with_pointer_offset(frag, 0);
    }
};



template <
        typename Shape_,
        typename Element_,
        typename Policy_,
        int PartitionsK,
        int PartitionGroupSize>
class MmaSimtTileIterator<Shape_, Operand::kA, Element_,
                          layout::ColumnMajorInterleaved<4>, Policy_,
                          PartitionsK, PartitionGroupSize> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kA;

    using Element = Element_;

    using Layout = layout::ColumnMajorInterleaved<4>;

    using Policy = Policy_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static const int kInterleave = 4;

    static const int kPartitionsK = PartitionsK;

    static const int kGroupPerTile = PartitionGroupSize / Shape::kColumn;


    static_assert(!(Shape::kRow % Policy::WarpShape::kRow),
                  "The warp-level GEMM M size must be divisible by the number "
                  "of threads arranged along the M dimension.");

    static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
    static_assert(Shape::kColumn > 0,
                  "Shape::kColumn must be greater than zero.");
    static_assert(Policy::WarpShape::kRow > 0,
                  "Policy::WarpShape::kRow must be greater than zero.");
    static_assert(
            Shape::kRow / Policy::WarpShape::kRow > 0,
            "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

    using ThreadShape =
            MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

    static_assert(
            !(ThreadShape::kRow % Policy::LaneMmaShape::kM) &&
                    !(ThreadShape::kColumn % Policy::LaneMmaShape::kK),
            "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

    using Iterations =
            MatrixShape<ThreadShape::kRow / Policy::LaneMmaShape::kM,
                        ThreadShape::kColumn / Policy::LaneMmaShape::kK>;

    using Fragment = Array<Element, ThreadShape::kCount>;

private:
    cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kMK>,
                       layout::ColumnMajorInterleaved<4>>
            ref_;

    int k_group_idx_;

public:
    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator(TensorRef ref, int lane_id) {
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

        MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
                                  MatrixCoord(Policy::LaneMmaShape::kM, 0);

        ref.add_coord_offset(lane_offset);

        k_group_idx_ = 0;
        ref_.reset(reinterpret_cast<Array<Element, Policy::LaneMmaShape::kMK>*>(
                           ref.data()),
                   ref.stride(0) / Policy::LaneMmaShape::kMK);
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_pointer_offset(LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_tile_offset(TensorCoord const& coord) {
        ref_.add_coord_offset(
                {coord.row() * Shape::kRow / Policy::LaneMmaShape::kMK,
                 coord.column() * Shape::kColumn});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator++() {
        add_tile_offset({0, 1});

        if (kPartitionsK > 1) {
            ++k_group_idx_;
            if (k_group_idx_ == kGroupPerTile) {
                k_group_idx_ = 0;
                add_tile_offset({0, kGroupPerTile * (kPartitionsK - 1)});
            }
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator--() {
        ref_.add_coord_offset({0, -Shape::kColumn});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kMK>* dst_ptr =
                reinterpret_cast<Array<Element, Policy::LaneMmaShape::kMK>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kColumn; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Iterations::kRow; ++m) {
                dst_ptr[m + k * Iterations::kRow] = *(
                        (ref_.data() +
                         ref_.offset({m * Policy::WarpShape::kRow / kInterleave,
                                      k * Policy::LaneMmaShape::kK}) +
                         pointer_offset / Policy::LaneMmaShape::kM));
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag,
                                   Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kMK> const* src_ptr =
                reinterpret_cast<Array<Element, Policy::LaneMmaShape::kMK>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kN; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < Iterations::kM; ++m) {
                *(ref_.data() + ref_.offset(m * Policy::WarpShape::kM, k) +
                  pointer_offset / Policy::LaneMmaShape::kM) =
                        src_ptr[m + k * Iterations::kM];
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


template <
        typename Shape_,
        typename Element_,
        typename Policy_,
        int PartitionsK,
        int PartitionGroupSize>
class MmaSimtTileIterator<Shape_, Operand::kB, Element_,
                          layout::RowMajorInterleaved<4>, Policy_, PartitionsK,
                          PartitionGroupSize> {
public:
    using Shape = Shape_;

    static Operand const kOperand = Operand::kB;

    using Element = Element_;

    using Layout = layout::RowMajorInterleaved<4>;

    using Policy = Policy_;

    using TensorRef = TensorRef<Element, Layout>;

    using Index = typename TensorRef::Index;

    using LongIndex = typename TensorRef::LongIndex;

    using TensorCoord = typename TensorRef::TensorCoord;

    static const int kInterleave = 4;

    static const int kPartitionsK = PartitionsK;

    static const int kGroupPerTile = PartitionGroupSize / Shape::kRow;


    static_assert(!(Shape::kColumn % Policy::WarpShape::kColumn),
                  "The warp-level GEMM N size must be divisible by the number "
                  "of threads arranged along the N dimension.");

    static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
    static_assert(Shape::kColumn > 0,
                  "Shape::kColumn must be greater than zero.");
    static_assert(Policy::WarpShape::kColumn > 0,
                  "Policy::WarpShape::kColumn must be greater than zero.");
    static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0,
                  "Shape::kColumn / Policy::WarpShape::kColumn must be greater "
                  "than zero.");

    using ThreadShape =
            MatrixShape<Shape::kRow,
                        Shape::kColumn / Policy::WarpShape::kColumn>;

    static_assert(
            !(ThreadShape::kColumn % Policy::LaneMmaShape::kN) &&
                    !(ThreadShape::kRow % Policy::LaneMmaShape::kK),
            "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

    using Iterations =
            MatrixShape<ThreadShape::kRow / Policy::LaneMmaShape::kK,
                        ThreadShape::kColumn / Policy::LaneMmaShape::kN>;

    using Fragment = Array<Element, ThreadShape::kCount>;

private:
    cutlass::TensorRef<Array<Element, Policy::LaneMmaShape::kKN>,
                       layout::RowMajorInterleaved<4>>
            ref_;

    int k_group_idx_;

public:
    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator(TensorRef ref, int lane_id) {
        typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

        MatrixCoord lane_offset = lane_layout.inverse(lane_id) *
                                  MatrixCoord(0, Policy::LaneMmaShape::kN);

        ref.add_coord_offset(lane_offset);

        k_group_idx_ = 0;

        ref_.reset(reinterpret_cast<Array<Element, Policy::LaneMmaShape::kKN>*>(
                           ref.data()),
                   ref.stride(0) / Policy::LaneMmaShape::kKN);
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_pointer_offset(LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& add_tile_offset(TensorCoord const& coord) {
        ref_.add_coord_offset(
                {coord.row() * Shape::kRow,
                 coord.column() * Shape::kColumn / Policy::LaneMmaShape::kKN});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator++() {
        add_tile_offset({1, 0});

        if (kPartitionsK > 1) {
            ++k_group_idx_;
            if (k_group_idx_ == kGroupPerTile) {
                k_group_idx_ = 0;
                add_tile_offset({kGroupPerTile * (kPartitionsK - 1), 0});
            }
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaSimtTileIterator& operator--() {
        ref_.add_coord_offset({-Shape::kRow, 0});

        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kKN>* dst_ptr =
                reinterpret_cast<Array<Element, Policy::LaneMmaShape::kKN>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kRow; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Iterations::kColumn; ++n) {
                dst_ptr[n + k * Iterations::kColumn] =
                        *(ref_.data() +
                          ref_.offset({k * Policy::LaneMmaShape::kK,
                                       n * Policy::WarpShape::kColumn /
                                               kInterleave}) +
                          pointer_offset / Policy::LaneMmaShape::kN);
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const& frag,
                                   Index pointer_offset) const {
        Array<Element, Policy::LaneMmaShape::kN> const* src_ptr =
                reinterpret_cast<Array<Element, Policy::LaneMmaShape::kN>*>(
                        &frag);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Iterations::kM; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Iterations::kN; ++n) {
                *(ref_.data() + ref_.offset({k, n * Policy::WarpShape::kN}) +
                  pointer_offset / Policy::LaneMmaShape::kN) =
                        src_ptr[n + k * Iterations::kN];
            }
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag, Index pointer_offset) const {
        store_with_pointer_offset(frag, 0);
    }

    CUTLASS_DEVICE
    void set_kgroup_index(int k_group) {
    }
};


}
}
}
