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
#include "cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h"

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
        typename OpDelta_>
class MmaTensorOpGaussianComplexAccumulatorTileIterator;


template <
        typename Shape_,
        typename RealElement,
        typename InstructionShape_,
        typename OpDelta_>
class MmaTensorOpGaussianComplexAccumulatorTileIterator<
        Shape_, complex<RealElement>, cutlass::layout::RowMajor,
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

    using Fragment = Array<RealElement, (Shape::kCount / kThreads) * 3>;

    static int const kPart1Index = (Shape::kCount / kThreads) * 0;
    static int const kPart2Index = (Shape::kCount / kThreads) * 1;
    static int const kPart3Index = (Shape::kCount / kThreads) * 2;

private:
    TensorRef ref_;

public:
    CUTLASS_HOST_DEVICE
    MmaTensorOpGaussianComplexAccumulatorTileIterator() {}

    CUTLASS_HOST_DEVICE
    MmaTensorOpGaussianComplexAccumulatorTileIterator(TensorRef const& ref,
                                                      int lane_id)
            : ref_(ref) {
        int quad = (lane_id >> 2);
        int lane_in_quad = (lane_id & 3);

        MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

        ref_.add_coord_offset(lane_offset);
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpGaussianComplexAccumulatorTileIterator& add_pointer_offset(
            LongIndex offset) {
        ref_.add_pointer_offset(offset);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpGaussianComplexAccumulatorTileIterator& add_tile_offset(
            TensorCoord const& tile_offset) {
        ref_.add_coord_offset(tile_offset *
                              make_Coord(Shape::kRow, Shape::kColumn));

        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpGaussianComplexAccumulatorTileIterator& operator++() {
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MmaTensorOpGaussianComplexAccumulatorTileIterator& operator--() {
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpGaussianComplexAccumulatorTileIterator& operator+=(
            TensorCoord const& tile_offset) {
        add_tile_offset(tile_offset);
        return *this;
    }

    CUTLASS_DEVICE
    MmaTensorOpGaussianComplexAccumulatorTileIterator& operator-=(
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
                             kPart1Index] = z.real() + z.imag();
                        frag[mma_accum_start + row * kElementsPerAccess + col +
                             kPart2Index] = -z.real();
                        frag[mma_accum_start + row * kElementsPerAccess + col +
                             kPart3Index] = z.imag();
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

                        Element z(frag[kPart1Index + idx] -
                                          frag[kPart3Index + idx],
                                  frag[kPart1Index + idx] +
                                          frag[kPart2Index + idx]);

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


}
}
}

