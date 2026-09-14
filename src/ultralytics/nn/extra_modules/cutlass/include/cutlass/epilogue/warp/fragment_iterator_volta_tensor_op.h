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

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/warp/volta_tensor_op_policy.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape,
          typename InterleavedTileShape,
          typename ElementC,
          typename Layout
          >
class FragmentIteratorVoltaTensorOp;


template <typename WarpShape_
          >
class FragmentIteratorVoltaTensorOp<WarpShape_, gemm::GemmShape<32, 32, 4>,
                                    half_t, layout::RowMajor> {
public:
    using WarpShape = WarpShape_;
    using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
    using ElementC = half_t;
    using Layout = layout::RowMajor;

    using Policy = VoltaTensorOpPolicy<WarpShape, InterleavedTileShape,
                                       ElementC, Layout>;

    using AccessType = typename Policy::AccessType;

    using Fragment = typename Policy::Fragment;

    using AccumulatorTile = typename Policy::AccumulatorTile;

    using OutputAccumulatorTile = AccumulatorTile;

    static int const kIterations = Policy::kIterations;

private:
private:

    AccessType const* accumulators_;

    int index_;

public:
    CUTLASS_HOST_DEVICE
    FragmentIteratorVoltaTensorOp(AccumulatorTile const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0) {}

    CUTLASS_HOST_DEVICE
    FragmentIteratorVoltaTensorOp& operator++() {
        ++index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    FragmentIteratorVoltaTensorOp& operator--() {
        --index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, int index_offset = 0) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        static int const kAccessesPerMma =
                Policy::kElementsPerMma / Policy::kElementsPerAccess;

        CUTLASS_PRAGMA_UNROLL
        for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn;
             ++tile_n) {
            int tile_access_idx =
                    (tile_n * Policy::TileIterations::kRow + (index_ & 2) / 2) *
                    Policy::MmaIterations::kCount * kAccessesPerMma;

            CUTLASS_PRAGMA_UNROLL
            for (int mma_n = 0;
                 mma_n < Policy::MmaIterations::kColumn * kAccessesPerMma;
                 ++mma_n) {
                int mma_access_idx =
                        ((mma_n & 1) * 2 + (index_ & 1)) * kAccessesPerMma +
                        (mma_n & 2) / 2;

                frag_ptr[tile_n * Policy::MmaIterations::kColumn *
                                 kAccessesPerMma +
                         mma_n] =
                        accumulators_[tile_access_idx + mma_access_idx];
            }
        }
    }
};


template <typename WarpShape_
          >
class FragmentIteratorVoltaTensorOp<WarpShape_, gemm::GemmShape<32, 32, 4>,
                                    float, layout::RowMajor> {
public:
    using WarpShape = WarpShape_;
    using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
    using ElementC = float;
    using Layout = layout::RowMajor;

    using Policy = VoltaTensorOpPolicy<WarpShape, InterleavedTileShape,
                                       ElementC, Layout>;

    using AccessType = typename Policy::AccessType;

    using Fragment = typename Policy::Fragment;

    using AccumulatorTile = typename Policy::AccumulatorTile;

    static int const kIterations = Policy::kIterations;

private:
private:

    AccessType const* accumulators_;

    int index_;

public:
    CUTLASS_HOST_DEVICE
    FragmentIteratorVoltaTensorOp(AccumulatorTile const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0) {}

    CUTLASS_HOST_DEVICE
    FragmentIteratorVoltaTensorOp& operator++() {
        ++index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    FragmentIteratorVoltaTensorOp& operator--() {
        --index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, int index_offset = 0) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        int const kRegsPerMmaRow = 2;

        CUTLASS_PRAGMA_UNROLL
        for (int reg_row = 0; reg_row < Policy::kRowsPerMmaTile; ++reg_row) {
            CUTLASS_PRAGMA_UNROLL
            for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn;
                 ++tile_n) {
                CUTLASS_PRAGMA_UNROLL
                for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn * 2;
                     ++mma_n) {
                    int mma_idx =
                            (index_ & 1) +
                            (index_ & 2) * Policy::MmaIterations::kCount / 2 +
                            (tile_n * Policy::TileIterations::kRow) *
                                    Policy::MmaIterations::kCount +
                            (mma_n & 1) * 2;

                    int reg_offset = reg_row * kRegsPerMmaRow + (mma_n & 2) * 2;
                    int reg_idx =
                            mma_idx * Policy::kElementsPerMma + reg_offset;

                    *frag_ptr =
                            accumulators_[reg_idx / Policy::kElementsPerAccess];
                    ++frag_ptr;
                }
            }
        }
    }
};


}
}
}

