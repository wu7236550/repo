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

#include "cutlass/epilogue/warp/tensor_op_policy.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape,
          typename OperatorShape,
          typename OperatorElementC,
          typename OperatorFragmentC,
          typename SmemLayout,
          typename GmemLayout = SmemLayout
          >
class FragmentIteratorTensorOp;


template <typename WarpShape_,
          typename OperatorShape_,
          typename OperatorElementC_,
          typename OperatorFragmentC_
          >
class FragmentIteratorTensorOp<WarpShape_, OperatorShape_, OperatorElementC_,
                               OperatorFragmentC_, layout::RowMajor> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using OperatorElementC = OperatorElementC_;
    using OperatorFragmentC = OperatorFragmentC_;
    using Layout = layout::RowMajor;

    using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Fragment =
            Array<OperatorElementC,
                  Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

    using AccumulatorTile =
            Array<OperatorElementC, OperatorFragmentC::kElements *
                                            Policy::OperatorCount::kRow *
                                            Policy::OperatorCount::kColumn>;

    using OutputAccumulatorTile = AccumulatorTile;

    static int const kIterations = Policy::kIterations;

private:
    using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;

private:

    AccessType const* accumulators_;

    int index_;

public:
    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp(AccumulatorTile const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0) {}

    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp& operator++() {
        ++index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp& operator--() {
        --index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, int index_offset = 0) const {
        int index = index_ + index_offset;

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            int accumulator_access_offset =
                    index + n * Policy::kAccumulatorColumnStride /
                                    Policy::kElementsPerAccess;

            frag_ptr[n] = accumulators_[accumulator_access_offset];
        }
    }
};


template <
        typename WarpShape_,
        typename OperatorShape_,
        typename OperatorElementC_,
        typename OperatorFragmentC_,
        int InterleavedK>
class FragmentIteratorTensorOp<WarpShape_, OperatorShape_, OperatorElementC_,
                               OperatorFragmentC_,
                               layout::ColumnMajorInterleaved<InterleavedK>> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using OperatorElementC = OperatorElementC_;
    using OperatorFragmentC = OperatorFragmentC_;
    static int const kInterleavedK = InterleavedK;
    using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;

    using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Fragment =
            Array<OperatorElementC, Policy::kElementsPerAccess * InterleavedK /
                                            OperatorShape::kN>;

    using AccumulatorTile =
            Array<OperatorElementC, OperatorFragmentC::kElements *
                                            Policy::OperatorCount::kRow *
                                            Policy::OperatorCount::kColumn>;

    static int const kIterations = Policy::kIterations;

private:
    using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;

private:

    AccessType const* accumulators_;

    int index_;

public:
    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp(AccumulatorTile const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0) {}

    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp& operator++() {
        ++index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp& operator--() {
        --index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, int index_offset = 0) const {
        int index = index_ + index_offset;

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < (InterleavedK / OperatorShape::kN); ++n) {
            int index_m = index % (Policy::OperatorCount::kRow *
                                   Policy::kIterationsPerInstruction);
            int index_n = index / (Policy::OperatorCount::kRow *
                                   Policy::kIterationsPerInstruction);
            int accumulator_access_offset =
                    (index_m / Policy::kIterationsPerInstruction) *
                            (Policy::OperatorCount::kColumn *
                             Policy::kIterationsPerInstruction) +
                    (index_m % Policy::kIterationsPerInstruction) +
                    index_n * (InterleavedK / OperatorShape::kN) *
                            Policy::kIterationsPerInstruction +
                    n * Policy::kIterationsPerInstruction;

            frag_ptr[n] = accumulators_[accumulator_access_offset];
        }
    }
};

template <
        typename WarpShape_,
        typename OperatorShape_,
        typename OperatorElementC_,
        typename OperatorFragmentC_,
        int InterleavedK>
class FragmentIteratorTensorOp<WarpShape_, OperatorShape_, OperatorElementC_,
                               OperatorFragmentC_, layout::RowMajor,
                               layout::TensorNCxHWx<InterleavedK>> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using OperatorElementC = OperatorElementC_;
    using OperatorFragmentC = OperatorFragmentC_;
    static int const kInterleavedK = InterleavedK;
    using SmemLayout = layout::RowMajor;
    using GmemLayout = layout::TensorNCxHWx<kInterleavedK>;

    using Policy =
            TensorOpPolicy<WarpShape, OperatorShape, SmemLayout, GmemLayout>;

    using Fragment =
            Array<OperatorElementC, Policy::kElementsPerAccess * InterleavedK /
                                            OperatorShape::kM>;

    using AccumulatorTile =
            Array<OperatorElementC, OperatorFragmentC::kElements *
                                            Policy::OperatorCount::kRow *
                                            Policy::OperatorCount::kColumn>;

    static int const kIterations = Policy::kIterations;

private:
    using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;

private:

    AccessType const* accumulators_;

    int index_;

public:
    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp(AccumulatorTile const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0) {}

    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp& operator++() {
        ++index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp& operator--() {
        --index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, int index_offset = 0) const {
        int index = index_ + index_offset;

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < (InterleavedK / OperatorShape::kM); ++n) {
            int index_m = index / Policy::OperatorCount::kColumn;
            int index_n = index % Policy::OperatorCount::kColumn;

            int accumulator_access_offset =
                    (index_m * (InterleavedK / OperatorShape::kM) + n) *
                            Policy::OperatorCount::kColumn +
                    index_n;
            frag_ptr[n] = accumulators_[accumulator_access_offset];
        }
    }
};


template <
        typename WarpShape_,
        typename OperatorShape_,
        typename OperatorElementC_,
        typename OperatorFragmentC_>
class FragmentIteratorTensorOp<WarpShape_, OperatorShape_, OperatorElementC_,
                               OperatorFragmentC_, layout::RowMajor,
                               layout::TensorNCxHWx<4>> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using OperatorElementC = OperatorElementC_;
    using OperatorFragmentC = OperatorFragmentC_;
    static int const kInterleavedK = 4;
    using SmemLayout = layout::RowMajor;
    using GmemLayout = layout::TensorNCxHWx<kInterleavedK>;

    using Policy =
            TensorOpPolicy<WarpShape, OperatorShape, SmemLayout, GmemLayout>;

    using Fragment =
            Array<OperatorElementC, Policy::kElementsPerAccess *
                                            Policy::kColumnsPerIteration /
                                            OperatorShape::kN>;

    using AccumulatorTile =
            Array<OperatorElementC, OperatorFragmentC::kElements *
                                            Policy::OperatorCount::kRow *
                                            Policy::OperatorCount::kColumn>;

    static int const kIterations = Policy::kIterations;

private:
    using AccessType = Array<OperatorElementC, Policy::kElementsPerAccess>;

private:

    AccessType const* accumulators_;

    int index_;

public:
    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp(AccumulatorTile const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0) {}

    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp& operator++() {
        ++index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    FragmentIteratorTensorOp& operator--() {
        --index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, int index_offset = 0) const {
        int index = index_ + index_offset;

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < (Policy::kColumnsPerIteration / OperatorShape::kN);
             ++n) {
            int index_m = index / Policy::Iterations::kColumn;
            int index_n = index % Policy::Iterations::kColumn;

            int accumulator_access_offset =
                    index_m * Policy::OperatorCount::kColumn +
                    (index_n * (Policy::kColumnsPerIteration /
                                OperatorShape::kN) +
                     n);
            frag_ptr[n] = accumulators_[accumulator_access_offset];
        }
    }
};



}
}
}

