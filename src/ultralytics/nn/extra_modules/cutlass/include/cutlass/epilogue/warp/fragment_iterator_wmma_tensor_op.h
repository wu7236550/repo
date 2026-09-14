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

#if !(defined(__clang__) && defined(__CUDA__))

#include "cutlass/wmma_array.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/epilogue/warp/wmma_tensor_op_policy.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape,
          typename OperatorShape,
          typename OperatorElementC,
          typename OperatorFragmentC,
          typename Layout
          >
class FragmentIteratorWmmaTensorOp;


template <typename WarpShape_,
          typename OperatorShape_,
          typename OperatorElementC_,
          typename OperatorFragmentC_
          >
class FragmentIteratorWmmaTensorOp<WarpShape_, OperatorShape_,
                                   OperatorElementC_, OperatorFragmentC_,
                                   layout::RowMajor> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using OperatorElementC = OperatorElementC_;
    using OperatorFragmentC = OperatorFragmentC_;
    using Layout = layout::RowMajor;

    using Policy = WmmaTensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Fragment = WmmaFragmentArray<OperatorFragmentC,
                                       Policy::OperatorCount::kColumn>;

    using AccumulatorTile =
            WmmaFragmentArray<OperatorFragmentC, Policy::OperatorCount::kCount>;

    using OutputAccumulatorTile = AccumulatorTile;

private:
    using AccessType = WmmaFragmentArray<OperatorFragmentC,
                                         Policy::kWmmaFragmentsPerAccess>;

private:

    AccessType const* accumulators_;

    int index_;

public:
    CUTLASS_HOST_DEVICE
    FragmentIteratorWmmaTensorOp(AccumulatorTile const& accum)
            : accumulators_(reinterpret_cast<AccessType const*>(&accum)),
              index_(0) {}

    CUTLASS_HOST_DEVICE
    FragmentIteratorWmmaTensorOp& operator++() {
        ++index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    FragmentIteratorWmmaTensorOp& operator--() {
        --index_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag, int index_offset = 0) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; n++) {
            int accumulator_access_offset =
                    index_ * Policy::OperatorCount::kColumn + n;

            frag_ptr[n] = accumulators_[accumulator_access_offset];
        }
    }
};

}
}
}


#else
#error(defined(__clang__) && defined(__CUDA__))
#endif
