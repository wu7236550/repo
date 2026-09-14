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
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"


namespace cutlass {
namespace epilogue {


template <typename Shape_,
          int WarpCount,
          typename FragmentC_
          >
class EpilogueWorkspace {
public:
    using Shape = Shape_;
    using FragmentC = FragmentC_;
    using ElementC = typename FragmentC::value_type;

    static int const kWarpCount = WarpCount;

    static int const kAccessSizeInBits = 128;

    static int const kWarpSize = 32;

    static int const kElementsPerAccess =
            kAccessSizeInBits / sizeof_bits<ElementC>::value;

    static int const kIterations = FragmentC::kElements / kElementsPerAccess;

    static_assert(
            !(FragmentC::kElements % kElementsPerAccess),
            "The number of accumulators must be divisible by the access size.");

    static int const kWarpAccesses = kIterations * kWarpSize;

    static int const kThreadblockAccesses = kWarpAccesses * kWarpCount;

    struct Params {
        ElementC* ptr_C;

        int stride_n;

        int stride_k;


        CUTLASS_HOST_DEVICE
        Params(ElementC* ptr_C,
               int stride_n_,
               int stride_k_
               )
                : ptr_C(ptr_C),
                  stride_n(stride_n_ / kElementsPerAccess),
                  stride_k(stride_k_ / kElementsPerAccess) {}
    };

    struct SharedStorage {
    };

private:
    struct alignas((kAccessSizeInBits / 8)) AccessType {
        Array<ElementC, kElementsPerAccess> storage;
    };

    AccessType* pointer_;

    int stride_n_;

    int stride_k_;

public:
    CUTLASS_DEVICE
    EpilogueWorkspace(
            Params const& params,
            SharedStorage&,
            int warp_idx,
            int lane_idx

            )
            : pointer_(reinterpret_cast<AccessType*>(params.ptr_C)),
              stride_n_(params.stride_n),
              stride_k_(params.stride_k) {
        pointer_ += lane_idx + warp_idx * kWarpAccesses;
    }

    CUTLASS_DEVICE
    void operator()(
            cutlass::gemm::GemmCoord
                    problem_size,
            cutlass::gemm::GemmCoord
                    tb_tile_coord,
            FragmentC const& accum) {

        AccessType* pointer =
                pointer_ + tb_tile_coord.m() * kThreadblockAccesses +
                tb_tile_coord.n() * stride_n_ + tb_tile_coord.k() * stride_k_;

        AccessType const* src_pointer =
                reinterpret_cast<AccessType const*>(&accum);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kIterations; ++i) {
            pointer[i * kWarpSize] = src_pointer[i];
        }
    }
};


}
}

