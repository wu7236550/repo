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

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif
#include "cutlass/layout/matrix.h"

namespace cutlass {
namespace arch {

template <typename Shape_, typename LayoutA_, typename LayoutB_,
          typename LayoutC_>
struct Wmma<Shape_,
            cutlass::int4b_t,
            LayoutA_,
            cutlass::int4b_t,
            LayoutB_,
            int32_t,
            LayoutC_,
            cutlass::arch::OpMultiplyAdd
            > {
#if defined(CUTLASS_ARCH_WMMA_SM75_ENABLED)
    using Shape = Shape_;
    using ElementA = cutlass::int4b_t;
    using LayoutA = LayoutA_;
    using ElementB = cutlass::int4b_t;
    using LayoutB = LayoutB_;
    using ElementC = int32_t;
    using LayoutC = LayoutC_;
    using Operator = cutlass::arch::OpMultiplyAdd;
    using ArchTag = arch::Sm75;

    static_assert(
            platform::is_same<cutlass::gemm::GemmShape<8, 8, 32>, Shape>::value,
            "Supported list of wmma operator shape for s8 multiplicands is: "
            "8x8x32");

    using FragmentA = nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_a, Shape::kM, Shape::kN, Shape::kK,
            typename CutlassToWmmaDataType<ElementA>::Type,
            typename CutlassToWmmaLayout<LayoutA>::Layout>;

    using FragmentB = nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_b, Shape::kM, Shape::kN, Shape::kK,
            typename CutlassToWmmaDataType<ElementB>::Type,
            typename CutlassToWmmaLayout<LayoutB>::Layout>;

    using FragmentC = nvcuda::wmma::fragment<
            nvcuda::wmma::accumulator, Shape::kM, Shape::kN, Shape::kK,
            typename CutlassToWmmaDataType<ElementC>::Type>;

    CUTLASS_DEVICE
    void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B,
                    FragmentC const& C) const {
        nvcuda::wmma::mma_sync(D, A, B, C);
    }

#else
    static_assert(false,
                  "wmma.mma.sync interger type multiplicands is avialable only "
                  "for SM75 and beyond");
#endif
};

template <typename Shape_, typename LayoutA_, typename LayoutB_,
          typename LayoutC_>
struct Wmma<Shape_,
            cutlass::uint1b_t,
            LayoutA_,
            cutlass::uint1b_t,
            LayoutB_,
            int32_t,
            LayoutC_,
            cutlass::arch::OpXorPopc
            > {
#if defined(CUTLASS_ARCH_WMMA_SM75_ENABLED)
    using Shape = Shape_;
    using ElementA = cutlass::uint1b_t;
    using LayoutA = LayoutA_;
    using ElementB = cutlass::uint1b_t;
    using LayoutB = LayoutB_;
    using ElementC = int32_t;
    using LayoutC = LayoutC_;
    using Operator = cutlass::arch::OpXorPopc;
    using ArchTag = arch::Sm75;

    static_assert(platform::is_same<cutlass::gemm::GemmShape<8, 8, 128>,
                                    Shape>::value,
                  "Supported list of wmma operator shape for b1 multiplicands "
                  "is: 8x8x128");

    using FragmentA = nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_a, Shape::kM, Shape::kN, Shape::kK,
            typename CutlassToWmmaDataType<ElementA>::Type,
            typename CutlassToWmmaLayout<LayoutA>::Layout>;

    using FragmentB = nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_b, Shape::kM, Shape::kN, Shape::kK,
            typename CutlassToWmmaDataType<ElementB>::Type,
            typename CutlassToWmmaLayout<LayoutB>::Layout>;

    using FragmentC = nvcuda::wmma::fragment<
            nvcuda::wmma::accumulator, Shape::kM, Shape::kN, Shape::kK,
            typename CutlassToWmmaDataType<ElementC>::Type>;

    CUTLASS_DEVICE
    void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B,
                    FragmentC const& C) const {
        nvcuda::wmma::bmma_sync(
                D, A, B, C, nvcuda::wmma::experimental::bmmaBitOpXOR,
                nvcuda::wmma::experimental::bmmaAccumulateOpPOPC);
    }

#else
    static_assert(false,
                  "wmma.mma.sync interger type multiplicands is avialable only "
                  "for SM75 and beyond");
#endif
};

}
}
