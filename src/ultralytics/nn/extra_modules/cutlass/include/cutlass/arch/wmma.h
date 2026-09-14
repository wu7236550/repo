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

#if !defined(__clang__)

#if (__CUDACC_VER_MAJOR__ >= 9)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))
#define CUTLASS_ARCH_WMMA_ENABLED
#define CUTLASS_ARCH_WMMA_SM70_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ >= 10)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 720))
#define CUTLASS_ARCH_INTEGER_MATRIX_MULTIPLY_ENABLED
#define CUTLASS_ARCH_WMMA_SM72_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ >= 10)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define CUTLASS_SUBBYTE_INTEGER_MATRIX_MULTIPLY_ENABLED
#define CUTLASS_ARCH_WMMA_SM75_ENABLED
#endif
#endif

#endif

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include <mma.h>
#include "cutlass/arch/mma.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"


namespace cutlass {
namespace arch {

template <typename Type_>
struct CutlassToWmmaDataType {
    using Type = Type_;
};

template <>
struct CutlassToWmmaDataType<cutlass::half_t> {
    using Type = __half;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && \
        (__CUDACC_VER_MAJOR__ >= 11)
template <>
struct CutlassToWmmaDataType<cutlass::bfloat16_t> {
    using Type = __nv_bfloat16;
};
#endif

template <>
struct CutlassToWmmaDataType<int8_t> {
    using Type = signed char;
};

template <>
struct CutlassToWmmaDataType<uint8_t> {
    using Type = unsigned char;
};

template <>
struct CutlassToWmmaDataType<int32_t> {
    using Type = int;
};

#if defined(CUTLASS_SUBBYTE_INTEGER_MATRIX_MULTIPLY_ENABLED)
template <>
struct CutlassToWmmaDataType<cutlass::int4b_t> {
    using Type = nvcuda::wmma::experimental::precision::s4;
};

template <>
struct CutlassToWmmaDataType<cutlass::uint4b_t> {
    using Type = nvcuda::wmma::experimental::precision::u4;
};

template <>
struct CutlassToWmmaDataType<cutlass::uint1b_t> {
    using Type = nvcuda::wmma::experimental::precision::b1;
};
#endif

template <typename Layout_>
struct CutlassToWmmaLayout {};

template <>
struct CutlassToWmmaLayout<cutlass::layout::RowMajor> {
    using Layout = nvcuda::wmma::row_major;
    static nvcuda::wmma::layout_t const value =
            nvcuda::wmma::layout_t::mem_row_major;
};

template <>
struct CutlassToWmmaLayout<cutlass::layout::ColumnMajor> {
    using Layout = nvcuda::wmma::col_major;
    static nvcuda::wmma::layout_t const value =
            nvcuda::wmma::layout_t::mem_col_major;
};

template <typename Type_>
struct WmmaToCutlassDataType {
    using Type = Type_;
};

template <>
struct WmmaToCutlassDataType<__half> {
    using Type = cutlass::half_t;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && \
        (__CUDACC_VER_MAJOR__ >= 11)
template <>
struct WmmaToCutlassDataType<__nv_bfloat16> {
    using Type = cutlass::bfloat16_t;
};
#endif


template <typename Shape_,
          typename ElementA_,
          typename LayoutA_,
          typename ElementB_,
          typename LayoutB_,
          typename ElementC_,
          typename LayoutC_,
          typename Operator_ =
                  cutlass::arch::OpMultiplyAdd
          >
struct Wmma;

}
}


#ifdef CUTLASS_ARCH_WMMA_SM70_ENABLED
#include "cutlass/arch/wmma_sm70.h"
#endif

#ifdef CUTLASS_ARCH_WMMA_SM72_ENABLED
#include "cutlass/arch/wmma_sm72.h"
#endif

#ifdef CUTLASS_ARCH_WMMA_SM75_ENABLED
#include "cutlass/arch/wmma_sm75.h"
#endif


#endif
