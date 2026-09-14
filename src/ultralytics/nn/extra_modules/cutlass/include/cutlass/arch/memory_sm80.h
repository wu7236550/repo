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
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/cache_operation.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define CUDA_CP_ASYNC_ACTIVATED 1
#else
#define CUDA_CP_ASYNC_ACTIVATED 0
#endif

namespace cutlass {
namespace arch {


template <
        int SizeInBytes,
        CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async;

template <
        int SizeInBytes,
        CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_zfill;


template <
        int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Always> {
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                  "Size is not supported");

    CUTLASS_DEVICE
    cp_async(void* smem_ptr, void const* global_ptr, bool pred_guard = true) {
#if CUDA_CP_ASYNC_ACTIVATED

        unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %0, 0;\n"
                "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
                "}\n" ::"r"((int)pred_guard),
                "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

#else
        using AccessType = Array<uint8_t, SizeInBytes>;

        if (pred_guard) {
            *static_cast<AccessType*>(smem_ptr) =
                    *static_cast<AccessType const*>(global_ptr);
        }
#endif
    }
};

template <
        int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Always> {
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                  "Size is not supported");

    CUTLASS_DEVICE
    cp_async_zfill(void* smem_ptr, void const* global_ptr, bool pred_guard) {
#if CUDA_CP_ASYNC_ACTIVATED

        unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
        int src_in_bytes = (pred_guard ? SizeInBytes : 0);

        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(
                             smem_int_ptr),
                     "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));

#else
        using AccessType = Array<uint8_t, SizeInBytes>;

        if (pred_guard) {
            *static_cast<AccessType*>(smem_ptr) =
                    *static_cast<AccessType const*>(global_ptr);
        } else {
            AccessType zeros;
            zeros.clear();
            *static_cast<AccessType*>(smem_ptr) = zeros;
        }
#endif
    }
};


template <
        int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Global> {
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                  "Size is not supported");

    CUTLASS_DEVICE
    cp_async(void* smem_ptr, void const* global_ptr, bool pred_guard = true) {
#if CUDA_CP_ASYNC_ACTIVATED

        static_assert(SizeInBytes == 16,
                      "cp.async only supports CacheOperation::Global when "
                      "access size is 16B.");

        unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %0, 0;\n"
                "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                "}\n" ::"r"((int)pred_guard),
                "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

#else
        using AccessType = Array<uint8_t, SizeInBytes>;

        if (pred_guard) {
            *static_cast<AccessType*>(smem_ptr) =
                    *static_cast<AccessType const*>(global_ptr);
        }
#endif
    }
};

template <
        int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Global> {
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                  "Size is not supported");

    CUTLASS_DEVICE
    cp_async_zfill(void* smem_ptr, void const* global_ptr,
                   bool pred_guard = true) {
#if CUDA_CP_ASYNC_ACTIVATED

        static_assert(SizeInBytes == 16,
                      "cp.async only supports CacheOperation::Global when "
                      "access size is 16B.");

        unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
        int src_in_bytes = (pred_guard ? SizeInBytes : 0);

        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(
                             smem_int_ptr),
                     "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));

#else
        using AccessType = Array<uint8_t, SizeInBytes>;

        if (pred_guard) {
            *static_cast<AccessType*>(smem_ptr) =
                    *static_cast<AccessType const*>(global_ptr);
        } else {
            AccessType zeros;
            zeros.clear();
            *static_cast<AccessType*>(smem_ptr) = zeros;
        }
#endif
    }
};


CUTLASS_DEVICE
void cp_async_fence() {
#if CUDA_CP_ASYNC_ACTIVATED
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}


template <int N>
CUTLASS_DEVICE void cp_async_wait() {
#if CUDA_CP_ASYNC_ACTIVATED
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <>
CUTLASS_DEVICE void cp_async_wait<0>() {
#if CUDA_CP_ASYNC_ACTIVATED
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}


}
}

