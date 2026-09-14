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

#include <memory>
#include "cutlass/library/library.h"


namespace cutlass {
namespace library {


class Handle {
private:
    static int const kHostWorkspaceSize = (4 << 10);

    Provider provider_;

    cudaDeviceProp device_;

    cudaStream_t stream_;

    void* workspace_;

    size_t workspace_size_;

    ScalarPointerMode scalar_pointer_mode_;

    Operation const* last_operation_;

public:
    Handle(cudaStream_t stream = nullptr, size_t workspace_size = (4 << 20));

    ~Handle();

    Handle(Handle&& handle);

    Handle& operator=(Handle&& handle);


    int compute_capability() const;

    void set_stream(cudaStream_t stream);

    cudaStream_t get_stream() const;

    Provider get_provider() const;

    void set_provider(Provider provider);

    size_t get_workspace_size() const;

    void* get_workspace() const;

    void set_workspace_size(size_t bytes);

    ScalarPointerMode get_scalar_pointer_mode() const;

    void set_scalar_pointer_mode(ScalarPointerMode mode);

    Operation const* get_last_operation() const;


    Status gemm(

            int M,
            int N,
            int K,

            NumericTypeID
                    element_compute,

            NumericTypeID element_scalar,

            void const* alpha,

            NumericTypeID element_A,
            LayoutTypeID layout_A,
            ComplexTransform
                    transform_A,

            void const* ptr_A,
            int lda,

            NumericTypeID element_B,
            LayoutTypeID layout_B,
            ComplexTransform
                    transform_B,

            void const* ptr_B,
            int ldb,

            void const* beta,

            NumericTypeID element_C,

            void const* ptr_C,
            int ldc,

            void* ptr_D,
            int ldd
    );

    Status gemm_universal(

            GemmUniversalMode mode,

            int M,
            int N,
            int K,

            NumericTypeID
                    element_compute,

            NumericTypeID element_scalar,

            void const* alpha,

            NumericTypeID element_A,
            LayoutTypeID layout_A,
            ComplexTransform
                    transform_A,

            void const* ptr_A,
            int lda,

            NumericTypeID element_B,
            LayoutTypeID layout_B,
            ComplexTransform
                    transform_B,

            void const* ptr_B,
            int ldb,

            void const* beta,

            NumericTypeID element_C,

            void const* ptr_C,
            int ldc,

            void* ptr_D,
            int ldd,

            int batch_count = 1,

            int64_t batch_stride_A = 0,
            int64_t batch_stride_B = 0,
            int64_t batch_stride_C = 0,
            int64_t batch_stride_D = 0
    );

    Status gemm_planar_complex(

            int M,
            int N,
            int K,

            NumericTypeID
                    element_compute,

            NumericTypeID element_scalar,

            void const* alpha,

            NumericTypeID element_A,
            LayoutTypeID layout_A,
            ComplexTransform
                    transform_A,

            void const* ptr_A_real,
            void const* ptr_A_imag,
            int lda_real,
            int lda_imag,

            NumericTypeID element_B,
            LayoutTypeID layout_B,
            ComplexTransform
                    transform_B,

            void const* ptr_B_real,
            void const* ptr_B_imag,
            int ldb_real,
            int ldb_imag,

            void const* beta,

            NumericTypeID element_C,

            void const* ptr_C_real,
            void const* ptr_C_imag,
            int ldc_real,
            int ldc_imag,

            void* ptr_D_real,
            void* ptr_D_imag,
            int ldd_real,
            int ldd_imag,

            int batch_count = 1,

            int64_t batch_stride_A_real = 0, int64_t batch_stride_A_imag = 0,

            int64_t batch_stride_B_real = 0, int64_t batch_stride_B_imag = 0,

            int64_t batch_stride_C_real = 0, int64_t batch_stride_C_imag = 0,

            int64_t batch_stride_D_real = 0, int64_t batch_stride_D_imag = 0);

    Status gemm_planar_complex_array(

            int expected_M,
            int expected_N,
            int expected_K,
            int batch_count,

            int const* M,
            int const* N,
            int const* K,

            NumericTypeID
                    element_compute,

            NumericTypeID element_scalar,

            void const* alpha,

            NumericTypeID element_A,
            LayoutTypeID layout_A,
            ComplexTransform
                    transform_A,

            void const* const*
                    ptr_A_real,
            void const* const*
                    ptr_A_imag,

            int lda_real,
            int lda_imag,

            NumericTypeID element_B,
            LayoutTypeID layout_B,
            ComplexTransform
                    transform_B,

            void const* const*
                    ptr_B_real,
            void const* const*
                    ptr_B_imag,

            int ldb_real,
            int ldb_imag,

            void const* beta,

            NumericTypeID element_C,

            void const* const*
                    ptr_C_real,
            void const* const*
                    ptr_C_imag,

            int ldc_real,
            int ldc_imag,

            void* const* ptr_D_real,
            void* const* ptr_D_imag,

            int ldd_real,
            int ldd_imag
    );
};


using HandlePtr = std::unique_ptr<Handle>;

Operation const* find_conv_operation_for_parallel_reduction(
        Operation const* operation);

}
}

