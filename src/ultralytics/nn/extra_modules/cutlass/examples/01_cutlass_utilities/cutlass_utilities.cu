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


#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"


#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/host_tensor.h"

#include "cutlass/numeric_types.h"

#include "cutlass/util/device_memory.h"

#include "cutlass/util/reference/device/tensor_fill.h"

#include "cutlass/util/reference/host/tensor_compare.h"

#include "cutlass/util/reference/host/gemm.h"

#pragma warning(disable : 4503)

cudaError_t cutlass_hgemm_nn(int M, int N, int K, cutlass::half_t alpha,
                             cutlass::half_t const* A, int lda,
                             cutlass::half_t const* B, int ldb,
                             cutlass::half_t beta, cutlass::half_t* C,
                             int ldc) {
    using Gemm = cutlass::gemm::device::Gemm<
            cutlass::half_t,
            cutlass::layout::ColumnMajor,
            cutlass::half_t,
            cutlass::layout::ColumnMajor,
            cutlass::half_t,
            cutlass::layout::ColumnMajor
            >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op(
            {{M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}});

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


cudaError_t TestCutlassGemm(int M, int N, int K, cutlass::half_t alpha,
                            cutlass::half_t beta) {
    cudaError_t result;


    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(
            cutlass::MatrixCoord(M, K));

    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(
            cutlass::MatrixCoord(K, N));

    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>
            C_cutlass(cutlass::MatrixCoord(M, N));

    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>
            C_reference(cutlass::MatrixCoord(M, N));


    uint64_t seed = 2080;

    cutlass::half_t mean = 0.0_hf;
    cutlass::half_t stddev = 5.0_hf;

    int bits_less_than_one = 0;

    cutlass::reference::device::TensorFillRandomGaussian(
            A.device_view(), seed, mean, stddev, bits_less_than_one);

    cutlass::reference::device::TensorFillRandomGaussian(
            B.device_view(), seed * 2019, mean, stddev, bits_less_than_one);

    cutlass::reference::device::TensorFillRandomGaussian(
            C_cutlass.device_view(), seed * 1993, mean, stddev,
            bits_less_than_one);

    cutlass::device_memory::copy_device_to_device(C_reference.device_data(),
                                                  C_cutlass.device_data(),
                                                  C_cutlass.capacity());

    C_reference.sync_host();


    result = cutlass_hgemm_nn(M, N, K, alpha, A.device_data(), A.stride(0),
                              B.device_data(), B.stride(0), beta,
                              C_cutlass.device_data(), C_cutlass.stride(0));

    if (result != cudaSuccess) {
        return result;
    }


    A.sync_host();
    B.sync_host();

    C_cutlass.sync_host();

    cutlass::reference::host::Gemm<
            cutlass::half_t,
            cutlass::layout::ColumnMajor,
            cutlass::half_t,
            cutlass::layout::ColumnMajor,
            cutlass::half_t,
            cutlass::layout::ColumnMajor,
            cutlass::half_t, cutlass::half_t>
            gemm_ref;

    gemm_ref({M, N, K},
             alpha,
             A.host_ref(),
             B.host_ref(),
             beta,
             C_reference.host_ref()
    );

    if (!cutlass::reference::host::TensorEquals(C_reference.host_view(),
                                                C_cutlass.host_view())) {
        char const* filename = "errors_01_cutlass_utilities.csv";

        std::cerr << "Error - CUTLASS GEMM kernel differs from reference. "
                     "Wrote computed and reference results to '"
                  << filename << "'" << std::endl;


        std::ofstream file(filename);

        file << "\n\nCUTLASS =\n" << C_cutlass.host_view() << std::endl;

        file << "\n\nReference =\n" << C_reference.host_view() << std::endl;

        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


int main(int argc, const char* arg[]) {

    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, 0);

    if (result != cudaSuccess) {
        std::cerr << "Failed to query device properties with error "
                  << cudaGetErrorString(result) << std::endl;
        return -1;
    }

    if (!(prop.major > 5 || (prop.major == 5 && prop.minor >= 3))) {
        std::cerr << "This example uses half precision and is only suitable "
                     "for devices with compute capability 5.3 or greater.\n";
        std::cerr << "You are using a CUDA device with compute capability "
                  << prop.major << "." << prop.minor << std::endl;
        return -1;
    }


    int problem[3] = {128, 128, 128};

    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(arg[i]);
        ss >> problem[i - 1];
    }

    cutlass::half_t scalars[2] = {1.0_hf, 0.0_hf};

    for (int i = 4; i < argc && i < 6; ++i) {
        std::stringstream ss(arg[i]);

        ss >> scalars[i - 4];
    }


    result = TestCutlassGemm(problem[0],
                             problem[1],
                             problem[2],
                             scalars[0],
                             scalars[1]
    );

    if (result == cudaSuccess) {
        std::cout << "Passed." << std::endl;
    }

    return result == cudaSuccess ? 0 : -1;
}

