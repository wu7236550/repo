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

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a
  naive reference matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This
  is kernel computes the general matrix product (GEMM) using single-precision
  floating-point arithmetic and assumes all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance
  for large matrices. See the CUTLASS Parallel for All blog post for more
  exposition on the tunable parameters available in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use
  any other components or utilities within CUTLASS. Such utilities are
  demonstrated elsewhere in other examples and are prevalent in the CUTLASS unit
  tests.

  This example has delibrately been kept similar to the basic_gemm example from
  cutass-1.3 to highlight the minimum amount of differences needed to transition
  to cutlass-2.0.

  Cutlass-1.3 sgemm:
  https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

#include <iostream>
#include <sstream>
#include <vector>

#include "helper.h"


#include "cutlass/gemm/device/gemm.h"


cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const* A,
                           int lda, float const* B, int ldb, float beta,
                           float* C, int ldc) {

    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm =
            cutlass::gemm::device::Gemm<float,
                                        ColumnMajor,
                                        float,
                                        ColumnMajor,
                                        float,
                                        ColumnMajor>;

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args(
            {M, N, K},
            {A, lda},
            {B, ldb},
            {C, ldc},
            {C, ldc},
            {alpha, beta});


    cutlass::Status status = gemm_operator(args);


    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


__global__ void InitializeMatrix_kernel(float* matrix, int ldm, int rows,
                                        int columns, int seed = 0) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns) {
        int offset = i + j * ldm;

        int const k = 16807;
        int const m = 16;
        float value = float(((offset + seed) * k % m) - m / 2);

        matrix[offset] = value;
    }
}

cudaError_t InitializeMatrix(float* matrix, int ldm, int rows, int columns,
                             int seed = 0) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x,
              (columns + block.y - 1) / block.y);

    InitializeMatrix_kernel<<<grid, block>>>(matrix, ldm, rows, columns, seed);

    return cudaGetLastError();
}


cudaError_t AllocateMatrix(float** matrix, int ldm, int rows, int columns,
                           int seed = 0) {
    cudaError_t result;

    size_t sizeof_matrix = sizeof(float) * ldm * columns;

    result = cudaMalloc(reinterpret_cast<void**>(matrix), sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result)
                  << std::endl;
        return result;
    }

    result = cudaMemset(*matrix, 0, sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to clear matrix device memory: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    result = InitializeMatrix(*matrix, ldm, rows, columns, seed);

    if (result != cudaSuccess) {
        std::cerr << "Failed to initialize matrix: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    return result;
}


__global__ void ReferenceGemm_kernel(int M, int N, int K, float alpha,
                                     float const* A, int lda, float const* B,
                                     int ldb, float beta, float* C, int ldc) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        float accumulator = 0;

        for (int k = 0; k < K; ++k) {
            accumulator += A[i + k * lda] * B[k + j * ldb];
        }

        C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    }
}

cudaError_t ReferenceGemm(int M, int N, int K, float alpha, float const* A,
                          int lda, float const* B, int ldb, float beta,
                          float* C, int ldc) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta,
                                          C, ldc);

    return cudaGetLastError();
}


cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
    cudaError_t result;


    int lda = M;
    int ldb = K;
    int ldc = M;

    size_t sizeof_C = sizeof(float) * ldc * N;

    float* A;
    float* B;
    float* C_cutlass;
    float* C_reference;


    result = AllocateMatrix(&A, lda, M, K, 0);

    if (result != cudaSuccess) {
        return result;
    }

    result = AllocateMatrix(&B, ldb, K, N, 17);

    if (result != cudaSuccess) {
        cudaFree(A);
        return result;
    }

    result = AllocateMatrix(&C_cutlass, ldc, M, N, 101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        return result;
    }

    result = AllocateMatrix(&C_reference, ldc, M, N, 101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C_cutlass);
        return result;
    }

    result = cudaMemcpy(C_reference, C_cutlass, sizeof_C,
                        cudaMemcpyDeviceToDevice);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }


    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass,
                            ldc);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }


    result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference,
                           ldc);

    if (result != cudaSuccess) {
        std::cerr << "Reference GEMM kernel failed: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    std::vector<float> host_cutlass(ldc * N, 0);
    std::vector<float> host_reference(ldc * N, 0);

    result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C,
                        cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy CUTLASS GEMM results: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C,
                        cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy Reference GEMM results: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }


    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);


    if (host_cutlass != host_reference) {
        std::cerr << "CUTLASS results incorrect." << std::endl;

        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


int main(int argc, const char* arg[]) {

    int problem[3] = {128, 128, 128};

    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(arg[i]);
        ss >> problem[i - 1];
    }

    float scalars[2] = {1, 0};

    for (int i = 4; i < argc && i < 6; ++i) {
        std::stringstream ss(arg[i]);
        ss >> scalars[i - 4];
    }


    cudaError_t result = TestCutlassGemm(problem[0],
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

