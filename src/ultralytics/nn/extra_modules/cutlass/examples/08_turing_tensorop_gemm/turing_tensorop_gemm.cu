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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

using ElementAccumulator = int32_t;
using ElementComputeEpilogue =
        ElementAccumulator;
using ElementInputA = int8_t;
using ElementInputB = int8_t;
using ElementOutput = int32_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm75;

using ShapeMMAThreadBlock =
        cutlass::gemm::GemmShape<128, 256, 64>;
using ShapeMMAWarp =
        cutlass::gemm::GemmShape<64, 64,
                                 64>;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;

using SwizzleThreadBlock =
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::
                        value,
        ElementAccumulator,
        ElementComputeEpilogue>;

constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
        LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
        ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

int run() {
    const int length_m = 5120;
    const int length_n = 4096;
    const int length_k = 4096;

    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
            problem_size.mk());
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
            problem_size.kn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
            problem_size.mn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
            problem_size.mn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
            problem_size.mn());

    cutlass::reference::host::TensorFillRandomUniform(
            tensor_a.host_view(), 1, ElementInputA(4), ElementInputA(-4),
            0);
    cutlass::reference::host::TensorFillRandomUniform(
            tensor_b.host_view(), 1, ElementInputB(4), ElementInputB(-4),
            0);
    cutlass::reference::host::TensorFillRandomUniform(
            tensor_c.host_view(), 1, ElementOutput(4), ElementOutput(-4),
            0);
    cutlass::reference::host::TensorFill(
            tensor_d.host_view());
    cutlass::reference::host::TensorFill(
            tensor_ref_d.host_view());

    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    int split_k_slices = 1;

    typename Gemm::Arguments arguments{
            problem_size,
            tensor_a.device_ref(),
            tensor_b.device_ref(),
            tensor_c.device_ref(),
            tensor_d.device_ref(),
            {alpha, beta},
            split_k_slices};

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = gemm_op();
    CUTLASS_CHECK(status);

    cutlass::reference::device::Gemm<ElementInputA, LayoutInputA, ElementInputB,
                                     LayoutInputB, ElementOutput, LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue>
            gemm_device;

    gemm_device(problem_size, alpha, tensor_a.device_ref(),
                tensor_b.device_ref(), beta, tensor_c.device_ref(),
                tensor_ref_d.device_ref());

    cudaDeviceSynchronize();

    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(
            tensor_d.host_view(), tensor_ref_d.host_view());

    std::cout << (passed ? "Passed" : "Failed") << std::endl;

    return (passed ? 0 : -1);
}

int main() {
    bool notSupported = false;

    if (!(__CUDACC_VER_MAJOR__ > 10 ||
          (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
        std::cerr << "Turing Tensor Core operations must be compiled with CUDA "
                     "10.2 Toolkit or later."
                  << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: "
                  << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (!((props.major * 10 + props.minor) >= 75)) {
        std::cerr << "Turing Tensor Core operations must be run on a machine "
                     "with compute capability at least 75."
                  << std::endl;

        notSupported = true;
    }

    if (notSupported) {
        return 0;
    }

    return run();
}
