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
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor_planar_complex.h"

#include "cutlass/util/reference/device/tensor_fill.h"

#include "cutlass/util/reference/device/gemm_planar_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "cutlass/library/handle.h"


struct Result {
    double runtime_ms;
    double gflops;
    cutlass::Status status;
    cudaError_t error;
    bool passed;


    Result(double runtime_ms = 0, double gflops = 0,
           cutlass::Status status = cutlass::Status::kSuccess,
           cudaError_t error = cudaSuccess)
            : runtime_ms(runtime_ms),
              gflops(gflops),
              status(status),
              error(error),
              passed(true) {}
};


struct Options {
    bool help;

    cutlass::gemm::GemmCoord problem_size;
    int batch_count;
    cutlass::complex<float> alpha;
    cutlass::complex<float> beta;

    bool reference_check;
    int iterations;

    Options()
            : help(false),
              problem_size({1024, 1024, 1024}),
              batch_count(1),
              reference_check(true),
              iterations(20),
              alpha(1),
              beta() {}

    bool valid() { return true; }

    void parse(int argc, char const** args) {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")) {
            help = true;
        }

        cmd.get_cmd_line_argument("m", problem_size.m());
        cmd.get_cmd_line_argument("n", problem_size.n());
        cmd.get_cmd_line_argument("k", problem_size.k());
        cmd.get_cmd_line_argument("batch", batch_count);

        cmd.get_cmd_line_argument("alpha", alpha.real());
        cmd.get_cmd_line_argument("alpha_i", alpha.imag());
        cmd.get_cmd_line_argument("beta", beta.real());
        cmd.get_cmd_line_argument("beta_i", beta.imag());

        cmd.get_cmd_line_argument("iterations", iterations);
    }

    std::ostream& print_usage(std::ostream& out) const {
        out << "11_planar_complex_array example\n\n"
            << "  This example uses the CUTLASS Library to execute Planar "
               "Complex Array GEMM computations.\n\n"
            << "Options:\n\n"
            << "  --help                      If specified, displays this "
               "usage statement.\n\n"
            << "  --m <int>                   GEMM M dimension\n"
            << "  --n <int>                   GEMM N dimension\n"
            << "  --k <int>                   GEMM K dimension\n"
            << "  --batch <int>               Number of GEMM operations "
               "executed in one batch\n"
            << "  --alpha <f32>               Epilogue scalar alpha (real "
               "part)\n"
            << "  --alpha_i <f32>             Epilogue scalar alpha (imaginary "
               "part)\n"
            << "  --beta <f32>                Epilogue scalar beta (real "
               "part)\n\n"
            << "  --beta_i <f32>              Epilogue scalar beta (imaginary "
               "part)\n\n"
            << "  --iterations <int>          Number of profiling iterations "
               "to perform.\n";

        out << "\n\nExamples:\n\n"
            << "$ "
               "./examples/11_planar_complex_array/11_planar_complex_array\n\n";

        return out;
    }

    double gflops(double runtime_s) const {
        int64_t fmas = problem_size.product() * batch_count * 4;

        return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
    }
};


class TestbedPlanarComplex {
public:
    using Element = cutlass::half_t;

    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using ElementCompute = float;
    using ElementAccumulator = float;


    cutlass::library::Handle handle;

    cutlass::gemm::GemmCoord problem_size;
    int batch_count;
    cutlass::DeviceAllocation<Element> tensor_A;
    cutlass::DeviceAllocation<Element> tensor_B;
    cutlass::DeviceAllocation<Element> tensor_C;
    cutlass::DeviceAllocation<Element> tensor_D;
    cutlass::DeviceAllocation<Element> tensor_D_ref;

    cutlass::DeviceAllocation<void*> ptr_A_real;
    cutlass::DeviceAllocation<void*> ptr_A_imag;
    cutlass::DeviceAllocation<void*> ptr_B_real;
    cutlass::DeviceAllocation<void*> ptr_B_imag;
    cutlass::DeviceAllocation<void*> ptr_C_real;
    cutlass::DeviceAllocation<void*> ptr_C_imag;
    cutlass::DeviceAllocation<void*> ptr_D_real;
    cutlass::DeviceAllocation<void*> ptr_D_imag;


    TestbedPlanarComplex(Options const& options)
            : problem_size(options.problem_size),
              batch_count(options.batch_count) {
        tensor_A.reset(int64_t(problem_size.m()) * problem_size.k() *
                       batch_count * 2);
        tensor_B.reset(int64_t(problem_size.k()) * problem_size.n() *
                       batch_count * 2);
        tensor_C.reset(int64_t(problem_size.m()) * problem_size.n() *
                       batch_count * 2);
        tensor_D.reset(int64_t(problem_size.m()) * problem_size.n() *
                       batch_count * 2);
        tensor_D_ref.reset(int64_t(problem_size.m()) * problem_size.n() *
                           batch_count * 2);

        ptr_A_real.reset(batch_count);
        ptr_A_imag.reset(batch_count);
        ptr_B_real.reset(batch_count);
        ptr_B_imag.reset(batch_count);
        ptr_C_real.reset(batch_count);
        ptr_C_imag.reset(batch_count);
        ptr_D_real.reset(batch_count);
        ptr_D_imag.reset(batch_count);
    }

    void initialize() {
        uint64_t seed = 1073;

        int scope_max = 6;
        int scope_min = -6;

        cutlass::reference::device::BlockFillRandomUniform(
                tensor_A.get(), tensor_A.size(), seed, Element(scope_max),
                Element(scope_min), 0);

        cutlass::reference::device::BlockFillRandomUniform(
                tensor_B.get(), tensor_B.size(), seed * 2019,
                Element(scope_max), Element(scope_min), 0);

        cutlass::reference::device::BlockFillRandomUniform(
                tensor_C.get(), tensor_C.size(), seed * 2020,
                Element(scope_max), Element(scope_min), 0);
    }

    Result profile(Options const& options) {
        Result result;

        initialize();

        Element* ptr_A = tensor_A.get();
        Element* ptr_B = tensor_B.get();
        Element* ptr_C = tensor_C.get();
        Element* ptr_D = tensor_D.get();

        int64_t batch_stride_A =
                int64_t(problem_size.m()) * problem_size.k() * 2;
        int64_t batch_stride_B =
                int64_t(problem_size.k()) * problem_size.n() * 2;
        int64_t batch_stride_C =
                int64_t(problem_size.m()) * problem_size.n() * 2;
        int64_t batch_stride_D =
                int64_t(problem_size.m()) * problem_size.n() * 2;

        int lda =
                LayoutA::packed({problem_size.m(), problem_size.k()}).stride(0);
        int ldb =
                LayoutB::packed({problem_size.k(), problem_size.n()}).stride(0);
        int ldc =
                LayoutC::packed({problem_size.m(), problem_size.n()}).stride(0);
        int ldd =
                LayoutC::packed({problem_size.m(), problem_size.n()}).stride(0);

        int64_t imag_stride_A = int64_t(problem_size.m()) * problem_size.k();
        int64_t imag_stride_B = int64_t(problem_size.k()) * problem_size.n();
        int64_t imag_stride_C = int64_t(problem_size.m()) * problem_size.n();
        int64_t imag_stride_D = int64_t(problem_size.m()) * problem_size.n();


        struct {
            Element* base;
            void** ptr_real;
            void** ptr_imag;
            int64_t batch_stride;
            int64_t imag_stride;
        } tensors[] = {{tensor_A.get(), ptr_A_real.get(), ptr_A_imag.get(),
                        batch_stride_A, imag_stride_A},
                       {tensor_B.get(), ptr_B_real.get(), ptr_B_imag.get(),
                        batch_stride_B, imag_stride_B},
                       {tensor_C.get(), ptr_C_real.get(), ptr_C_imag.get(),
                        batch_stride_C, imag_stride_C},
                       {tensor_D.get(), ptr_D_real.get(), ptr_D_imag.get(),
                        batch_stride_D, imag_stride_D}};

        for (auto const& tensor : tensors) {
            for (int idx = 0; idx < batch_count; ++idx) {
                void* ptr_real = tensor.base + idx * tensor.batch_stride;
                void* ptr_imag = tensor.base + idx * tensor.batch_stride +
                                 tensor.imag_stride;

                cudaError_t error =
                        cudaMemcpy(tensor.ptr_real + idx, &ptr_real,
                                   sizeof(void*), cudaMemcpyHostToDevice);

                if (error != cudaSuccess) {
                    throw std::runtime_error(
                            "Failed to copy pointer to device memory");
                }

                error = cudaMemcpy(tensor.ptr_imag + idx, &ptr_imag,
                                   sizeof(void*), cudaMemcpyHostToDevice);

                if (error != cudaSuccess) {
                    throw std::runtime_error(
                            "Failed to copy pointer to device memory");
                }
            }
        }


        cudaEvent_t events[2];

        for (auto& event : events) {
            result.error = cudaEventCreate(&event);
            if (result.error != cudaSuccess) {
                std::cerr << "cudaEventCreate() failed: "
                          << cudaGetErrorString(result.error) << std::endl;
                return -1;
            }
        }

        result.error = cudaEventRecord(events[0]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventRecord() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }


        for (int iter = 0; iter < options.iterations; ++iter) {

            result.status = handle.gemm_planar_complex_array(

                    problem_size.m(),
                    problem_size.n(),
                    problem_size.k(),
                    batch_count,

                    nullptr, nullptr, nullptr,

                    cutlass::library::NumericTypeID::kF32,
                    cutlass::library::NumericTypeID::
                            kF32,

                    &options.alpha,

                    cutlass::library::NumericTypeID::
                            kF16,
                    cutlass::library::LayoutTypeID::kColumnMajor,
                    cutlass::library::ComplexTransform::
                            kConjugate,

                    ptr_A_real.get(),
                    ptr_A_imag.get(),

                    lda,
                    lda,

                    cutlass::library::NumericTypeID::
                            kF16,
                    cutlass::library::LayoutTypeID::kColumnMajor,
                    cutlass::library::ComplexTransform::
                            kNone,

                    ptr_B_real.get(),
                    ptr_B_imag.get(),

                    ldb,
                    ldb,

                    &options.beta,

                    cutlass::library::NumericTypeID::kF16,

                    ptr_C_real.get(),
                    ptr_C_imag.get(),

                    ldc,
                    ldc,

                    ptr_D_real.get(),
                    ptr_D_imag.get(),

                    ldd,
                    ldd
            );

            if (result.status != cutlass::Status::kSuccess) {
                std::cerr << "CUTLASS internal error - configuration not "
                             "supported"
                          << std::endl;
                return result;
            }
        }


        result.error = cudaEventRecord(events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventRecord() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        result.error = cudaEventSynchronize(events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventSynchronize() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        float runtime_ms = 0;
        result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventElapsed() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        result.runtime_ms = double(runtime_ms) / double(options.iterations);
        result.gflops = options.gflops(result.runtime_ms / 1000.0);

        for (auto event : events) {
            (void)cudaEventDestroy(event);
        }

        if (handle.get_last_operation()) {
            std::cout << "Recently executed '"
                      << handle.get_last_operation()->description().name << "'"
                      << std::endl;
        }


        if (options.reference_check) {
            result.passed = true;

            for (int64_t idx = 0; result.passed && idx < int64_t(batch_count);
                 ++idx) {
                cutlass::reference::device::GemmPlanarComplex<
                        Element, LayoutA, Element, LayoutB, Element, LayoutC,
                        ElementAccumulator>(
                        problem_size, options.alpha,
                        {tensor_A.get() + idx * batch_stride_A, lda,
                         imag_stride_A},
                        cutlass::ComplexTransform::kConjugate,
                        {tensor_B.get() + idx * batch_stride_B, ldb,
                         imag_stride_B},
                        cutlass::ComplexTransform::kNone, options.beta,
                        {tensor_C.get() + idx * batch_stride_C, ldc,
                         imag_stride_C},
                        {tensor_D_ref.get() + idx * batch_stride_D, ldd,
                         imag_stride_D});

                Element epsilon = 0.1_hf;
                Element nonzero_floor = 0.1_hf;

                result.passed =
                        cutlass::reference::device::BlockCompareRelativelyEqual(
                                tensor_D.get() + idx * batch_stride_D,
                                tensor_D_ref.get() + idx * batch_stride_D,
                                batch_stride_D, epsilon, nonzero_floor);
            }

            if (result.passed) {
                std::cout << "Reference check passed." << std::endl;
            } else {
                std::cerr << "Error - reference check failed." << std::endl;
            }
        }

        std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
        std::cout << " GFLOPs: " << result.gflops << std::endl;

        return result;
    }
};


int main(int argc, char const** args) {

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: "
                  << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (props.major < 7) {
        std::cerr << "Tensor Core operations must be run on a machine with "
                     "compute capability at least 70."
                  << std::endl;

        return 0;
    } else if (props.major == 7 && props.minor <= 2) {
        if (!(__CUDACC_VER_MAJOR__ > 10 ||
              (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
            std::cerr << "Volta Tensor Core operations must be compiled with "
                         "CUDA 10.1 Toolkit or later."
                      << std::endl;

            return 0;
        }
    } else if (props.major == 7 && props.minor >= 5) {
        if (!(__CUDACC_VER_MAJOR__ > 10 ||
              (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
            std::cerr << "Turing Tensor Core operations must be compiled with "
                         "CUDA 10.2 Toolkit or later."
                      << std::endl;

            return 0;
        }
    } else {
    }


    Options options;

    options.parse(argc, args);

    if (options.help) {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    if (!options.valid()) {
        std::cerr << "Invalid problem." << std::endl;
        return -1;
    }

    TestbedPlanarComplex testbed(options);

    Result result = testbed.profile(options);

    return result.passed ? 0 : -1;
}

