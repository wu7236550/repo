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

/**
 * \file examples/99_turing_conv2ddgrad_int8/99_turing_conv2ddgrad_int8.cu
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */


#include <iostream>
#include <sstream>

#include "cutlass/convolution/device/implicit_gemm_precomp_convolution.h"
#include "cutlass/convolution/kernel/default_conv2d_dgrad.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#define INTERLEAVED_K 32

using ElementBias = int32_t;
using ElementAccumulator = int32_t;
using ElementComputeEpilogue =
        float;
using ElementInputA = int8_t;
using ElementInputB = int8_t;
using ElementOutput = int8_t;

using LayoutInputA = cutlass::layout::TensorNCxHWx<INTERLEAVED_K>;
using LayoutInputB = cutlass::layout::TensorKxRSCx<INTERLEAVED_K>;
using LayoutOutput = cutlass::layout::TensorNCxHWx<INTERLEAVED_K>;

using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm75;

using ThreadblockShape =
        cutlass::gemm::GemmShape<64, 128, 64>;

using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;

using InstructionShape =
        cutlass::gemm::GemmShape<8, 8, 16>;

using SwizzleThreadBlock =
        cutlass::conv::threadblock::ConvolutionDgradNCxHWxThreadblockSwizzle;

constexpr int NumStages = 2;

using EpilogueOp = cutlass::epilogue::thread::BiasAddLinearCombinationClamp<
        ElementOutput,
        8,
        ElementAccumulator,
        ElementBias,
        ElementComputeEpilogue>;

using Conv2dDgradKernel =
        typename cutlass::conv::kernel::DefaultConvolution2dDgrad<
                ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch,
                ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
                SwizzleThreadBlock, NumStages,
                cutlass::arch::OpMultiplyAddSaturate, 16, 16>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmPrecompConvolution<
        Conv2dDgradKernel>;


struct Options {
    bool help;
    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    bool reference_check;
    bool measure_performance;
    int iterations;
    bool save_workspace;
    ElementComputeEpilogue alpha;
    ElementComputeEpilogue beta;
    bool benchmark;
    std::string tag;

    Options()
            : help(false),
              input_size(1, 32, 32, 32),
              filter_size(32, 3, 3, 32),
              padding(1, 1, 1, 1),
              conv_stride(1, 1),
              dilation(1, 1),
              reference_check(false),
              measure_performance(true),
              iterations(20),
              save_workspace(false),
              alpha(1),
              beta(0),
              benchmark(false) {}

    bool valid() {
        int const kAlignment = 32;

        if ((input_size.c() % kAlignment) || (filter_size.n() % kAlignment)) {
            return false;
        }

        if ((padding.h() != filter_size.h() / 2) ||
            (padding.w() != filter_size.w() / 2)) {
            return false;
        }

        return true;
    }

    void update(cutlass::Tensor4DCoord input_size,
                cutlass::Tensor4DCoord filter_size,
                cutlass::MatrixCoord conv_stride) {
        this->input_size = input_size;
        this->filter_size = filter_size;
        this->conv_stride = conv_stride;

        padding.n() = filter_size.h() / 2;
        padding.h() = filter_size.h() / 2;
        padding.w() = filter_size.w() / 2;
        padding.c() = filter_size.w() / 2;
    }

    void parse(int argc, char const** args) {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")) {
            help = true;
        }

        if (cmd.check_cmd_line_flag("ref-check")) {
            reference_check = true;
        }

        if (cmd.check_cmd_line_flag("perf-check")) {
            measure_performance = true;
        }

        if (cmd.check_cmd_line_flag("save-workspace")) {
            save_workspace = true;
        }

        if (cmd.check_cmd_line_flag("benchmark")) {
            benchmark = true;
        }

        cmd.get_cmd_line_argument("n", input_size.n());
        cmd.get_cmd_line_argument("h", input_size.h());
        cmd.get_cmd_line_argument("w", input_size.w());
        cmd.get_cmd_line_argument("c", input_size.c());

        cmd.get_cmd_line_argument("k", filter_size.n());
        cmd.get_cmd_line_argument("r", filter_size.h());
        cmd.get_cmd_line_argument("s", filter_size.w());
        filter_size.c() = input_size.c();

        cmd.get_cmd_line_argument("alpha", alpha);
        cmd.get_cmd_line_argument("beta", beta);

        cmd.get_cmd_line_argument("iterations", iterations);
        cmd.get_cmd_line_argument("tag", tag);

        if (filter_size.h() == 3 && filter_size.w() == 3) {
            padding = {1, 1, 1, 1};
        } else {
            filter_size.h() = 1;
            filter_size.w() = 1;
            padding = {0, 0, 0, 0};
        }
    }

    std::ostream& print_usage(std::ostream& out) const {
        out << "09_turing_tensorop_conv2dfprop example\n\n"
            << "  This example uses Turing's Tensor Core operators on int4 "
               "data types to compute\n"
            << "  forward convolution on tensors of layout NHWC.\n\n"
            << "Options:\n\n"
            << "  --help               If specified, displays this usage "
               "statement.\n\n"
            << "  --n <int>            Input tensor extent N\n"
            << "  --h <int>            Input tensor extent H\n"
            << "  --w <int>            Input tensor extent W\n"
            << "  --c <int>            Input tensor extent C\n"
            << "  --k <int>            Filter extent K\n"
            << "  --r <int>            Filter extent R\n"
            << "  --s <int>            Filter extent S\n\n"
            << "  --alpha <float>      Epilogue scalar alpha\n"
            << "  --beta <float>       Epilogue scalar beta\n\n"
            << "  --ref-check          If set (true), reference check on the "
               "host is computed\n"
            << "  --perf-check         If set (true), performance is "
               "measured.\n"
            << "  --benchmark          If set (true), performance benchmarking "
               "on several layers and batch-size.\n"
            << "  --iterations <int>   Number of profiling iterations to "
               "perform.\n"
            << "  --save-workspace     If set, workspace is written to a text "
               "file.\n"
            << "  --tag <string>       String to replicate across the first "
               "column in the results table\n";

        out << "\n\nExamples:\n\n"
            << "$ "
               "./examples/09_turing_tensorop_conv2dfprop/"
               "09_turing_tensorop_conv2dfprop  --n=32 --h=224 --w=224 --c=128 "
               "--k=256 --r=1 --s=1\n\n"
            << "$ "
               "./examples/09_turing_tensorop_conv2dfprop/"
               "09_turing_tensorop_conv2dfprop  --n=1 --h=224 --w=224 --c=32 "
               "--k=32 --r=3 --s=3 --ref-check\n\n";

        return out;
    }

    cutlass::Tensor4DCoord output_size() const {
        return cutlass::Tensor4DCoord(
                input_size.n(),
                (input_size.h() + padding.n() + padding.h() - filter_size.h()) /
                                conv_stride.row() +
                        1,
                (input_size.w() + padding.w() + padding.c() - filter_size.w()) /
                                conv_stride.column() +
                        1,
                filter_size.n());
    }

    double gflops(double runtime_s) const {
        int64_t fmas =
                output_size().product() *
                int64_t(filter_size.h() * filter_size.w() * filter_size.c());

        return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
    }
};


struct Result {
    double runtime_ms;
    double gflops;
    cutlass::Status status;
    cutlass::Status reference_check;
    cudaError_t error;

    Result()
            : runtime_ms(0),
              gflops(0),
              status(cutlass::Status::kSuccess),
              reference_check(cutlass::Status::kInvalid),
              error(cudaSuccess) {}

    static std::ostream& print_header(std::ostream& out,
                                      Options const& options) {
        if (!options.tag.empty()) {
            out << "Name,";
        }

        out << "Layer,N,H,W,C,K,R,S,Runtime,TFLOPs";

        return out;
    }

    std::ostream& print(std::ostream& out, int idx, Options const& options) {
        if (!options.tag.empty()) {
            out << options.tag << ",";
        }

        out << "conv_" << idx << "," << options.input_size.n() << ","
            << options.input_size.h() << "," << options.input_size.w() << ","
            << options.input_size.c() << "," << options.filter_size.n() << ","
            << options.filter_size.h() << "," << options.filter_size.w() << ","
            << runtime_ms << "," << gflops / 1000.f;

        return out;
    }
};


Result profile_convolution(Options const& options) {
    Result result;

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    int split_k_slices = 1;

    cutlass::conv::Conv2dProblemSize problem_size(
            options.input_size, options.filter_size, options.padding,
            options.conv_stride, options.dilation, options.output_size(), mode,
            split_k_slices);

    static cutlass::conv::Operator const kConvolutionalOperator =
            ImplicitGemm::kConvolutionalOperator;


    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
            implicit_gemm_tensor_a_extent(kConvolutionalOperator,
                                          problem_size));
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
            implicit_gemm_tensor_b_extent(kConvolutionalOperator,
                                          problem_size));
    cutlass::HostTensor<ElementBias, LayoutOutput> tensor_bias(
            implicit_gemm_tensor_bias_extent(kConvolutionalOperator,
                                             problem_size));
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
            implicit_gemm_tensor_c_extent(kConvolutionalOperator,
                                          problem_size));
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_c(
            implicit_gemm_tensor_c_extent(kConvolutionalOperator,
                                          problem_size));


    cutlass::reference::host::TensorFillRandomUniform(
            tensor_a.host_view(), 1, ElementInputA(7), ElementInputA(-8), 0);

    cutlass::reference::host::TensorFillRandomUniform(
            tensor_b.host_view(), 1, ElementInputB(7), ElementInputB(-8), 0);

    cutlass::reference::host::TensorFill(tensor_c.host_view());

    cutlass::reference::host::TensorFill(tensor_ref_c.host_view());

    cutlass::reference::host::TensorFill(tensor_bias.host_view());

    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_ref_c.sync_device();
    tensor_bias.sync_device();


    typename ImplicitGemm::Arguments arguments{
            problem_size,
            tensor_a.device_ref(),
            tensor_b.device_ref(),
            tensor_bias.device_ref(),
            tensor_c.device_ref(),
            tensor_c.device_ref(),
            {options.alpha, 0.f, options.beta},
    };


    ImplicitGemm implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    result.status = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(result.status);

    result.status = implicit_gemm_op();

    CUTLASS_CHECK(result.status);


    if (options.reference_check) {
        std::cout << "Verification on host...\n";

        // Compute with reference implementation
        cutlass::reference::host::Conv2dDgrad<
                ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                ElementOutput, LayoutOutput, ElementComputeEpilogue,
                ElementAccumulator,
                cutlass::NumericConverterClamp<ElementOutput,
                                               ElementComputeEpilogue> >(
                problem_size, tensor_a.host_ref(), tensor_b.host_ref(),
                tensor_c.host_ref(), tensor_ref_c.host_ref(), options.alpha,
                options.beta);

        tensor_c.sync_host();

        bool passed = cutlass::reference::host::TensorEquals(
                tensor_c.host_view(), tensor_ref_c.host_view());

        if (!passed) {
            result.reference_check = cutlass::Status::kErrorInternal;
            std::cout << "ERROR - results miscompared.\n";
        } else {
            result.reference_check = cutlass::Status::kSuccess;
            std::cout << "Passed.\n";
        }
    } else {
        result.reference_check = cutlass::Status::kInvalid;
    }

    if (options.save_workspace) {
        std::stringstream ss;

        ss << "09_tensor_conv_workspace_conv2dfprop_" << options.input_size.n()
           << "x" << options.input_size.h() << "x" << options.input_size.w()
           << "x" << options.input_size.c() << "_" << options.filter_size.n()
           << "x" << options.filter_size.h() << "x" << options.filter_size.w()
           << "x" << options.filter_size.c() << ".dat";

        std::ofstream output_workspace(ss.str());

        output_workspace << "Input = \n"
                         << tensor_a.host_view() << "\n\n"
                         << "Filters = \n"
                         << tensor_b.host_view() << "\n\n";

        if (options.reference_check) {
            output_workspace << "Reference = \n"
                             << tensor_ref_c.host_view() << "\n\n";
        }

        output_workspace << "Computed = \n"
                         << tensor_c.host_view() << std::endl;

        std::cout << "Results written to '" << ss.str() << "'." << std::endl;
    }


    if (options.measure_performance) {
        cudaEvent_t events[2];

        for (auto& event : events) {
            result.error = cudaEventCreate(&event);
            if (result.error != cudaSuccess) {
                std::cerr << "cudaEventCreate() failed: "
                          << cudaGetErrorString(result.error) << std::endl;
                return result;
            }
        }

        result.error = cudaEventRecord(events[0]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventRecord() failed: "
                      << cudaGetErrorString(result.error) << std::endl;
            return result;
        }

        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            result.status = implicit_gemm_op();
            CUTLASS_CHECK(result.status);
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
    }

    return result;
}


int main(int argc, char const** args) {
    if (!(__CUDACC_VER_MAJOR__ > 10 ||
          (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
        std::cerr << "Turing Tensor Core operations must be compiled with CUDA "
                     "10.2 Toolkit or later."
                  << std::endl;
        return 0;
    }

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    if (!(props.major > 7 || (props.major == 7 && props.minor >= 5))) {
        std::cerr << "Turing Tensor Ops must be run on a machine with compute "
                     "capability at least 75."
                  << std::endl;
        return 0;
    }

    Options options;

    options.parse(argc, args);

    if (options.help) {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    if (options.benchmark) {

        int batch_sizes[] = {16};

        struct Benchmark {
            int h, w, c, k, r, s, stride_h, stride_w;
        } layers[] = {
                {92, 180, 64, 64, 4, 4, 2, 2},
        };

        Result::print_header(std::cout, options) << std::endl;

        int idx = 1;

        for (auto const& layer : layers) {
            for (auto N : batch_sizes) {
                options.update({N, layer.h, layer.w, layer.c},
                               {layer.k, layer.r, layer.s, layer.c},
                               {layer.stride_h, layer.stride_w});

                Result result = profile_convolution(options);
                result.print(std::cout, idx, options) << std::endl;
            }

            ++idx;
        }
    } else {
        if (!options.valid()) {
            std::cerr << "Invalid problem." << std::endl;
            return -1;
        }

        Result result = profile_convolution(options);

        Result::print_header(std::cout, options) << std::endl;
        result.print(std::cout, 1, options) << std::endl;
    }

    return 0;
}

