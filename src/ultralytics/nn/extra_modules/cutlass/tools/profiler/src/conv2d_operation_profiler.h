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

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>

#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/library/handle.h"
#include "cutlass/library/manifest.h"
#include "cutlass/library/singleton.h"

#include "options.h"
#include "device_context.h"
#include "operation_profiler.h"
#include "performance_result.h"
#include "problem_space.h"
#include "reduction_operation_profiler.h"
#if CUTLASS_ENABLE_CUDNN
#include "cudnn_helpers.h"
#endif
#include "debug.h"


namespace cutlass {
namespace profiler {


class Conv2dOperationProfiler : public OperationProfiler {
public:
    struct Conv2dProblem {
        int64_t n, h, w, c, p, q, k, r, s;
        int64_t pad_h, pad_w;
        int64_t stride_h, stride_w;
        int64_t dilation_h, dilation_w;

        std::vector<uint8_t> alpha;
        std::vector<uint8_t> beta;

        library::SplitKMode split_k_mode;
        int64_t split_k_slices;

        library::ConvModeID conv_mode;

        library::Provider eq_gemm_provider;

        std::vector<uint8_t> alpha_one;
        std::vector<uint8_t> beta_zero;


        int64_t bytes(library::ConvDescription const& operation_desc) const;

        int64_t flops(library::ConvDescription const& operation_desc) const;

        void set_default_output_size() {
            p = ((h + pad_h - r * dilation_h) / stride_h) + 1;
            q = ((w + pad_w - s * dilation_w) / stride_w) + 1;
        }

        cutlass::gemm::GemmCoord eq_gemm_size(
                library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                    return cutlass::gemm::GemmCoord(int(n * p * q), int(k),
                                                    int(r * s * c));
                case library::ConvKind::kDgrad:
                    return cutlass::gemm::GemmCoord(int(n * h * w), int(c),
                                                    int(k * r * s));
                case library::ConvKind::kWgrad:
                    return cutlass::gemm::GemmCoord(int(k), int(r * s * c),
                                                    int(n * p * q));
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        std::vector<int> extent_a(library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                    return {int(n), int(h), int(w), int(c)};
                case library::ConvKind::kDgrad:
                    return {int(n), int(p), int(q), int(k)};
                case library::ConvKind::kWgrad:
                    return {int(n), int(p), int(q), int(k)};
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        std::vector<int> extent_b(library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                    return {int(k), int(r), int(s), int(c)};
                case library::ConvKind::kDgrad:
                    return {int(k), int(r), int(s), int(c)};
                case library::ConvKind::kWgrad:
                    return {int(n), int(h), int(w), int(c)};
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        std::vector<int> extent_c(library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                    return {int(n), int(p), int(q), int(k)};
                case library::ConvKind::kDgrad:
                    return {int(n), int(h), int(w), int(c)};
                case library::ConvKind::kWgrad:
                    return {int(k), int(r), int(s), int(c)};
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        library::LayoutTypeID eq_gemm_layout_a(
                library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                    return library::LayoutTypeID::kRowMajor;
                case library::ConvKind::kDgrad:
                    return library::LayoutTypeID::kRowMajor;
                case library::ConvKind::kWgrad:
                    return library::LayoutTypeID::kColumnMajor;
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        library::LayoutTypeID eq_gemm_layout_b(
                library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                    return library::LayoutTypeID::kColumnMajor;
                case library::ConvKind::kDgrad:
                    return library::LayoutTypeID::kRowMajor;
                case library::ConvKind::kWgrad:
                    return library::LayoutTypeID::kRowMajor;
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        library::LayoutTypeID eq_gemm_layout_c(
                library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                case library::ConvKind::kDgrad:
                case library::ConvKind::kWgrad:
                    return library::LayoutTypeID::kColumnMajor;
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        int64_t eq_gemm_lda(library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                    return eq_gemm_size(conv_kind).k();
                case library::ConvKind::kDgrad:
                    return eq_gemm_size(conv_kind).k();
                case library::ConvKind::kWgrad:
                    return eq_gemm_size(conv_kind).m();
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        int64_t eq_gemm_ldb(library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                    return eq_gemm_size(conv_kind).k();
                case library::ConvKind::kDgrad:
                    return eq_gemm_size(conv_kind).n();
                case library::ConvKind::kWgrad:
                    return eq_gemm_size(conv_kind).n();
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }

        int64_t eq_gemm_ldc(library::ConvKind const& conv_kind) const {
            switch (conv_kind) {
                case library::ConvKind::kFprop:
                case library::ConvKind::kDgrad:
                case library::ConvKind::kWgrad:
                    return eq_gemm_size(conv_kind).m();
                default:
                    throw std::runtime_error(
                            "Invalid Conv Operator (fprop, dgrad, wgrad)");
            }
        }
    };

    struct Conv2dWorkspace {
        DeviceAllocation* A;
        DeviceAllocation* B;
        DeviceAllocation* C;
        DeviceAllocation* Computed;
        DeviceAllocation* Reference;

        library::Conv2dConfiguration configuration;
        library::ConvArguments arguments;

        int problem_count;

        std::vector<uint8_t> host_workspace;

        DeviceAllocation device_workspace;

        library::ReductionConfiguration reduction_configuration;
        library::ReductionArguments reduction_arguments;

        std::vector<uint8_t> reduction_host_workspace;

        std::vector<uint8_t> host_tensor_a;

        std::vector<uint8_t> host_tensor_b;

        std::vector<uint8_t> host_tensor_c;


        Conv2dWorkspace()
                : A(nullptr),
                  B(nullptr),
                  C(nullptr),
                  Computed(nullptr),
                  Reference(nullptr) {}

        std::vector<int> stride_a(library::ConvKind const& conv_kind) {
            return {configuration.layout_a(conv_kind).stride()[0],
                    configuration.layout_a(conv_kind).stride()[1],
                    configuration.layout_a(conv_kind).stride()[2]};
        }

        std::vector<int> stride_b(library::ConvKind const& conv_kind) {
            return {configuration.layout_b(conv_kind).stride()[0],
                    configuration.layout_b(conv_kind).stride()[1],
                    configuration.layout_b(conv_kind).stride()[2]};
        }

        std::vector<int> stride_c(library::ConvKind const& conv_kind) {
            return {configuration.layout_c(conv_kind).stride()[0],
                    configuration.layout_c(conv_kind).stride()[1],
                    configuration.layout_c(conv_kind).stride()[2]};
        }
    };

protected:

    Conv2dProblem problem_;

    Conv2dWorkspace conv_workspace_;

    library::Operation const* reduction_op_;

public:

    Conv2dOperationProfiler(Options const& options);

    virtual ~Conv2dOperationProfiler();

    virtual void print_usage(std::ostream& out) const;

    virtual void print_examples(std::ostream& out) const;

    virtual Status initialize_configuration(
            Options const& options, PerformanceReport& report,
            DeviceContext& device_context, library::Operation const* operation,
            ProblemSpace const& problem_space,
            ProblemSpace::Problem const& problem);

    virtual Status initialize_workspace(Options const& options,
                                        PerformanceReport& report,
                                        DeviceContext& device_context,
                                        library::Operation const* operation,
                                        ProblemSpace const& problem_space,
                                        ProblemSpace::Problem const& problem);

    virtual bool verify_cutlass(Options const& options,
                                PerformanceReport& report,
                                DeviceContext& device_context,
                                library::Operation const* operation,
                                ProblemSpace const& problem_space,
                                ProblemSpace::Problem const& problem);

    virtual bool profile(Options const& options, PerformanceReport& report,
                         DeviceContext& device_context,
                         library::Operation const* operation,
                         ProblemSpace const& problem_space,
                         ProblemSpace::Problem const& problem);

protected:
    virtual Status profile_cutlass_(double& runtime, Options const& options,
                                    library::Operation const* operation,
                                    void* arguments, void* host_workspace,
                                    void* device_workspace);

    bool initialize_reduction_configuration_(
            Options const& options, PerformanceReport& report,
            DeviceContext& device_context, library::Operation const* operation,
            ProblemSpace const& problem_space,
            ProblemSpace::Problem const& problem);

    void initialize_result_(PerformanceResult& result, Options const& options,
                            library::ConvDescription const& operation_desc,
                            ProblemSpace const& problem_space);

    bool verify_with_host_reference_(Options const& options,
                                     PerformanceReport& report,
                                     DeviceContext& device_context,
                                     library::Operation const* operation,
                                     ProblemSpace const& problem_space,
                                     ProblemSpace::Problem const& problem);

    bool verify_with_device_reference_(Options const& options,
                                       PerformanceReport& report,
                                       DeviceContext& device_context,
                                       library::Operation const* operation,
                                       ProblemSpace const& problem_space,
                                       ProblemSpace::Problem const& problem);

#if CUTLASS_ENABLE_CUDNN

    bool verify_with_cudnn_(Options const& options, PerformanceReport& report,
                            DeviceContext& device_context,
                            library::Operation const* operation,
                            ProblemSpace const& problem_space,
                            ProblemSpace::Problem const& problem);

#endif
};


}
}

