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
#if CUTLASS_ENABLE_CUDNN
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/library/library.h"
#include "enumerated_types.h"


namespace cutlass {
namespace profiler {

Status get_cutlass_status(cudnnStatus_t cudnn_status);

Disposition get_cutlass_disposition(cudnnStatus_t cudnn_status);

Status checkCudnnErr(cudnnStatus_t cudnn_status);

bool get_cudnn_conv_mode(cudnnConvolutionMode_t& cudnn_conv_mode,
                         conv::Mode conv_mode);

bool get_cudnn_layout(cudnnTensorFormat_t& cudnn_layout,
                      library::LayoutTypeID layout);

bool get_cudnn_datatype(cudnnDataType_t& cudnn_element_type,
                        library::NumericTypeID element_type);

bool get_cudnn_mathtype(cudnnMathType_t& cudnn_math_type,
                        library::ConvDescription const& conv_desc);

Status cudnn_satisfies(library::ConvDescription const& desc,
                       library::Conv2dConfiguration const& configuration);

Status cudnn_satisfies(library::ConvDescription const& desc,
                       library::Conv3dConfiguration const& configuration);

float cast_cudnn_compute_type_to_float(library::NumericTypeID type,
                                       void const* src);

class CudnnCreate {
private:
    cudnnHandle_t handle;
    cudnnStatus_t status;

public:
    CudnnCreate() { status = cudnnCreate(&handle); }

    ~CudnnCreate() { cudnnDestroy(handle); }

    operator cudnnHandle_t() const { return handle; }

    cudnnStatus_t get_cudnn_create_status() { return status; }
};

namespace detail {

struct cudnnConvDispatcher {
    library::ConvArguments arguments;
    library::ConvKind conv_kind;

    cudnnTensorDescriptor_t activation_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnDataType_t data_type_activation;
    cudnnDataType_t data_type_filter;
    cudnnDataType_t data_type_output;

    cudnnTensorFormat_t layout_activation;
    cudnnTensorFormat_t layout_filter;
    cudnnTensorFormat_t layout_output;

    cudnnConvolutionMode_t conv_mode;

    cudnnMathType_t math_type;

    cudnnDataType_t compute_type;

    float alpha;
    float beta;

    size_t workspace_size_in_bytes = 0;
    cutlass::device_memory::allocation<char> workspace;

    static cudnnConvolutionFwdAlgo_t const fprop_algo =
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    static cudnnConvolutionBwdDataAlgo_t const dgrad_algo =
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    static cudnnConvolutionBwdFilterAlgo_t const wgrad_algo =
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    Status status;



    cudnnConvDispatcher(library::ConvDescription const& op_desc,
                        library::Conv2dConfiguration configuration,
                        library::ConvArguments arguments_, cudnnHandle_t handle)
            :
              arguments(arguments_),
              conv_kind(op_desc.conv_kind),
              status(Status::kSuccess) {
        bool good = true;

        good = (good &&
                get_cudnn_datatype(data_type_activation, op_desc.A.element));
        good = (good &&
                get_cudnn_datatype(data_type_filter, op_desc.B.element));
        good = (good &&
                get_cudnn_datatype(data_type_output, op_desc.C.element));
        good = (good && get_cudnn_layout(layout_activation, op_desc.A.layout));
        good = (good && get_cudnn_layout(layout_filter, op_desc.B.layout));
        good = (good && get_cudnn_layout(layout_output, op_desc.C.layout));
        good = (good && get_cudnn_conv_mode(conv_mode,
                                            configuration.problem_size.mode));
        good = (good && get_cudnn_mathtype(math_type, op_desc));
        good = (good &&
                get_cudnn_datatype(compute_type,
                                   op_desc.tile_description.math_instruction
                                           .element_accumulator));
        if (!good) {
            status = Status::kErrorNotSupported;
            return;
        }
        alpha = cast_cudnn_compute_type_to_float(op_desc.element_epilogue,
                                                 arguments.alpha);
        beta = cast_cudnn_compute_type_to_float(op_desc.element_epilogue,
                                                arguments.beta);

        status = get_cutlass_status(
                cudnnCreateConvolutionDescriptor(&conv_desc));

        std::vector<int> padding{configuration.problem_size.pad_h,
                                 configuration.problem_size.pad_w};
        std::vector<int> stride{configuration.problem_size.stride_h,
                                configuration.problem_size.stride_w};
        std::vector<int> dilation{configuration.problem_size.dilation_h,
                                  configuration.problem_size.dilation_w};

        status = get_cutlass_status(cudnnSetConvolutionNdDescriptor(
                conv_desc, op_desc.conv_dim, padding.data(), stride.data(),
                dilation.data(), conv_mode, compute_type));

        status = get_cutlass_status(cudnnSetConvolutionGroupCount(
                conv_desc, configuration.problem_size.groups));

        status = get_cutlass_status(
                cudnnCreateTensorDescriptor(&activation_desc));
        status = get_cutlass_status(cudnnCreateFilterDescriptor(&filter_desc));
        status = get_cutlass_status(cudnnCreateTensorDescriptor(&output_desc));

        status = get_cutlass_status(cudnnSetTensor4dDescriptor(
                activation_desc, layout_activation, data_type_activation,
                configuration.problem_size.N, configuration.problem_size.C,
                configuration.problem_size.H, configuration.problem_size.W));

        status = get_cutlass_status(cudnnSetFilter4dDescriptor(
                filter_desc, data_type_filter, layout_filter,
                configuration.problem_size.K, configuration.problem_size.C,
                configuration.problem_size.R, configuration.problem_size.S));

        status = get_cutlass_status(cudnnSetTensor4dDescriptor(
                output_desc, layout_output, data_type_output,
                configuration.problem_size.N, configuration.problem_size.K,
                configuration.problem_size.P, configuration.problem_size.Q));

        status = get_cutlass_status(
                cudnnSetConvolutionMathType(conv_desc, math_type));

        switch (conv_kind) {
            case library::ConvKind::kFprop:
                status = get_cutlass_status(
                        cudnnGetConvolutionForwardWorkspaceSize(
                                handle, activation_desc, filter_desc, conv_desc,
                                output_desc, fprop_algo,
                                &workspace_size_in_bytes));
                break;
            case library::ConvKind::kDgrad:
                status = get_cutlass_status(
                        cudnnGetConvolutionBackwardDataWorkspaceSize(
                                handle, filter_desc, output_desc, conv_desc,
                                activation_desc, dgrad_algo,
                                &workspace_size_in_bytes));
                break;
            case library::ConvKind::kWgrad:
                status = get_cutlass_status(
                        cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                handle, activation_desc, output_desc, conv_desc,
                                filter_desc, wgrad_algo,
                                &workspace_size_in_bytes));
                break;
        }

        workspace = cutlass::device_memory::allocation<char>(
                workspace_size_in_bytes);
    }

    cudnnConvDispatcher(library::ConvDescription const& op_desc,
                        library::Conv3dConfiguration configuration,
                        library::ConvArguments arguments_, cudnnHandle_t handle)
            :
              arguments(arguments_),
              conv_kind(op_desc.conv_kind),
              status(Status::kSuccess) {
        bool good = true;

        good = (good &&
                get_cudnn_datatype(data_type_activation, op_desc.A.element));
        good = (good &&
                get_cudnn_datatype(data_type_filter, op_desc.B.element));
        good = (good &&
                get_cudnn_datatype(data_type_output, op_desc.C.element));

        good = (good && get_cudnn_layout(layout_activation, op_desc.A.layout));
        good = (good && get_cudnn_layout(layout_filter, op_desc.B.layout));
        good = (good && get_cudnn_layout(layout_output, op_desc.C.layout));

        good = (good && get_cudnn_conv_mode(conv_mode,
                                            configuration.problem_size.mode));

        alpha = cast_cudnn_compute_type_to_float(op_desc.element_epilogue,
                                                 arguments.alpha);
        beta = cast_cudnn_compute_type_to_float(op_desc.element_epilogue,
                                                arguments.beta);

        good = (good &&
                get_cudnn_datatype(compute_type,
                                   op_desc.tile_description.math_instruction
                                           .element_accumulator));

        if (!good) {
            status = Status::kErrorNotSupported;
        }

        status = get_cutlass_status(
                cudnnCreateConvolutionDescriptor(&conv_desc));

        std::vector<int> padding{configuration.problem_size.pad_d,
                                 configuration.problem_size.pad_h,
                                 configuration.problem_size.pad_w};
        std::vector<int> stride{configuration.problem_size.stride_d,
                                configuration.problem_size.stride_h,
                                configuration.problem_size.stride_w};
        std::vector<int> dilation{configuration.problem_size.dilation_d,
                                  configuration.problem_size.dilation_h,
                                  configuration.problem_size.dilation_w};

        status = get_cutlass_status(cudnnSetConvolutionNdDescriptor(
                conv_desc, op_desc.conv_dim, padding.data(), stride.data(),
                dilation.data(), conv_mode, compute_type));

        status = get_cutlass_status(cudnnSetConvolutionGroupCount(
                conv_desc, configuration.problem_size.groups));

        status = get_cutlass_status(
                cudnnCreateTensorDescriptor(&activation_desc));
        status = get_cutlass_status(cudnnCreateFilterDescriptor(&filter_desc));
        status = get_cutlass_status(cudnnCreateTensorDescriptor(&output_desc));

        std::vector<int> activation_extent{
                configuration.problem_size.N, configuration.problem_size.C,
                configuration.problem_size.D, configuration.problem_size.H,
                configuration.problem_size.W};

        std::vector<int> activation_stride{
                configuration.layout_activations.stride()[3], 1,
                configuration.layout_activations.stride()[2],
                configuration.layout_activations.stride()[1],
                configuration.layout_activations.stride()[0]};

        status = get_cutlass_status(cudnnSetTensorNdDescriptor(
                activation_desc, data_type_activation, op_desc.conv_dim + 2,
                activation_extent.data(), activation_stride.data()));

        std::vector<int> filter_extent{
                configuration.problem_size.K, configuration.problem_size.C,
                configuration.problem_size.T, configuration.problem_size.R,
                configuration.problem_size.S};

        std::vector<int> filter_stride{
                configuration.layout_filters.stride()[3], 1,
                configuration.layout_filters.stride()[2],
                configuration.layout_filters.stride()[1],
                configuration.layout_filters.stride()[0]};

        status = get_cutlass_status(cudnnSetFilterNdDescriptor(
                filter_desc, data_type_filter, layout_filter,
                op_desc.conv_dim + 2, filter_extent.data()));

        std::vector<int> output_extent{
                configuration.problem_size.N, configuration.problem_size.K,
                configuration.problem_size.Z, configuration.problem_size.P,
                configuration.problem_size.Q};

        std::vector<int> output_stride{configuration.layout_output.stride()[3],
                                       1,
                                       configuration.layout_output.stride()[2],
                                       configuration.layout_output.stride()[1],
                                       configuration.layout_output.stride()[0]};

        status = get_cutlass_status(cudnnSetTensorNdDescriptor(
                output_desc, data_type_output, op_desc.conv_dim + 2,
                output_extent.data(), output_stride.data()));

        status = get_cutlass_status(
                cudnnSetConvolutionMathType(conv_desc, math_type));

        switch (conv_kind) {
            case library::ConvKind::kFprop:
                status = get_cutlass_status(
                        cudnnGetConvolutionForwardWorkspaceSize(
                                handle, activation_desc, filter_desc, conv_desc,
                                output_desc, fprop_algo,
                                &workspace_size_in_bytes));
                break;
            case library::ConvKind::kDgrad:
                status = get_cutlass_status(
                        cudnnGetConvolutionBackwardDataWorkspaceSize(
                                handle, filter_desc, output_desc, conv_desc,
                                activation_desc, dgrad_algo,
                                &workspace_size_in_bytes));
                break;
            case library::ConvKind::kWgrad:
                status = get_cutlass_status(
                        cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                handle, activation_desc, output_desc, conv_desc,
                                filter_desc, wgrad_algo,
                                &workspace_size_in_bytes));
                break;
        }

        workspace = cutlass::device_memory::allocation<char>(
                workspace_size_in_bytes);
    }

    cudnnStatus_t operator()(cudnnHandle_t handle) {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return cudnnConvolutionForward(
                        handle, &alpha, activation_desc, activation(),
                        filter_desc, filter(), conv_desc, fprop_algo,
                        workspace.get(), workspace_size_in_bytes, &beta,
                        output_desc, arguments.D);
            case library::ConvKind::kDgrad:
                return cudnnConvolutionBackwardData(
                        handle, &alpha, filter_desc, filter(), output_desc,
                        output(), conv_desc, dgrad_algo, workspace.get(),
                        workspace_size_in_bytes, &beta, activation_desc,
                        arguments.D);
            case library::ConvKind::kWgrad:
                return cudnnConvolutionBackwardFilter(
                        handle, &alpha, activation_desc, activation(),
                        output_desc, output(), conv_desc, wgrad_algo,
                        workspace.get(), workspace_size_in_bytes, &beta,
                        filter_desc, arguments.D);
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    void const* activation() const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return arguments.A;
            case library::ConvKind::kDgrad:
                return arguments.C;
            case library::ConvKind::kWgrad:
                return arguments.B;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    void const* filter() const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return arguments.B;
            case library::ConvKind::kDgrad:
                return arguments.B;
            case library::ConvKind::kWgrad:
                return arguments.C;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    void const* output() const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return arguments.C;
            case library::ConvKind::kDgrad:
                return arguments.A;
            case library::ConvKind::kWgrad:
                return arguments.A;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }
};

}
#endif
}
}
