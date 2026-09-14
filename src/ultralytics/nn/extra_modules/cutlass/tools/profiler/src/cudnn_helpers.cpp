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
#if CUTLASS_ENABLE_CUDNN

#include <stdexcept>

#include "cudnn_helpers.h"

namespace cutlass {
namespace profiler {

Status get_cutlass_status(cudnnStatus_t cudnn_status) {
    if (cudnn_status == CUDNN_STATUS_SUCCESS) {
        return Status::kSuccess;
    } else if (cudnn_status == CUDNN_STATUS_INVALID_VALUE) {
        return Status::kErrorInvalidProblem;
    }
    if (cudnn_status == CUDNN_STATUS_NOT_SUPPORTED) {
        return Status::kErrorNotSupported;
    }
    return Status::kErrorInternal;
}

Disposition get_cutlass_disposition(cudnnStatus_t cudnn_status) {
    if (cudnn_status == CUDNN_STATUS_INVALID_VALUE) {
        return Disposition::kInvalidProblem;
    } else if (cudnn_status == CUDNN_STATUS_NOT_SUPPORTED) {
        return Disposition::kNotSupported;
    }
    return Disposition::kFailed;
}

Status checkCudnnErr(cudnnStatus_t cudnn_status) {
    Status cutlass_status = get_cutlass_status(cudnn_status);
    if (cutlass_status != Status::kSuccess) {
        throw std::runtime_error("checkCudnnErr failed");
    }
    return cutlass_status;
}

bool get_cudnn_conv_mode(cudnnConvolutionMode_t& cudnn_conv_mode,
                         conv::Mode conv_mode) {
    switch (conv_mode) {
        case conv::Mode::kCrossCorrelation:
            cudnn_conv_mode = CUDNN_CROSS_CORRELATION;
            return true;
        case conv::Mode::kConvolution:
            cudnn_conv_mode = CUDNN_CONVOLUTION;
            return true;
        default:
            break;
    }
    return false;
}

bool get_cudnn_layout(cudnnTensorFormat_t& cudnn_layout,
                      library::LayoutTypeID layout) {
    switch (layout) {
        case library::LayoutTypeID::kTensorNCHW:
        case library::LayoutTypeID::kTensorNCDHW:
            cudnn_layout = CUDNN_TENSOR_NCHW;
            return true;
        case library::LayoutTypeID::kTensorNHWC:
        case library::LayoutTypeID::kTensorNDHWC:
            cudnn_layout = CUDNN_TENSOR_NHWC;
            return true;
        default:
            break;
    }
    return false;
}

bool get_cudnn_datatype(cudnnDataType_t& cudnn_element_type,
                        library::NumericTypeID element_type) {
    switch (element_type) {
        case library::NumericTypeID::kF16:
            cudnn_element_type = CUDNN_DATA_HALF;
            return true;

        case library::NumericTypeID::kF32:
            cudnn_element_type = CUDNN_DATA_FLOAT;
            return true;

        case library::NumericTypeID::kF64:
            cudnn_element_type = CUDNN_DATA_DOUBLE;
            return true;

        case library::NumericTypeID::kS2:
            break;

        case library::NumericTypeID::kS4:
            break;

        case library::NumericTypeID::kS8:
            cudnn_element_type = CUDNN_DATA_INT8;
            return true;

        case library::NumericTypeID::kS16:
            break;

        case library::NumericTypeID::kS32:
            cudnn_element_type = CUDNN_DATA_INT32;
            return true;

        case library::NumericTypeID::kS64:
            break;

        case library::NumericTypeID::kU2:
            break;

        case library::NumericTypeID::kU4:
            break;

        case library::NumericTypeID::kU8:
            cudnn_element_type = CUDNN_DATA_UINT8;
            return true;

        case library::NumericTypeID::kU16:
            break;

        case library::NumericTypeID::kU32:
            break;

        case library::NumericTypeID::kU64:
            break;

        case library::NumericTypeID::kB1:
            break;

        case library::NumericTypeID::kInvalid:

        default:
            break;
    }

    return false;
}

bool get_cudnn_mathtype(cudnnMathType_t& cudnn_math_type,
                        library::ConvDescription const& conv_desc) {
    switch (conv_desc.tile_description.math_instruction.opcode_class) {
        case library::OpcodeClassID::kTensorOp: {
            cudnn_math_type = CUDNN_TENSOR_OP_MATH;

            library::MathOperationID math_op =
                    conv_desc.tile_description.math_instruction.math_operation;

            if (math_op == library::MathOperationID::kMultiplyAddFastF16 ||
                math_op == library::MathOperationID::kMultiplyAddFastBF16) {
                cudnn_math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
            }

            return true;
        }
        case library::OpcodeClassID::kSimt:
            return false;
    }

    return false;
}

float cast_cudnn_compute_type_to_float(library::NumericTypeID type,
                                       void const* src) {
    switch (type) {
        case library::NumericTypeID::kF16: {
            return float(*(static_cast<half_t const*>(src)));
        }
        case library::NumericTypeID::kF32: {
            return float(*(static_cast<float const*>(src)));
        }
        case library::NumericTypeID::kS32: {
            return float(*(static_cast<int const*>(src)));
        }
        default:
            throw std::runtime_error(
                    "Data type handled in cast_compute_type_to_float");
    }
}

Status cudnn_satisfies(library::ConvDescription const& desc,
                       library::Conv2dConfiguration const& configuration) {
    auto const& a_tensor = desc.A;
    auto const& b_tensor = desc.B;
    auto const& c_tensor = desc.C;
    auto const& math_instruction = desc.tile_description.math_instruction;

    if (a_tensor.element != b_tensor.element) {
        return Status::kErrorInvalidDataType;
    }


    int cudnn_output_p = ((configuration.problem_size.H +
                           2 * configuration.problem_size.pad_h -
                           ((configuration.problem_size.R - 1) *
                                    configuration.problem_size.dilation_h +
                            1)) / (configuration.problem_size.stride_h) +
                          1);

    if (cudnn_output_p != configuration.problem_size.P) {
        return Status::kErrorInvalidProblem;
    }

    int cudnn_output_q = ((configuration.problem_size.W +
                           2 * configuration.problem_size.pad_w -
                           ((configuration.problem_size.S - 1) *
                                    configuration.problem_size.dilation_w +
                            1)) / (configuration.problem_size.stride_w) +
                          1);

    if (cudnn_output_q != configuration.problem_size.Q) {
        return Status::kErrorInvalidProblem;
    }

    if (a_tensor.element == library::NumericTypeID::kF16 &&
        b_tensor.element == library::NumericTypeID::kF16 &&
        math_instruction.element_accumulator == library::NumericTypeID::kF32 &&
        c_tensor.element == library::NumericTypeID::kF32) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kBF16 ||
        b_tensor.element == library::NumericTypeID::kBF16 ||
        c_tensor.element == library::NumericTypeID::kBF16) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kTF32 ||
        b_tensor.element == library::NumericTypeID::kTF32 ||
        c_tensor.element == library::NumericTypeID::kTF32) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kS8 ||
        b_tensor.element == library::NumericTypeID::kS8 ||
        c_tensor.element == library::NumericTypeID::kS8) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kU8 ||
        b_tensor.element == library::NumericTypeID::kU8 ||
        c_tensor.element == library::NumericTypeID::kU8) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kS4 ||
        b_tensor.element == library::NumericTypeID::kS4 ||
        c_tensor.element == library::NumericTypeID::kS4) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kU4 ||
        b_tensor.element == library::NumericTypeID::kU4 ||
        c_tensor.element == library::NumericTypeID::kU4) {
        return Status::kErrorNotSupported;
    }

    return Status::kSuccess;
}


Status cudnn_satisfies(library::ConvDescription const& desc,
                       library::Conv3dConfiguration const& configuration) {
    auto const& a_tensor = desc.A;
    auto const& b_tensor = desc.B;
    auto const& c_tensor = desc.C;
    auto const& math_instruction = desc.tile_description.math_instruction;

    if (a_tensor.element != b_tensor.element) {
        return Status::kErrorInvalidDataType;
    }


    int cudnn_output_z = ((configuration.problem_size.D +
                           2 * configuration.problem_size.pad_d -
                           ((configuration.problem_size.T - 1) *
                                    configuration.problem_size.dilation_d +
                            1)) / (configuration.problem_size.stride_d) +
                          1);

    if (cudnn_output_z != configuration.problem_size.Z) {
        return Status::kErrorInvalidProblem;
    }

    int cudnn_output_p = ((configuration.problem_size.H +
                           2 * configuration.problem_size.pad_h -
                           ((configuration.problem_size.R - 1) *
                                    configuration.problem_size.dilation_h +
                            1)) / (configuration.problem_size.stride_h) +
                          1);

    if (cudnn_output_p != configuration.problem_size.P) {
        return Status::kErrorInvalidProblem;
    }

    int cudnn_output_q = ((configuration.problem_size.W +
                           2 * configuration.problem_size.pad_w -
                           ((configuration.problem_size.S - 1) *
                                    configuration.problem_size.dilation_w +
                            1)) / (configuration.problem_size.stride_w) +
                          1);

    if (cudnn_output_q != configuration.problem_size.Q) {
        return Status::kErrorInvalidProblem;
    }

    if (a_tensor.element == library::NumericTypeID::kF16 &&
        b_tensor.element == library::NumericTypeID::kF16 &&
        math_instruction.element_accumulator == library::NumericTypeID::kF32 &&
        c_tensor.element == library::NumericTypeID::kF32) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kBF16 ||
        b_tensor.element == library::NumericTypeID::kBF16 ||
        c_tensor.element == library::NumericTypeID::kBF16) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kTF32 ||
        b_tensor.element == library::NumericTypeID::kTF32 ||
        c_tensor.element == library::NumericTypeID::kTF32) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kS8 ||
        b_tensor.element == library::NumericTypeID::kS8 ||
        c_tensor.element == library::NumericTypeID::kS8) {
        return Status::kErrorNotSupported;
    }

    if (a_tensor.element == library::NumericTypeID::kS4 ||
        b_tensor.element == library::NumericTypeID::kS4 ||
        c_tensor.element == library::NumericTypeID::kS4) {
        return Status::kErrorNotSupported;
    }

    return Status::kSuccess;
}


}
}

#endif
