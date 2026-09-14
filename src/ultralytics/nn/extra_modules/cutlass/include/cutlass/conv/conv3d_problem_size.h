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

#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

namespace cutlass {
namespace conv {


struct Conv3dProblemSize : public Conv2dProblemSize {

    using Coord3D = Coord<3>;


    int D, T, Z;
    int pad_d;
    int stride_d;
    int dilation_d;

public:
    CUTLASS_HOST_DEVICE
    Conv3dProblemSize()
            : D(0),
              T(0),
              Z(0),
              pad_d(0),
              stride_d(1),
              dilation_d(1),
              Conv2dProblemSize() {}

    CUTLASS_HOST_DEVICE
    Conv3dProblemSize(int N, int D, int H, int W, int C, int Z, int P, int Q,
                      int K, int T, int R, int S, Mode mode)
            : D(D),
              T(T),
              Z(Z),
              pad_d(T / 2),
              stride_d(1),
              dilation_d(1),
              Conv2dProblemSize(N, H, W, C, P, Q, K, R, S, mode) {}

    CUTLASS_HOST_DEVICE
    Conv3dProblemSize(int N, int D, int H, int W, int C, int K, int T, int R,
                      int S, int Z, int P, int Q, int pad_d, int pad_h,
                      int pad_w, int stride_d, int stride_h, int stride_w,
                      int dilation_d, int dilation_h, int dilation_w, Mode mode,
                      int split_k_slices = 1, int groups = 1)
            : D(D),
              T(T),
              Z(Z),
              pad_d(pad_d),
              stride_d(stride_d),
              dilation_d(dilation_d),
              Conv2dProblemSize(N, H, W, C, K, R, S, P, Q, pad_h, pad_w,
                                stride_h, stride_w, dilation_h, dilation_w,
                                mode, split_k_slices, groups) {}

    CUTLASS_HOST_DEVICE
    Conv3dProblemSize(
            cutlass::Tensor5DCoord input_size,
            cutlass::Tensor5DCoord filter_size,
            Coord3D padding,
            Coord3D stride,
            Coord3D dilation,
            cutlass::Tensor5DCoord output_size,
            cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation,
            int split_k_slices = 1, int groups = 1)
            : D(input_size.d()),
              T(filter_size.d()),
              Z(output_size.d()),
              pad_d(padding[0]),
              stride_d(stride[0]),
              dilation_d(dilation[0]),
              Conv2dProblemSize(
                      {input_size.n(), input_size.h(), input_size.w(),
                       input_size.c()},
                      {filter_size.n(), filter_size.h(), filter_size.w(),
                       filter_size.c()},
                      {padding[1], padding[1], padding[2], padding[2]},
                      {stride[1], stride[2]}, {dilation[1], dilation[2]},
                      {output_size.n(), output_size.h(), output_size.w(),
                       output_size.c()},
                      mode, split_k_slices, groups) {}

    CUTLASS_HOST_DEVICE
    Conv3dProblemSize(
            cutlass::Tensor5DCoord input_size,
            cutlass::Tensor5DCoord filter_size,
            Coord3D padding,
            Coord3D stride,
            Coord3D dilation,
            cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation,
            int split_k_slices = 1, int groups = 1)
            : D(input_size.d()),
              T(filter_size.d()),
              pad_d(padding[0]),
              stride_d(stride[0]),
              dilation_d(dilation[0]),
              Conv2dProblemSize(
                      {input_size.n(), input_size.h(), input_size.w(),
                       input_size.c()},
                      {filter_size.n(), filter_size.h(), filter_size.w(),
                       filter_size.c()},
                      {padding[1], padding[1], padding[2], padding[2]},
                      {stride[1], stride[2]}, {dilation[1], dilation[2]}, mode,
                      split_k_slices, groups) {
        Z = ((D + pad_d - T * dilation_d) / stride_d) + 1;
    }

    CUTLASS_HOST_DEVICE
    bool operator==(Conv3dProblemSize const& conv) const {
        return ((N == conv.N) && (D == conv.D) && (H == conv.H) &&
                (W == conv.W) && (C == conv.C) && (K == conv.K) &&
                (T == conv.T) && (R == conv.R) && (S == conv.S) &&
                (Z == conv.Z) && (P == conv.P) && (Q == conv.Q) &&
                (pad_d == conv.pad_d) && (pad_h == conv.pad_h) &&
                (pad_w == conv.pad_w) && (stride_d == conv.stride_d) &&
                (stride_h == conv.stride_h) && (stride_w == conv.stride_h) &&
                (dilation_d == conv.dilation_d) &&
                (dilation_h == conv.dilation_h) &&
                (dilation_h == conv.dilation_h));
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(Conv3dProblemSize const& rhs) const {
        return !(*this == rhs);
    }

    CUTLASS_HOST_DEVICE
    Conv3dProblemSize reset_mode(cutlass::conv::Mode mode_) {
        Conv3dProblemSize tmp(*this);
        tmp.mode = mode_;
        return tmp;
    }

    CUTLASS_HOST_DEVICE
    Conv3dProblemSize reset_split_k_slices(int split_k_slices_) {
        Conv3dProblemSize tmp(*this);
        tmp.split_k_slices = split_k_slices_;
        return tmp;
    }

    CUTLASS_HOST_DEVICE
    cutlass::Tensor5DCoord activation_extent() const {
        return cutlass::Tensor5DCoord({N, D, H, W, C});
    }

    CUTLASS_HOST_DEVICE
    cutlass::Tensor5DCoord filter_extent() const {
        return cutlass::Tensor5DCoord({K, T, R, S, C});
    }

    CUTLASS_HOST_DEVICE
    cutlass::Tensor5DCoord output_extent() const {
        return cutlass::Tensor5DCoord({N, Z, P, Q, K});
    }

    CUTLASS_HOST_DEVICE
    int64_t activation_size() const { return (N * D * H * W * C); }

    CUTLASS_HOST_DEVICE
    int64_t filter_size() const { return (K * T * R * S * C); }

    CUTLASS_HOST_DEVICE
    int64_t output_size() const { return (N * Z * P * Q * K); }

    CUTLASS_HOST_DEVICE
    Coord3D padding() const { return Coord3D({pad_d, pad_h, pad_w}); }

    CUTLASS_HOST_DEVICE
    Coord3D stride() const { return Coord3D({stride_d, stride_h, stride_w}); }

    CUTLASS_HOST_DEVICE
    Coord3D dilation() const {
        return Coord3D({dilation_d, dilation_h, dilation_w});
    }
};


CUTLASS_HOST_DEVICE
cutlass::gemm::GemmCoord implicit_gemm_problem_size(
        Operator conv_operator, Conv3dProblemSize const& problem_size) {
    switch (conv_operator) {
        case Operator::kFprop:
            return gemm::GemmCoord(problem_size.N * problem_size.Z *
                                           problem_size.P * problem_size.Q,
                                   problem_size.K,
                                   problem_size.T * problem_size.R *
                                           problem_size.S * problem_size.C);
        case Operator::kDgrad:
            return gemm::GemmCoord(problem_size.N * problem_size.D *
                                           problem_size.H * problem_size.W,
                                   problem_size.C,
                                   problem_size.T * problem_size.R *
                                           problem_size.S * problem_size.K);
        case Operator::kWgrad:
            return gemm::GemmCoord(problem_size.K,
                                   problem_size.T * problem_size.R *
                                           problem_size.S * problem_size.C,
                                   problem_size.N * problem_size.Z *
                                           problem_size.P * problem_size.Q);
        default:
            break;
    }
    return gemm::GemmCoord();
}

CUTLASS_HOST_DEVICE
int implicit_gemm_k_iterations(Operator conv_operator, int threadblock_K,
                               Conv3dProblemSize const& problem_size) {
    int iterations = 0;
    int elements_per_split_k_slice = 0;

    switch (conv_operator) {
        case Operator::kFprop:
            elements_per_split_k_slice =
                    (problem_size.C + problem_size.split_k_slices - 1) /
                    problem_size.split_k_slices;
            iterations = problem_size.T * problem_size.R * problem_size.S *
                         ((elements_per_split_k_slice + threadblock_K - 1) /
                          threadblock_K);
            break;

        case Operator::kDgrad:
            elements_per_split_k_slice =
                    (problem_size.K + problem_size.split_k_slices - 1) /
                    problem_size.split_k_slices;
            iterations = problem_size.T * problem_size.R * problem_size.S *
                         ((elements_per_split_k_slice + threadblock_K - 1) /
                          threadblock_K);
            break;

        case Operator::kWgrad:
            elements_per_split_k_slice =
                    (problem_size.N * problem_size.Z * problem_size.P *
                             problem_size.Q +
                     problem_size.split_k_slices - 1) /
                    problem_size.split_k_slices;
            iterations = (elements_per_split_k_slice + threadblock_K - 1) /
                         threadblock_K;
            break;

        default:
            break;
    }

    return iterations;
}

CUTLASS_HOST_DEVICE
cutlass::Tensor5DCoord implicit_gemm_tensor_a_extent(
        Operator conv_operator, Conv3dProblemSize const& problem_size) {
    switch (conv_operator) {
        case cutlass::conv::Operator::kFprop:
            return problem_size.activation_extent();
        case cutlass::conv::Operator::kDgrad:
            return problem_size.output_extent();
        case cutlass::conv::Operator::kWgrad:
            return problem_size.output_extent();
        default:
            break;
    }
    return cutlass::Tensor5DCoord();
}

CUTLASS_HOST_DEVICE
cutlass::Tensor5DCoord implicit_gemm_tensor_b_extent(
        Operator conv_operator, Conv3dProblemSize const& problem_size) {
    switch (conv_operator) {
        case cutlass::conv::Operator::kFprop:
            return problem_size.filter_extent();
        case cutlass::conv::Operator::kDgrad:
            return problem_size.filter_extent();
        case cutlass::conv::Operator::kWgrad:
            return problem_size.activation_extent();
        default:
            break;
    }
    return cutlass::Tensor5DCoord();
}

CUTLASS_HOST_DEVICE
cutlass::Tensor5DCoord implicit_gemm_tensor_c_extent(
        Operator conv_operator, Conv3dProblemSize const& problem_size) {
    switch (conv_operator) {
        case cutlass::conv::Operator::kFprop:
            return problem_size.output_extent();
        case cutlass::conv::Operator::kDgrad:
            return problem_size.activation_extent();
        case cutlass::conv::Operator::kWgrad:
            return problem_size.filter_extent();
        default:
            break;
    }
    return cutlass::Tensor5DCoord();
}

CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_a_size(Operator conv_operator,
                                    Conv3dProblemSize const& problem_size) {
    switch (conv_operator) {
        case cutlass::conv::Operator::kFprop:
            return problem_size.activation_size();
        case cutlass::conv::Operator::kDgrad:
            return problem_size.output_size();
        case cutlass::conv::Operator::kWgrad:
            return problem_size.output_size();
        default:
            break;
    }
    return 0;
}

CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_b_size(Operator conv_operator,
                                    Conv3dProblemSize const& problem_size) {
    switch (conv_operator) {
        case cutlass::conv::Operator::kFprop:
            return problem_size.filter_size();
        case cutlass::conv::Operator::kDgrad:
            return problem_size.filter_size();
        case cutlass::conv::Operator::kWgrad:
            return problem_size.activation_size();
        default:
            break;
    }
    return 0;
}

CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_c_size(Operator conv_operator,
                                    Conv3dProblemSize const& problem_size) {
    switch (conv_operator) {
        case cutlass::conv::Operator::kFprop:
            return problem_size.output_size();
        case cutlass::conv::Operator::kDgrad:
            return problem_size.activation_size();
        case cutlass::conv::Operator::kWgrad:
            return problem_size.filter_size();
        default:
            break;
    }
    return 0;
}

}
}

