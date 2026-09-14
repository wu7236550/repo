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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_complex_tensor_op.h"
#include "cutlass/gemm/warp/mma_gaussian_complex_tensor_op.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename WarpShape_,
        typename InstructionShape_,
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone,
        typename Operator_ = arch::OpMultiplyAddComplex>
struct DefaultMmaComplexTensorOp;


template <
        typename WarpShape_,
        typename InstructionShape_,
        typename RealElementA,
        typename LayoutA,
        typename RealElementB,
        typename LayoutB,
        typename RealElementC,
        typename LayoutC,
        ComplexTransform TransformA,
        ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
        WarpShape_, InstructionShape_, complex<RealElementA>, LayoutA,
        complex<RealElementB>, LayoutB, complex<RealElementC>, LayoutC,
        TransformA, TransformB, arch::OpMultiplyAddComplex> {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<InstructionShape_, 32, RealElementA,
                               cutlass::layout::RowMajor, RealElementB,
                               cutlass::layout::ColumnMajor, RealElementC,
                               cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaComplexTensorOp<
            WarpShape_, complex<RealElementA>, LayoutA, complex<RealElementB>,
            LayoutB, complex<RealElementC>, LayoutC, Policy, TransformA,
            TransformB>;
};


template <
        typename WarpShape_,
        typename InstructionShape_,
        typename RealElementA,
        typename LayoutA,
        typename RealElementB,
        typename LayoutB,
        typename RealElementC,
        typename LayoutC,
        ComplexTransform TransformA,
        ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<
        WarpShape_, InstructionShape_, complex<RealElementA>, LayoutA,
        complex<RealElementB>, LayoutB, complex<RealElementC>, LayoutC,
        TransformA, TransformB, arch::OpMultiplyAddGaussianComplex> {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<InstructionShape_, 32, RealElementA,
                               cutlass::layout::RowMajor, RealElementB,
                               cutlass::layout::ColumnMajor, RealElementC,
                               cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaGaussianComplexTensorOp<
            WarpShape_, complex<RealElementA>, LayoutA, complex<RealElementB>,
            LayoutB, complex<RealElementC>, LayoutC, Policy, TransformA,
            TransformB>;
};

template <
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutA,
        typename LayoutB,
        typename LayoutC,
        ComplexTransform TransformA,
        ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, complex<float>,
                                 LayoutA, complex<float>, LayoutB,
                                 complex<float>, LayoutC, TransformA,
                                 TransformB, arch::OpMultiplyAddComplex> {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<InstructionShape_, 32, tfloat32_t,
                               cutlass::layout::RowMajor, tfloat32_t,
                               cutlass::layout::ColumnMajor, float,
                               cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaComplexTensorOp<
            WarpShape_, complex<float>, LayoutA, complex<float>, LayoutB,
            complex<float>, LayoutC, Policy, TransformA, TransformB>;
};

template <
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutA,
        typename LayoutB,
        typename LayoutC,
        ComplexTransform TransformA,
        ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, complex<float>,
                                 LayoutA, complex<float>, LayoutB,
                                 complex<float>, LayoutC, TransformA,
                                 TransformB, arch::OpMultiplyAddFastBF16> {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<InstructionShape_, 32, bfloat16_t,
                               cutlass::layout::RowMajor, bfloat16_t,
                               cutlass::layout::ColumnMajor, float,
                               cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaComplexTensorOp<
            WarpShape_, complex<float>, LayoutA, complex<float>, LayoutB,
            complex<float>, LayoutC, Policy, TransformA, TransformB>;
};

template <
        typename WarpShape_,
        typename InstructionShape_,
        typename LayoutA,
        typename LayoutB,
        typename LayoutC,
        ComplexTransform TransformA,
        ComplexTransform TransformB>
struct DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, complex<float>,
                                 LayoutA, complex<float>, LayoutB,
                                 complex<float>, LayoutC, TransformA,
                                 TransformB, arch::OpMultiplyAddFastF16> {
    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
            cutlass::arch::Mma<InstructionShape_, 32, half_t,
                               cutlass::layout::RowMajor, half_t,
                               cutlass::layout::ColumnMajor, float,
                               cutlass::layout::RowMajor, arch::OpMultiplyAdd>,
            cutlass::MatrixShape<1, 1> >;

    using Type = cutlass::gemm::warp::MmaComplexTensorOp<
            WarpShape_, complex<float>, LayoutA, complex<float>, LayoutB,
            complex<float>, LayoutC, Policy, TransformA, TransformB>;
};

}
}
}
