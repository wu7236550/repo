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

#include "cutlass/array.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_gaussian_complex_tensor_op_tile_iterator_sm80.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename Shape_,
        typename RealElementA,
        typename LayoutA_,
        typename RealElementB,
        typename LayoutB_,
        typename RealElementC,
        typename LayoutC_,
        typename Policy_,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone,
        typename Enable = bool>
class MmaGaussianComplexTensorOp;


template <
        typename Shape_,
        typename RealElementA,
        typename LayoutA_,
        typename RealElementB,
        typename LayoutB_,
        typename RealElementC,
        typename LayoutC_,
        typename Policy_,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Enable>
class MmaGaussianComplexTensorOp<Shape_, complex<RealElementA>, LayoutA_,
                                 complex<RealElementB>, LayoutB_,
                                 complex<RealElementC>, LayoutC_, Policy_,
                                 TransformA, TransformB, Enable> {
public:
    using Shape = Shape_;

    using ElementA = complex<RealElementA>;

    using LayoutA = LayoutA_;

    using ElementB = complex<RealElementB>;

    using LayoutB = LayoutB_;

    using ElementC = complex<RealElementC>;

    using LayoutC = LayoutC_;

    using Policy = Policy_;

    using ArchMmaOperator = typename Policy::Operator;

    using InstructionShape = typename ArchMmaOperator::Shape;

    using ArchTag = typename ArchMmaOperator::ArchTag;

    using OperatorClass = arch::OpClassTensorOp;

    static ComplexTransform const kTransformA = TransformA;

    static ComplexTransform const kTransformB = TransformB;

    static int const kThreadCount = 32;

public:
    using IteratorA = MmaTensorOpMultiplicandTileIterator<
            MatrixShape<Shape::kM, Shape::kK>, Operand::kA, ElementA, LayoutA,
            MatrixShape<ArchMmaOperator::Shape::kM, ArchMmaOperator::Shape::kK>,
            Policy::OpDelta::kRow, 32, 1>;

    using FragmentA = typename IteratorA::Fragment;

    using TransformedFragmentA = FragmentA;

    using IteratorB = MmaTensorOpMultiplicandTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
            MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
            Policy::OpDelta::kColumn, 32, 1>;

    using FragmentB = typename IteratorB::Fragment;

    using TransformedFragmentB = FragmentB;

    static_assert(
            !(Shape::kM % ArchMmaOperator::Shape::kM) &&
                    !(Shape::kN % ArchMmaOperator::Shape::kN),
            "Shape of warp-level Mma must be divisible by operator shape.");

    using MmaIterations = MatrixShape<Shape::kM / ArchMmaOperator::Shape::kM,
                                      Shape::kN / ArchMmaOperator::Shape::kN>;

    using IteratorC = MmaTensorOpGaussianComplexAccumulatorTileIterator<
            MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
            typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

    using FragmentC = typename IteratorC::Fragment;

    static_assert(FragmentC::kElements ==
                          3 * MmaIterations::kCount *
                                  ArchMmaOperator::FragmentC::kElements,
                  "Unexpected gaussian complex fragment length.");

private:

    ArchMmaOperator mma;

public:

    CUTLASS_DEVICE
    MmaGaussianComplexTensorOp() {}

    CUTLASS_DEVICE
    void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B,
                    FragmentC const& C) const {
        using MmaOperandA = typename ArchMmaOperator::FragmentA;
        using MmaOperandB = typename ArchMmaOperator::FragmentB;
        using MmaOperandC = typename ArchMmaOperator::FragmentC;

        static_assert(MmaOperandA::kElements == 1,
                      "This implementation only supports math instructions in "
                      "which exactly one element is needed for the A operand."
                      "We can geneneralize later.");

        static_assert(MmaOperandB::kElements == 1,
                      "This implementation only supports math instructions in "
                      "which exactly one element is needed for the B operand."
                      "We can geneneralize later.");

        D = C;

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MmaIterations::kRow; ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaIterations::kColumn; ++n) {
                MmaOperandA operand_Asum;
                MmaOperandB operand_Br;

                operand_Asum[0] = A[m].real() +
                                  ((kTransformA == ComplexTransform::kConjugate)
                                           ? -A[m].imag()
                                           : +A[m].imag());
                operand_Br[0] = B[n].real();

                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow);

                mma(*accum, operand_Asum, operand_Br, *accum);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {
                MmaOperandA operand_Ar;
                MmaOperandB operand_Bdiff;

                operand_Ar[0] = -A[m].real();
                operand_Bdiff[0] = B[n].real() - ((kTransformB ==
                                                   ComplexTransform::kConjugate)
                                                          ? -B[n].imag()
                                                          : +B[n].imag());

                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow) +
                                     MmaIterations::kCount;

                mma(*accum, operand_Ar, operand_Bdiff, *accum);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaIterations::kColumn; ++n) {
                MmaOperandA operand_Ai;
                MmaOperandB operand_Bsum;

                operand_Ai[0] = (kTransformA == ComplexTransform::kConjugate)
                                        ? -A[m].imag()
                                        : +A[m].imag();
                operand_Bsum[0] = B[n].real() +
                                  ((kTransformB == ComplexTransform::kConjugate)
                                           ? -B[n].imag()
                                           : +B[n].imag());

                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow) +
                                     2 * MmaIterations::kCount;

                mma(*accum, operand_Ai, operand_Bsum, *accum);
            }
        }
    }

    CUTLASS_DEVICE
    void transform(TransformedFragmentA& dst_A, TransformedFragmentB& dst_B,
                   FragmentA const& A, FragmentB const& B) const {
        dst_A = A;
        dst_B = B;
    }
};



}
}
}

