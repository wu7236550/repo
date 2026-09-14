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
#include "cutlass/functional.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"
#include "cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h"


namespace cutlass {
namespace gemm {
namespace warp {


namespace detail {

template <
        typename RealElement,
        typename DestinationFragment,
        typename SourceFragment,
        typename MmaIterations,
        typename MmaOperandShape,
        ComplexTransform Transform_,
        Operand Operand_,
        FloatRoundStyle Round_>
struct UnpackComplexConvertAndPackForMma;

template <typename RealElement, typename DestinationFragment,
          typename SourceFragment, typename MmaIterations,
          typename MmaOperandShape, ComplexTransform Transform_,
          FloatRoundStyle Round_>
struct UnpackComplexConvertAndPackForMma<
        RealElement, DestinationFragment, SourceFragment, MmaIterations,
        MmaOperandShape, Transform_, Operand::kA, Round_> {
    static Operand const kOperand = Operand::kA;
    static ComplexTransform const kTransform = Transform_;
    static FloatRoundStyle const kRound = Round_;

    using MmaElement = typename DestinationFragment::Element;

    using Converter = NumericConverter<MmaElement, RealElement, kRound>;

    using SourceFragmentLayout = layout::ColumnMajor;
    static int const kLdm = MmaIterations::kRow * MmaOperandShape::kRow;

    CUTLASS_DEVICE
    UnpackComplexConvertAndPackForMma() {}

    CUTLASS_DEVICE
    void operator()(DestinationFragment* dest, SourceFragment const& source) {
        Converter convert_op;
        SourceFragmentLayout layout(kLdm);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < MmaIterations::kRow; i++) {
            int pos = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < MmaOperandShape::kColumn; c++) {
                CUTLASS_PRAGMA_UNROLL
                for (int r = 0; r < MmaOperandShape::kRow; r++) {
                    int row = r + i * MmaOperandShape::kRow;
                    int col = c;

                    MmaElement a = convert_op(
                            source[layout(MatrixCoord{row, col})].real());
                    MmaElement b = convert_op(
                            source[layout(MatrixCoord{row, col})].imag());

                    dest[i][pos] = a;
                    dest[i + MmaIterations::kRow][pos++] =
                            (kTransform == ComplexTransform::kConjugate ? -b
                                                                        : b);
                }
            }
        }
    }
};

template <typename RealElement, typename DestinationFragment,
          typename SourceFragment, typename MmaIterations,
          typename MmaOperandShape, ComplexTransform Transform_,
          FloatRoundStyle Round_>
struct UnpackComplexConvertAndPackForMma<
        RealElement, DestinationFragment, SourceFragment, MmaIterations,
        MmaOperandShape, Transform_, Operand::kB, Round_> {
    static Operand const kOperand = Operand::kB;
    static ComplexTransform const kTransform = Transform_;
    static FloatRoundStyle const kRound = Round_;

    using MmaElement = typename DestinationFragment::Element;

    using Converter = NumericConverter<MmaElement, RealElement, kRound>;

    using SourceFragmentLayout = layout::RowMajor;
    static int const kLdm = MmaIterations::kColumn * MmaOperandShape::kColumn;

    CUTLASS_DEVICE
    UnpackComplexConvertAndPackForMma() {}

    CUTLASS_HOST_DEVICE
    void operator()(DestinationFragment* dest, SourceFragment const& source) {
        Converter convert_op;
        SourceFragmentLayout layout(kLdm);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < MmaIterations::kColumn; i++) {
            int pos = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < MmaOperandShape::kColumn; c++) {
                CUTLASS_PRAGMA_UNROLL
                for (int r = 0; r < MmaOperandShape::kRow; r++) {
                    int row = r;
                    int col = c + i * MmaOperandShape::kColumn;

                    MmaElement a = convert_op(
                            source[layout(MatrixCoord{row, col})].real());
                    MmaElement b = convert_op(
                            source[layout(MatrixCoord{row, col})].imag());

                    dest[i][pos] = a;
                    dest[i + MmaIterations::kColumn][pos++] =
                            (kTransform == ComplexTransform::kConjugate ? -b
                                                                        : b);
                }
            }
        }
    }
};
}


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
class MmaComplexTensorOp;


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
class MmaComplexTensorOp<Shape_, complex<RealElementA>, LayoutA_,
                         complex<RealElementB>, LayoutB_, complex<RealElementC>,
                         LayoutC_, Policy_, TransformA, TransformB, Enable> {
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

    using ArchTag = typename ArchMmaOperator::ArchTag;

    using OperatorClass = arch::OpClassTensorOp;

    using InstructionShape = typename ArchMmaOperator::Shape;

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

    using IteratorC = MmaTensorOpAccumulatorTileIterator<
            MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
            typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

    using FragmentC = typename IteratorC::Fragment;

    static_assert(FragmentC::kElements ==
                          2 * MmaIterations::kCount *
                                  ArchMmaOperator::FragmentC::kElements,
                  "Unexpected planar complex fragment length.");

private:

    ArchMmaOperator mma;

public:

    CUTLASS_DEVICE
    MmaComplexTensorOp() {}

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
                MmaOperandA operand_A;
                MmaOperandB operand_B;

                operand_A[0] = A[m].real();
                operand_B[0] = B[n].real();

                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow);

                mma(*accum, operand_A, operand_B, *accum);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {
                MmaOperandA operand_A;
                MmaOperandB operand_B;

                operand_A[0] = A[m].real();
                operand_B[0] = (kTransformB == ComplexTransform::kConjugate
                                        ? -B[n].imag()
                                        : B[n].imag());

                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow) +
                                     MmaIterations::kCount;

                mma(*accum, operand_A, operand_B, *accum);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaIterations::kColumn; ++n) {
                MmaOperandA operand_A;
                MmaOperandB operand_B;

                operand_A[0] = (kTransformA == ComplexTransform::kConjugate
                                        ? A[m].imag()
                                        : -A[m].imag());
                operand_B[0] = (kTransformB == ComplexTransform::kConjugate
                                        ? -B[n].imag()
                                        : B[n].imag());

                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow);

                mma(*accum, operand_A, operand_B, *accum);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {
                MmaOperandA operand_A;
                MmaOperandB operand_B;

                operand_A[0] = (kTransformA == ComplexTransform::kConjugate
                                        ? -A[m].imag()
                                        : A[m].imag());
                operand_B[0] = B[n].real();

                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow) +
                                     MmaIterations::kCount;

                mma(*accum, operand_A, operand_B, *accum);
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


template <
        typename Shape_,
        typename LayoutA_,
        typename LayoutB_,
        typename LayoutC_,
        typename Policy_,
        ComplexTransform TransformA,
        ComplexTransform TransformB,
        typename Enable>
class MmaComplexTensorOp<Shape_, complex<float>, LayoutA_, complex<float>,
                         LayoutB_, complex<float>, LayoutC_, Policy_,
                         TransformA, TransformB, Enable> {
public:
    using Shape = Shape_;

    using RealElementA = float;

    using ElementA = complex<RealElementA>;

    using LayoutA = LayoutA_;

    using RealElementB = float;

    using ElementB = complex<RealElementB>;

    using LayoutB = LayoutB_;

    using RealElementC = float;

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

    using TransformedFragmentA =
            Array<typename ArchMmaOperator::ElementA, FragmentA::kElements * 2>;

    using IteratorB = MmaTensorOpMultiplicandTileIterator<
            MatrixShape<Shape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB,
            MatrixShape<ArchMmaOperator::Shape::kK, ArchMmaOperator::Shape::kN>,
            Policy::OpDelta::kColumn, 32, 1>;

    using FragmentB = typename IteratorB::Fragment;

    using TransformedFragmentB =
            Array<typename ArchMmaOperator::ElementB, FragmentB::kElements * 2>;

    static_assert(
            !(Shape::kM % ArchMmaOperator::Shape::kM) &&
                    !(Shape::kN % ArchMmaOperator::Shape::kN),
            "Shape of warp-level Mma must be divisible by operator shape.");

    using MmaIterations = MatrixShape<Shape::kM / ArchMmaOperator::Shape::kM,
                                      Shape::kN / ArchMmaOperator::Shape::kN>;

    using IteratorC = MmaTensorOpAccumulatorTileIterator<
            MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC,
            typename ArchMmaOperator::Shape, typename Policy::OpDelta>;

    using FragmentC = typename IteratorC::Fragment;

private:

    ArchMmaOperator mma;

public:

    CUTLASS_DEVICE
    MmaComplexTensorOp() {}

    CUTLASS_DEVICE
    void operator()(FragmentC& D, TransformedFragmentA const& A,
                    TransformedFragmentB const& B, FragmentC const& C) const {
        using InstMmaOperandA = typename ArchMmaOperator::FragmentA;
        using InstMmaOperandB = typename ArchMmaOperator::FragmentB;
        using MmaOperandC = typename ArchMmaOperator::FragmentC;

        static_assert(platform::is_same<cutlass::gemm::GemmShape<16, 8, 8>,
                                        typename ArchMmaOperator::Shape>::value,
                      "This implementation only supports MMA.1688 math "
                      "instructions.");

        static_assert(InstMmaOperandA::kElements == 4,
                      "This implementation only supports math instructions in "
                      "which exactly four element is needed for the A operand."
                      "We can geneneralize later.");

        static_assert(InstMmaOperandB::kElements == 2,
                      "This implementation only supports math instructions in "
                      "which exactly two element is needed for the B operand."
                      "We can geneneralize later.");

        InstMmaOperandA const* operand_A =
                reinterpret_cast<InstMmaOperandA const*>(&A);
        InstMmaOperandB const* operand_B =
                reinterpret_cast<InstMmaOperandB const*>(&B);

        D = C;

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MmaIterations::kRow; ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaIterations::kColumn; ++n) {
                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow);

                mma(*accum, operand_A[m], operand_B[n], *accum);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {
                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow) +
                                     MmaIterations::kCount;

                mma(*accum, operand_A[m], operand_B[n + MmaIterations::kColumn],
                    *accum);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaIterations::kColumn; ++n) {
                negate<InstMmaOperandB> negate_op;

                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow);

                mma(*accum, operand_A[m + MmaIterations::kRow],
                    negate_op(operand_B[n + MmaIterations::kColumn]), *accum);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int n = MmaIterations::kColumn - 1; n >= 0; --n) {
                MmaOperandC* accum = reinterpret_cast<MmaOperandC*>(&D) +
                                     (m + n * MmaIterations::kRow) +
                                     MmaIterations::kCount;

                mma(*accum, operand_A[m + MmaIterations::kRow], operand_B[n],
                    *accum);
            }
        }
    }

    CUTLASS_DEVICE
    void transform(TransformedFragmentA& dst_A, TransformedFragmentB& dst_B,
                   FragmentA const& A, FragmentB const& B) const {
        using InstMmaOperandA = typename ArchMmaOperator::FragmentA;
        using InstMmaOperandB = typename ArchMmaOperator::FragmentB;


        FloatRoundStyle const kRoundA =
                FloatRoundStyle::round_half_ulp_trunc_dntz;
        FloatRoundStyle const kRoundB =
                FloatRoundStyle::round_half_ulp_trunc_dntz;

        detail::UnpackComplexConvertAndPackForMma<
                RealElementA, InstMmaOperandA, FragmentA, MmaIterations,
                MatrixShape<2, 2>, kTransformA, Operand::kA, kRoundA>
                convert_A;

        detail::UnpackComplexConvertAndPackForMma<
                RealElementB, InstMmaOperandB, FragmentB, MmaIterations,
                MatrixShape<2, 1>, kTransformB, Operand::kB, kRoundB>
                convert_B;

        convert_A(reinterpret_cast<InstMmaOperandA*>(&dst_A), A);
        convert_B(reinterpret_cast<InstMmaOperandB*>(&dst_B), B);
    }
};




}
}
}

