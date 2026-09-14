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
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_conversion.h"
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
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sparse.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename Shape_,
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_,
        typename Policy_,
        int PartitionsK_ = 1,
        bool AccumulatorsInRowMajor = false,
        typename Enable = bool>
class SparseMmaTensorOp {
public:
    using Shape = Shape_;

    using ElementA = ElementA_;

    using LayoutA = LayoutA_;

    using ElementB = ElementB_;

    using LayoutB = LayoutB_;

    using ElementC = ElementC_;

    using LayoutC = LayoutC_;

    using Policy = Policy_;

    using Base = MmaTensorOp<Shape, ElementA, LayoutA, ElementB, LayoutB,
                             ElementC, LayoutC, Policy, PartitionsK_,
                             AccumulatorsInRowMajor, Enable>;

    using ArchMmaOperator = typename Base::ArchMmaOperator;

    using ArchTag = typename Base::ArchTag;

    using OperatorClass = typename Base::OperatorClass;

    using InstructionShape = typename Base::InstructionShape;

    static ComplexTransform const kTransformA = Base::kTransformA;

    static ComplexTransform const kTransformB = Base::kTransformB;

    static int const kThreadCount = 32;

    static int const kPartitionsK = PartitionsK_;

    static int const kSparse = Policy::Operator::kSparse;

    static int const kMetaSizeInBits = Policy::Operator::kMetaSizeInBits;

    static int const kMaxID2 = Policy::Operator::kMaxID2;

    using ElementE =
            typename cutlass::platform::conditional<kMaxID2 == 1, uint32_t,
                                                    uint16_t>::type;

    static int const kElementsPerElementE =
            128 / cutlass::sizeof_bits<ElementA>::value;

    static int const kInterleaved = 2;

    using LayoutE = cutlass::layout::ColumnMajor;

public:
    using IteratorA = MmaTensorOpMultiplicandTileIterator<
            MatrixShape<Shape::kM, Shape::kK / kSparse>, Operand::kA, ElementA,
            LayoutA,
            MatrixShape<Policy::Operator::Shape::kM,
                        Policy::Operator::Shape::kK / kSparse>,
            Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

    using FragmentA = typename IteratorA::Fragment;

    using TransformedFragmentA =
            Array<typename Policy::Operator::ElementA, FragmentA::kElements>;

    using IteratorB = typename Base::IteratorB;

    using FragmentB = typename Base::FragmentB;

    using TransformedFragmentB = typename Base::TransformedFragmentB;

    using IteratorC = typename Base::IteratorC;

    using FragmentC = typename Base::FragmentC;

    using IteratorE = SparseMmaTensorOpMetaTileIterator<
            MatrixShape<Shape::kM * kInterleaved, Shape::kK / kSparse /
                                                          kElementsPerElementE /
                                                          kInterleaved>,
            ElementE, LayoutE,
            MatrixShape<Policy::Operator::Shape::kM,
                        Policy::Operator::Shape::kK / kSparse /
                                kElementsPerElementE / kInterleaved>,
            Policy::OpDelta::kRow, kThreadCount, kPartitionsK>;

    using FragmentE = typename IteratorE::Fragment;

    using MmaIterations = typename Base::MmaIterations;

public:
    ArchMmaOperator mma;

public:

    CUTLASS_DEVICE
    SparseMmaTensorOp() {}

    CUTLASS_DEVICE
    void operator()(FragmentC& D, TransformedFragmentA const& A,
                    TransformedFragmentB const& B, FragmentC const& C,
                    FragmentE const& E) const {
        using MmaOperandA = typename Policy::Operator::FragmentA;
        using MmaOperandB = typename Policy::Operator::FragmentB;
        using MmaOperandC = typename Policy::Operator::FragmentC;
        using MmaOperandE = typename Policy::Operator::FragmentE;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

        D = C;

        MmaOperandA const* ptr_A = reinterpret_cast<MmaOperandA const*>(&A);
        MmaOperandB const* ptr_B = reinterpret_cast<MmaOperandB const*>(&B);
        MmaOperandC* ptr_D = reinterpret_cast<MmaOperandC*>(&D);
        MmaOperandE const* ptr_E = reinterpret_cast<MmaOperandE const*>(&E);

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MmaIterations::kRow; ++m) {
            int id2 = m % kMaxID2;

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaIterations::kColumn; ++n) {
                int n_serpentine =
                        ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

                if (AccumulatorsInRowMajor) {
                    mma(ptr_D[n_serpentine + m * MmaIterations::kColumn],
                        ptr_A[m], ptr_B[n_serpentine],
                        ptr_D[n_serpentine + m * MmaIterations::kColumn],
                        ptr_E[(m / kMaxID2)], id2);
                } else {
                    mma(ptr_D[m + n_serpentine * MmaIterations::kRow], ptr_A[m],
                        ptr_B[n_serpentine],
                        ptr_D[m + n_serpentine * MmaIterations::kRow],
                        ptr_E[(m / kMaxID2)], id2);
                }
            }
        }
#else
        assert(0);
#endif
    }

    CUTLASS_DEVICE
    void transform(TransformedFragmentA& dst_A, TransformedFragmentB& dst_B,
                   FragmentA const& A, FragmentB const& B) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        FloatRoundStyle const kRoundA =
                PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                                      ElementA>::kRound;
        FloatRoundStyle const kRoundB =
                PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                                      ElementB>::kRound;
        detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                               FragmentA::kElements / 2, kRoundA>
                convert_A;
        NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                              FragmentB::kElements, kRoundB>
                convert_B;
        Array<ElementA, FragmentA::kElements / 2> const* ptr_A =
                reinterpret_cast<
                        Array<ElementA, FragmentA::kElements / 2> const*>(&A);
        Array<typename ArchMmaOperator::ElementA,
              FragmentA::kElements / 2>* ptr_dst_A =
                reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                       FragmentA::kElements / 2>*>(&dst_A);

        dst_B = convert_B(B);

        ptr_dst_A[0] = convert_A(ptr_A[0]);
        ptr_dst_A[1] = convert_A(ptr_A[1]);
#else
        assert(0);
#endif
    }
};


}
}
}

