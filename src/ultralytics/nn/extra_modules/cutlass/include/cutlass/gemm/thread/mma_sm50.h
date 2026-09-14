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
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/thread/mma.h"


namespace cutlass {
namespace gemm {
namespace thread {


template <
        typename Shape_,
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_,
        typename Operator_>
struct MmaGeneric {
    using Shape = Shape_;

    using ElementA = ElementA_;

    using LayoutA = LayoutA_;

    using ElementB = ElementB_;

    using LayoutB = LayoutB_;

    using ElementC = ElementC_;

    using LayoutC = LayoutC_;

    using Operator = Operator_;

    using FragmentA = Array<ElementA, Shape::kMK>;

    using FragmentB = Array<ElementB, Shape::kKN>;

    using FragmentC = Array<ElementC, Shape::kMN>;

    using MmaOp = arch::Mma<gemm::GemmShape<1, 1, 1>, 1, ElementA, LayoutA,
                            ElementB, LayoutB, ElementC, LayoutC, Operator>;


    CUTLASS_HOST_DEVICE
    void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B,
                    FragmentC const& C) {
        TensorRef<ElementA const, LayoutA> a_ref(
                reinterpret_cast<ElementA const*>(&A),
                LayoutA::packed({Shape::kM, Shape::kK}));

        TensorRef<ElementB const, LayoutB> b_ref(
                reinterpret_cast<ElementB const*>(&B),
                LayoutB::packed({Shape::kK, Shape::kN}));

        TensorRef<ElementC, LayoutC> d_ref(
                reinterpret_cast<ElementC*>(&D),
                LayoutC::packed({Shape::kM, Shape::kN}));

        MmaOp mma_op;

        D = C;

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Shape::kK; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Shape::kN; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int m = 0; m < Shape::kM; ++m) {
                    int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;

                    MatrixCoord mn(m_serpentine, n);
                    MatrixCoord mk(m_serpentine, k);
                    MatrixCoord kn(k, n);

                    Array<ElementC, 1> d;
                    Array<ElementA, 1> a;
                    Array<ElementB, 1> b;

                    d[0] = d_ref.at(mn);
                    a[0] = a_ref.at(mk);
                    b[0] = b_ref.at(kn);

                    mma_op(d, a, b, d);

                    d_ref.at(mn) = d[0];
                }
            }
        }
    }
};


template <
        typename Shape_,
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_>
struct Mma<Shape_, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
           LayoutC_, arch::OpMultiplyAdd, bool> {
    using Shape = Shape_;

    using ElementA = ElementA_;

    using LayoutA = LayoutA_;

    using ElementB = ElementB_;

    using LayoutB = LayoutB_;

    using ElementC = ElementC_;

    using LayoutC = LayoutC_;

    using Operator = arch::OpMultiplyAdd;

    using FragmentA = Array<ElementA, Shape::kMK>;

    using FragmentB = Array<ElementB, Shape::kKN>;

    using FragmentC = Array<ElementC, Shape::kMN>;

    using ArchMmaOperator =
            typename MmaGeneric<Shape, ElementA, LayoutA, ElementB, LayoutB,
                                ElementC, LayoutC, Operator>::MmaOp;

    CUTLASS_HOST_DEVICE
    void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B,
                    FragmentC const& C) {
        MmaGeneric<Shape, ElementA, LayoutA, ElementB, LayoutB, ElementC,
                   LayoutC, Operator>
                mma;

        mma(D, A, B, C);
    }
};


}
}
}

