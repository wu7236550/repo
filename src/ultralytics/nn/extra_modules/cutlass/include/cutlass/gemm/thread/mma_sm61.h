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
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/thread/mma.h"


namespace cutlass {
namespace gemm {
namespace thread {


template <
        typename Shape_,
        typename LayoutC_>
struct Mma<Shape_, int8_t, layout::RowMajor, int8_t, layout::ColumnMajor,
           int32_t, LayoutC_, arch::OpMultiplyAdd, bool> {
    using Shape = Shape_;

    using ElementA = int8_t;

    using LayoutA = layout::RowMajor;

    using ElementB = int8_t;

    using LayoutB = layout::ColumnMajor;

    using ElementC = int32_t;

    using LayoutC = LayoutC_;

    using Operator = arch::OpMultiplyAdd;

    using FragmentA = Array<ElementA, Shape::kMK>;

    using FragmentB = Array<ElementB, Shape::kKN>;

    using FragmentC = Array<ElementC, Shape::kMN>;

    using ArchMmaOperator =
            arch::Mma<gemm::GemmShape<1, 1, 4>, 1, ElementA, LayoutA, ElementB,
                      LayoutB, ElementC, LayoutC, arch::OpMultiplyAdd>;


    CUTLASS_HOST_DEVICE
    void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B,
                    FragmentC const& C) {
        TensorRef<ElementC, LayoutC> d(reinterpret_cast<ElementC*>(&D),
                                       LayoutC::packed({Shape::kM, Shape::kN}));

        D = C;

        ArchMmaOperator mma;

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Shape::kK / ArchMmaOperator::Shape::kK; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Shape::kN; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int m = 0; m < Shape::kM; ++m) {
                    MatrixCoord mn(m, n);

                    Array<int8_t, 4> const* ptr_A =
                            reinterpret_cast<Array<int8_t, 4> const*>(&A);
                    Array<int8_t, 4> const* ptr_B =
                            reinterpret_cast<Array<int8_t, 4> const*>(&B);

                    Array<int32_t, 1> tmp =
                            reinterpret_cast<Array<int32_t, 1>&>(d.at(mn));

                    mma(tmp,
                        ptr_A[m * Shape::kK / ArchMmaOperator::Shape::kK + k],
                        ptr_B[n * Shape::kK / ArchMmaOperator::Shape::kK + k],
                        tmp);

                    d.at(mn) = reinterpret_cast<int32_t&>(tmp);
                }
            }
        }
    }
};

template <
        typename Shape_,
        typename LayoutC_>
struct Mma<Shape_, int8_t, layout::ColumnMajor, int8_t, layout::RowMajor,
           int32_t, LayoutC_, arch::OpMultiplyAdd, int8_t> {
    using Shape = Shape_;

    using ElementA = int8_t;

    using LayoutA = layout::ColumnMajor;

    using ElementB = int8_t;

    using LayoutB = layout::RowMajor;

    using ElementC = int32_t;

    using LayoutC = LayoutC_;

    using Operator = arch::OpMultiplyAdd;

    using FragmentA = Array<ElementA, Shape::kMK>;

    using FragmentB = Array<ElementB, Shape::kKN>;

    using FragmentC = Array<ElementC, Shape::kMN>;

    using ArchMmaOperator =
            arch::Mma<gemm::GemmShape<1, 1, 4>, 1, ElementA, LayoutA, ElementB,
                      LayoutB, ElementC, LayoutC, arch::OpMultiplyAdd>;


    CUTLASS_HOST_DEVICE
    void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B,
                    FragmentC const& C) {
        TensorRef<ElementC, LayoutC> d(reinterpret_cast<ElementC*>(&D),
                                       LayoutC::packed({Shape::kM, Shape::kN}));

        D = C;

        ArchMmaOperator mma;

        Array<int8_t, 4> const* ptr_A =
                reinterpret_cast<Array<int8_t, 4> const*>(&A);
        Array<int8_t, 4> const* ptr_B =
                reinterpret_cast<Array<int8_t, 4> const*>(&B);

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Shape::kK / ArchMmaOperator::Shape::kK; ++k) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < Shape::kN; ++n) {
                CUTLASS_PRAGMA_UNROLL
                for (int m = 0; m < Shape::kM; ++m) {
                    MatrixCoord mn(m, n);

                    Array<int32_t, 1> tmp =
                            reinterpret_cast<Array<int32_t, 1>&>(d.at(mn));

                    mma(tmp, ptr_A[m + k * Shape::kM], ptr_B[n + k * Shape::kN],
                        tmp);

                    d.at(mn) = reinterpret_cast<int32_t&>(tmp);
                }
            }
        }
    }
};

}
}
}

