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
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/thread/mma.h"

#include "cutlass/gemm/warp/mma_simt_tile_iterator.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"


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
        int PartitionsK = 1,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone,
        typename Enable = bool>
class MmaSimt {
public:
    using Shape = Shape_;

    using ElementA = ElementA_;

    using LayoutA = LayoutA_;

    using ElementB = ElementB_;

    using LayoutB = LayoutB_;

    using ElementC = ElementC_;

    using LayoutC = LayoutC_;

    using Policy = Policy_;

    using OperatorClass = arch::OpClassSimt;

    using ArchTag = arch::Sm50;

    static ComplexTransform const kTransformA = TransformA;

    static ComplexTransform const kTransformB = TransformB;

    using ThreadLayoutA = typename platform::conditional<
            platform::is_same<layout::ColumnMajorInterleaved<4>,
                              LayoutA>::value,
            layout::ColumnMajor,
            typename platform::conditional<
                    platform::is_same<layout::RowMajorInterleaved<4>,
                                      LayoutA>::value,
                    layout::RowMajor, LayoutA>::type>::type;

    using ThreadLayoutB = typename platform::conditional<
            platform::is_same<layout::ColumnMajorInterleaved<4>,
                              LayoutB>::value,
            layout::ColumnMajor,
            typename platform::conditional<
                    platform::is_same<layout::RowMajorInterleaved<4>,
                                      LayoutB>::value,
                    layout::RowMajor, LayoutB>::type>::type;

    static constexpr bool use_dp4a =
            (platform::is_same<layout::ColumnMajorInterleaved<4>,
                               LayoutA>::value ||
             platform::is_same<layout::RowMajorInterleaved<4>,
                               LayoutA>::value) &&
            platform::is_same<ElementA, int8_t>::value &&
            platform::is_same<ElementB, int8_t>::value;

    using dp4a_type =
            typename platform::conditional<use_dp4a, int8_t, bool>::type;

    using ThreadMma =
            thread::Mma<GemmShape<Shape::kM / Policy::WarpShape::kRow,
                                  Shape::kN / Policy::WarpShape::kColumn,
                                  Policy::LaneMmaShape::kK>,
                        ElementA, ThreadLayoutA, ElementB, ThreadLayoutB,
                        ElementC, LayoutC, arch::OpMultiplyAdd, dp4a_type>;

    using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;

    using InstructionShape = GemmShape<1, 1, use_dp4a ? 4 : 1>;

public:
    using IteratorA = MmaSimtTileIterator<
            MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>, Operand::kA,
            ElementA, LayoutA, Policy, PartitionsK, Shape::kK>;

    using FragmentA = typename IteratorA::Fragment;

    using TransformedFragmentA = FragmentA;

    using IteratorB = MmaSimtTileIterator<
            MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>, Operand::kB,
            ElementB, LayoutB, Policy, PartitionsK, Shape::kK>;

    using FragmentB = typename IteratorB::Fragment;

    using TransformedFragmentB = FragmentB;

    using IteratorC =
            MmaSimtTileIterator<MatrixShape<Shape::kM, Shape::kN>, Operand::kC,
                                ElementC, LayoutC, Policy>;

    using FragmentC = typename ThreadMma::FragmentC;

public:

    CUTLASS_DEVICE
    MmaSimt() {}

    CUTLASS_DEVICE
    void operator()(FragmentC& d, FragmentA a, FragmentB b, FragmentC const& c,
                    int group_idx = 0) const {
        ThreadMma mma;

        if (kTransformA == ComplexTransform::kConjugate) {
            conjugate<FragmentA> conj_a;
            a = conj_a(a);
        }

        if (kTransformB == ComplexTransform::kConjugate) {
            conjugate<FragmentB> conj_b;
            b = conj_b(b);
        }

        mma(d, a, b, c);
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
