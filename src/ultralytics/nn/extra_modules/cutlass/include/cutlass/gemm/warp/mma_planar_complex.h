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
#include "cutlass/gemm/gemm.h"

#include "cutlass/array_planar_complex.h"
#include "cutlass/gemm/warp/tile_iterator_planar_complex.h"


namespace cutlass {
namespace gemm {
namespace warp {


template <
        typename Operator_,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone>
class MmaPlanarComplex {
public:
    using Operator = Operator_;

    using Shape = typename Operator::Shape;

    static ComplexTransform const kTransformA = TransformA;

    static ComplexTransform const kTransformB = TransformB;

    using FragmentA = ArrayPlanarComplex<typename Operator::ElementA,
                                         Operator::FragmentA::kElements>;

    using IteratorA = TileIteratorPlanarComplex<typename Operator::IteratorA>;

    using LayoutA = typename Operator::LayoutA;

    using FragmentB = ArrayPlanarComplex<typename Operator::ElementB,
                                         Operator::FragmentB::kElements>;

    using IteratorB = TileIteratorPlanarComplex<typename Operator::IteratorB>;

    using LayoutB = typename Operator::LayoutB;

    using IteratorC = TileIteratorPlanarComplex<typename Operator::IteratorC>;

    using FragmentC = ArrayPlanarComplex<typename Operator::ElementC,
                                         Operator::FragmentC::kElements>;

    using LayoutC = typename Operator::LayoutC;

private:
    using MmaIterations = MatrixShape<
            Operator::Shape::kM / Operator::Policy::Operator::Shape::kM,
            Operator::Shape::kN / Operator::Policy::Operator::Shape::kN>;

public:
    CUTLASS_DEVICE
    MmaPlanarComplex() {}

    CUTLASS_DEVICE
    void operator()(FragmentC& D, FragmentA const& A_in, FragmentB const& B_in,
                    FragmentC const& C) const {
        D.real = C.real;
        D.imag = C.imag;


        negate<typename FragmentA::ArrayReal> neg_A;

        FragmentA frag_A;
        frag_A.real = A_in.real;

        if (kTransformA == ComplexTransform::kConjugate) {
            frag_A.imag = neg_A(frag_A.imag);
        } else {
            frag_A.imag = frag_A.imag;
        }

        FragmentB frag_B;
        frag_B.real = B_in.real;

        if (kTransformB == ComplexTransform::kConjugate) {
            negate<typename FragmentB::ArrayReal> neg;
            frag_B.imag = neg(frag_B.imag);
        } else {
            frag_B.imag = frag_B.imag;
        }


        Operator real_mma;

        real_mma(D.imag, frag_A.imag, frag_B.real, D.imag);

        real_mma(D.real, frag_A.real, frag_B.real, D.real);

        real_mma(D.imag, frag_A.real, frag_B.imag, D.imag);

        frag_A.imag = neg_A(frag_A.imag);
        real_mma(D.real, frag_A.imag, frag_B.imag, D.real);
    }
};


}
}
}

