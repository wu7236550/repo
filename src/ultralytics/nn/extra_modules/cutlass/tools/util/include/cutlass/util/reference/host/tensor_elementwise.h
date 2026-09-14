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
#include "cutlass/functional.h"

#include "tensor_foreach.h"

namespace cutlass {
namespace reference {
namespace host {


namespace detail {


template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementD, typename LayoutD,
          typename BinaryFunc>
struct TensorFuncBinaryOp {

    TensorView<ElementD, LayoutD> view_d;
    TensorRef<ElementA, LayoutA> ref_a;
    TensorRef<ElementB, LayoutB> ref_b;
    BinaryFunc func;


    TensorFuncBinaryOp() {}

    TensorFuncBinaryOp(TensorView<ElementD, LayoutD> const& view_d_,
                       TensorRef<ElementA, LayoutA> const& ref_a_,
                       TensorRef<ElementB, LayoutB> const& ref_b_,
                       BinaryFunc func = BinaryFunc())
            : view_d(view_d_), view_a(view_a_), view_b(view_b_), func(func) {}

    void operator()(Coord<LayoutD::kRank> const& coord) const {
        view_d.at(coord) =
                func(ElementD(view_a.at(coord)), ElementD(view_b.at(coord)));
    }
};

}


template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA, typename ElementB, typename LayoutB>
void TensorAdd(TensorView<ElementD, LayoutD> d,
               TensorRef<ElementA, LayoutA> a,
               TensorRef<ElementB, LayoutB> b
) {
    detail::TensorFuncBinaryOp<ElementD, LayoutD, ElementA, LayoutA, ElementB,
                               LayoutB, cutlass::plus<ElementD> >
            func(d, a, b);

    TensorForEach(d.extent(), func);
}

template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA>
void TensorAdd(TensorView<ElementD, LayoutD> d,
               TensorRef<ElementA, LayoutA> a
) {
    TensorAdd(d, d, a);
}


template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA, typename ElementB, typename LayoutB>
void TensorSub(TensorView<ElementD, LayoutD> d,
               TensorRef<ElementA, LayoutA> a,
               TensorRef<ElementB, LayoutB> b
) {
    detail::TensorFuncBinaryOp<ElementD, LayoutD, ElementA, LayoutA, ElementB,
                               LayoutB, cutlass::minus<ElementD> >
            func(d, a, b);

    TensorForEach(d.extent(), func);
}

template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA, typename ElementB, typename LayoutB>
void TensorSub(TensorView<ElementD, LayoutD> d,
               TensorRef<ElementA, LayoutA> a
) {
    TensorSub(d, d, a);
}


template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA, typename ElementB, typename LayoutB>
void TensorMul(TensorView<ElementD, LayoutD> d,
               TensorRef<ElementA, LayoutA> a,
               TensorRef<ElementB, LayoutB> b
) {
    detail::TensorFuncBinaryOp<ElementD, LayoutD, ElementA, LayoutA, ElementB,
                               LayoutB, cutlass::multiplies<ElementD> >
            func(d, a, b);

    TensorForEach(d.extent(), func);
}

template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA>
void TensorMul(TensorView<ElementD, LayoutD> d,
               TensorRef<ElementA, LayoutA> a
) {
    TensorMul(d, d, a);
}


template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA, typename ElementB, typename LayoutB>
void TensorDiv(TensorView<ElementD, LayoutD> d,
               TensorRef<ElementA, LayoutA> a,
               TensorRef<ElementB, LayoutB> b
) {
    detail::TensorFuncBinaryOp<ElementD, LayoutD, ElementA, LayoutA, ElementB,
                               LayoutB, cutlass::divides<ElementD> >
            func(d, a, b);

    TensorForEach(d.extent(), func);
}

template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA>
void TensorDiv(TensorView<ElementD, LayoutD> d,
               TensorRef<ElementA, LayoutA> a
) {
    TensorMul(d, d, a);
}


template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA, typename ElementB, typename LayoutB>
void TensorModulus(
        TensorView<ElementD, LayoutD> d,
        TensorRef<ElementA, LayoutA> a,
        TensorRef<ElementB, LayoutB> b
) {
    detail::TensorFuncBinaryOp<ElementD, LayoutD, ElementA, LayoutA, ElementB,
                               LayoutB, cutlass::modulus<ElementD> >
            func(d, a, b);

    TensorForEach(d.extent(), func);
}

template <typename ElementD, typename LayoutD, typename ElementA,
          typename LayoutA>
void TensorModulus(
        TensorView<ElementD, LayoutD> d,
        TensorRef<ElementA, LayoutA> a
) {
    TensorMul(d, d, a);
}


}
}
}
