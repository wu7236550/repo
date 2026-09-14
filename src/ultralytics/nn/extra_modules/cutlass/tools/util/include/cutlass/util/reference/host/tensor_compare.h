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

#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_view.h"
#include "cutlass/tensor_view_planar_complex.h"

#include "cutlass/util/distribution.h"
#include "tensor_foreach.h"

namespace cutlass {
namespace reference {
namespace host {


namespace detail {

template <typename Element,
          typename Layout>
struct TensorEqualsFunc {

    TensorView<Element, Layout> lhs;
    TensorView<Element, Layout> rhs;
    bool result;

    TensorEqualsFunc() : result(true) {}

    TensorEqualsFunc(TensorView<Element, Layout> const& lhs_,
                     TensorView<Element, Layout> const& rhs_,
                     double )
            : lhs(lhs_), rhs(rhs_), result(true) {}

    void operator()(Coord<Layout::kRank> const& coord) {
        Element lhs_ = lhs.at(coord);
        Element rhs_ = rhs.at(coord);

        if (lhs_ != rhs_) {
            result = false;
        }
    }

    operator bool() const { return result; }
};

template <typename Layout>
struct TensorEqualsFunc<float, Layout> {
    using Element = float;

    TensorView<Element, Layout> lhs;
    TensorView<Element, Layout> rhs;
    double episilon;
    bool result;

    TensorEqualsFunc() : episilon(1e-4), result(true) {}

    TensorEqualsFunc(double episilon_) : episilon(episilon_), result(true) {}

    TensorEqualsFunc(TensorView<Element, Layout> const& lhs_,
                     TensorView<Element, Layout> const& rhs_,
                     double episilon_ = 1e-4)
            : lhs(lhs_), rhs(rhs_), episilon(episilon_), result(true) {}

    void operator()(Coord<Layout::kRank> const& coord) {
        Element lhs_ = lhs.at(coord);
        Element rhs_ = rhs.at(coord);

        auto numerator = lhs_ - rhs_;
        auto denominator =
                std::max(std::max(std::abs(lhs_), std::abs(rhs_)), 1.f);
        if (std::abs(numerator / denominator) >= episilon) {
            result = false;
        }
    }

    operator bool() const { return result; }
};

}


template <typename Element,
          typename Layout>
bool TensorEquals(TensorView<Element, Layout> const& lhs,
                  TensorView<Element, Layout> const& rhs,
                  double episilon = 1e-4) {
    if (lhs.extent() != rhs.extent()) {
        return false;
    }

    detail::TensorEqualsFunc<Element, Layout> func(lhs, rhs, episilon);
    TensorForEach(lhs.extent(), func);

    return bool(func);
}

template <typename Element,
          typename Layout>
bool TensorEquals(TensorViewPlanarComplex<Element, Layout> const& lhs,
                  TensorViewPlanarComplex<Element, Layout> const& rhs) {
    if (lhs.extent() != rhs.extent()) {
        return false;
    }

    detail::TensorEqualsFunc<Element, Layout> real_func(
            {lhs.data(), lhs.layout(), lhs.extent()},
            {rhs.data(), rhs.layout(), rhs.extent()});

    TensorForEach(lhs.extent(), real_func);

    if (!bool(real_func)) {
        return false;
    }

    detail::TensorEqualsFunc<Element, Layout> imag_func(
            {lhs.data() + lhs.imaginary_stride(), lhs.layout(), lhs.extent()},
            {rhs.data() + rhs.imaginary_stride(), rhs.layout(), rhs.extent()});

    TensorForEach(lhs.extent(), imag_func);

    return bool(imag_func);
}


template <typename Element,
          typename Layout>
bool TensorNotEquals(TensorView<Element, Layout> const& lhs,
                     TensorView<Element, Layout> const& rhs) {
    if (lhs.extent() != rhs.extent()) {
        return true;
    }

    detail::TensorEqualsFunc<Element, Layout> func(lhs, rhs);
    TensorForEach(lhs.extent(), func);

    return !bool(func);
}

template <typename Element,
          typename Layout>
bool TensorNotEquals(TensorViewPlanarComplex<Element, Layout> const& lhs,
                     TensorViewPlanarComplex<Element, Layout> const& rhs) {
    return !TensorEquals(lhs, rhs);
}


namespace detail {

template <typename Element,
          typename Layout>
struct TensorContainsFunc {

    TensorView<Element, Layout> view;
    Element value;
    bool contains;
    Coord<Layout::kRank> location;


    TensorContainsFunc() : contains(false) {}

    TensorContainsFunc(TensorView<Element, Layout> const& view_, Element value_)
            : view(view_), value(value_), contains(false) {}

    void operator()(Coord<Layout::kRank> const& coord) {
        if (view.at(coord) == value) {
            if (!contains) {
                location = coord;
            }
            contains = true;
        }
    }

    operator bool() const { return contains; }
};

}


template <typename Element,
          typename Layout>
bool TensorContains(TensorView<Element, Layout> const& view, Element value) {
    detail::TensorContainsFunc<Element, Layout> func(view, value);

    TensorForEach(view.extent(), func);

    return bool(func);
}


template <typename Element,
          typename Layout>
std::pair<bool, Coord<Layout::kRank> > TensorFind(
        TensorView<Element, Layout> const& view, Element value) {
    detail::TensorContainsFunc<Element, Layout> func(view, value);

    TensorForEach(view.extent(), func);

    return std::make_pair(bool(func), func.location);
}


}
}
}
