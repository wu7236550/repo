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
/*! \file
    \brief Defines a structure containing strides and a pointer to tensor data.

    TensorView is derived from TensorRef and contributes bounds to the tensor's
   index space. Thus, it is a complete mathematical object and may be used in
   tensor algorithms. It is decoupled from data storage and is therefore
   lightweight and may be embedded in larger tensor objects or memory
   structures.

    See cutlass/tensor_ref.h for more details about the mapping of the logical
   tensor index space to linear memory.
*/

#pragma once

#if !defined(__CUDACC_RTC__)
#include <cmath>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref_planar_complex.h"

namespace cutlass {


template <
        typename Element_,
        typename Layout_>
class TensorViewPlanarComplex
        : public TensorRefPlanarComplex<Element_, Layout_> {
public:
    using Base = cutlass::TensorRefPlanarComplex<Element_, Layout_>;

    using Layout = Layout_;

    using ConstTensorRef = typename Base::ConstTensorRef;

    using TensorRef = Base;

    using Element = Element_;

    using Reference = Element&;

    static int const kRank = Layout::kRank;

    using Index = typename Layout::Index;

    using LongIndex = typename Layout::LongIndex;

    using TensorCoord = typename Layout::TensorCoord;

    using Stride = typename Layout::Stride;

    using ConstTensorView = TensorViewPlanarComplex<
            typename platform::remove_const<Element>::type const, Layout>;

    using NonConstTensorView = TensorViewPlanarComplex<
            typename platform::remove_const<Element>::type, Layout>;

    static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

private:
    TensorCoord extent_;

public:

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex(TensorCoord const& extent = TensorCoord())
            : extent_(extent) {}

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex(
            Element* ptr,
            Layout const& layout,
            LongIndex imaginary_stride,
            TensorCoord const&
                    extent
            )
            : Base(ptr, layout, imaginary_stride), extent_(extent) {}

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex(
            TensorRef const&
                    ref,
            TensorCoord const& extent
            )
            : Base(ref), extent_(extent) {}

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex(
            NonConstTensorView const& view
            )
            : Base(view), extent_(view.extent_) {}

    CUTLASS_HOST_DEVICE
    void reset(Element* ptr, Layout const& layout, LongIndex imaginary_stride,
               TensorCoord size) {
        Base::reset(ptr, layout, imaginary_stride);
        this->resize(extent_);
    }

    CUTLASS_HOST_DEVICE
    void resize(TensorCoord extent) { this->extent_ = extent; }

    CUTLASS_HOST_DEVICE
    TensorCoord const& extent() const { return extent_; }

    CUTLASS_HOST_DEVICE
    Index extent(int dim) const { return extent_.at(dim); }

    CUTLASS_HOST_DEVICE
    bool contains(TensorCoord const& coord) const {
        CUTLASS_PRAGMA_UNROLL
        for (int dim = 0; dim < kRank; ++dim) {
            if (!(coord[dim] >= 0 && coord[dim] < extent(dim))) {
                return false;
            }
        }
        return true;
    }

    CUTLASS_HOST_DEVICE
    Base ref() const {
        return Base(this->data(), this->layout(), this->imaginary_stride());
    }

    CUTLASS_HOST_DEVICE
    ConstTensorRef const_ref() const {
        return ConstTensorRef(this->data(), this->layout());
    }

    CUTLASS_HOST_DEVICE
    ConstTensorView const_view() const {
        return ConstTensorView(const_ref(), extent_);
    }

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex subview(
            TensorCoord extent,
            TensorCoord const& location =
                    TensorCoord()
            ) const {
        return TensorViewPlanarComplex(ref(), extent.clamp(extent_ - location))
                .add_coord_offset(location);
    }

    CUTLASS_HOST_DEVICE
    size_t capacity() const { return Base::layout().capacity(extent_); }

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex operator+(
            TensorCoord const&
                    b
            ) const {
        TensorViewPlanarComplex result(*this);
        result.add_pointer_offset(this->offset(b));
        return result;
    }

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex& operator+=(
            TensorCoord const&
                    b
    ) {
        this->add_pointer_offset(this->offset(b));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex operator-(
            TensorCoord const&
                    b
            ) const {
        TensorRef result(*this);
        result.add_pointer_offset(-this->offset(b));
        return result;
    }

    CUTLASS_HOST_DEVICE
    TensorViewPlanarComplex& operator-=(
            TensorCoord const&
                    b
    ) {
        this->add_pointer_offset(-this->offset(b));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    cutlass::TensorView<Element, Layout> view_real() const {
        return cutlass::TensorView<Element, Layout>(this->data(),
                                                    this->layout(), extent_);
    }

    CUTLASS_HOST_DEVICE
    cutlass::TensorView<Element, Layout> view_imag() const {
        return cutlass::TensorView<Element, Layout>(this->imaginary_data(),
                                                    this->layout(), extent_);
    }
};


template <typename Element, typename Layout>
CUTLASS_HOST_DEVICE TensorViewPlanarComplex<Element, Layout>
make_TensorViewPlanarComplex(Element* ptr, Layout const& layout,
                             typename Layout::LongIndex imaginary_stride,
                             typename Layout::TensorCoord const& extent) {
    return TensorViewPlanarComplex<Element, Layout>(ptr, layout,
                                                    imaginary_stride, extent);
}


}
