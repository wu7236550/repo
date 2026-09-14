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

#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/tensor_ref.h"


namespace cutlass {


template <typename Element_>
struct PlanarComplexReference {

    using Element = Element_;
    using ComplexElement = complex<Element>;


    Element* real;
    Element* imag;


    CUTLASS_HOST_DEVICE
    PlanarComplexReference(Element* real_ = nullptr, Element* imag_ = nullptr)
            : real(real_), imag(imag_) {}

    CUTLASS_HOST_DEVICE
    operator complex<Element>() const { return complex<Element>{*real, *imag}; }

    CUTLASS_HOST_DEVICE
    PlanarComplexReference& operator=(complex<Element> const& rhs) {
        *real = rhs.real();
        *imag = rhs.imag();
        return *this;
    }
};


template <
        typename Element_,
        typename Layout_>
class TensorRefPlanarComplex {
public:
    using Element = Element_;

    using ComplexElement = complex<Element>;

    using Layout = Layout_;

    static_assert(
            sizeof_bits<Element>::value >= 8,
            "Planar complex not suitable for subbyte elements at this time");

    using Reference = PlanarComplexReference<Element>;

    static int const kRank = Layout::kRank;

    using Index = typename Layout::Index;

    using LongIndex = typename Layout::LongIndex;

    using TensorCoord = typename Layout::TensorCoord;

    using Stride = typename Layout::Stride;

    using ConstTensorRef = TensorRefPlanarComplex<
            typename platform::remove_const<Element>::type const, Layout>;

    using NonConstTensorRef = TensorRefPlanarComplex<
            typename platform::remove_const<Element>::type, Layout>;

    static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

private:
    Element* ptr_;

    Layout layout_;

    LongIndex imaginary_stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorRefPlanarComplex(
            Element* ptr = nullptr,
            Layout const& layout = Layout(),
            LongIndex imaginary_stride = 0)
            : ptr_(ptr), layout_(layout), imaginary_stride_(imaginary_stride) {}

    CUTLASS_HOST_DEVICE
    TensorRefPlanarComplex(
            NonConstTensorRef const& ref
            )
            : ptr_(ref.data()),
              layout_(ref.layout()),
              imaginary_stride_(ref.imaginary_stride_) {}

    CUTLASS_HOST_DEVICE
    ConstTensorRef const_ref() const {
        return ConstTensorRef(ptr_, layout_, imaginary_stride_);
    }

    CUTLASS_HOST_DEVICE
    NonConstTensorRef non_const_ref() const {
        return NonConstTensorRef(
                const_cast<typename platform::remove_const<Element>::type*>(
                        ptr_),
                layout_, imaginary_stride_);
    }

    CUTLASS_HOST_DEVICE
    void reset(Element* ptr = nullptr, LongIndex imaginary_stride = 0) {
        ptr_ = ptr;
        imaginary_stride_ = imaginary_stride;
    }

    CUTLASS_HOST_DEVICE
    void reset(Element* ptr, Layout const& layout, LongIndex imaginary_stride) {
        ptr_ = ptr;
        layout_ = layout;
        imaginary_stride_ = imaginary_stride;
    }

    CUTLASS_HOST_DEVICE
    bool good() const { return ptr_ != nullptr; }

    CUTLASS_HOST_DEVICE
    Element* data() const { return ptr_; }

    CUTLASS_HOST_DEVICE
    Element* imaginary_data() const { return ptr_ + imaginary_stride_; }

    CUTLASS_HOST_DEVICE
    Reference data(LongIndex idx) const {
        return Reference(ptr_ + idx, ptr_ + idx + imaginary_stride_);
    }

    CUTLASS_HOST_DEVICE
    Layout& layout() { return layout_; }

    CUTLASS_HOST_DEVICE
    Layout layout() const { return layout_; }

    LongIndex imaginary_stride() const { return imaginary_stride_; }

    LongIndex& imaginary_stride() { return imaginary_stride_; }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return layout_.stride(); }

    CUTLASS_HOST_DEVICE
    Index stride(int dim) const { return layout_.stride().at(dim); }

    CUTLASS_HOST_DEVICE
    Index& stride(int dim) { return layout_.stride().at(dim); }

    CUTLASS_HOST_DEVICE
    LongIndex offset(TensorCoord const& coord) const { return layout_(coord); }

    CUTLASS_HOST_DEVICE
    Reference at(TensorCoord const& coord) const { return data(offset(coord)); }

    CUTLASS_HOST_DEVICE
    Reference operator[](TensorCoord const& coord) const {
        return data(offset(coord));
    }

    CUTLASS_HOST_DEVICE
    TensorRefPlanarComplex& add_pointer_offset(LongIndex offset_) {
        ptr_ += offset_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TensorRefPlanarComplex& add_coord_offset(TensorCoord const& coord) {
        add_pointer_offset(offset(coord));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TensorRefPlanarComplex operator+(TensorCoord const& b) const {
        TensorRefPlanarComplex result(*this);
        result.add_coord_offset(b);
        return result;
    }

    CUTLASS_HOST_DEVICE
    TensorRefPlanarComplex& operator+=(TensorCoord const& b) {
        add_coord_offset(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TensorRefPlanarComplex operator-(TensorCoord const& b) const {
        TensorRefPlanarComplex result(*this);
        result.add_pointer_offset(-offset(b));
        return result;
    }

    CUTLASS_HOST_DEVICE
    TensorRefPlanarComplex& operator-=(TensorCoord const& b) {
        add_pointer_offset(-offset(b));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    cutlass::TensorRef<Element, Layout> ref_real() const {
        return cutlass::TensorRef<Element, Layout>(data(), layout());
    }

    CUTLASS_HOST_DEVICE
    cutlass::TensorRef<Element, Layout> ref_imag() const {
        return cutlass::TensorRef<Element, Layout>(imaginary_data(), layout());
    }
};


template <typename Element, typename Layout>
CUTLASS_HOST_DEVICE TensorRefPlanarComplex<Element, Layout>
make_TensorRefPlanarComplex(Element* ptr, Layout const& layout,
                            int64_t imaginary_stride) {
    return TensorRefPlanarComplex<Element, Layout>(ptr, layout,
                                                   imaginary_stride);
}


}

