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
#include "cutlass/coord.h"
#include "cutlass/platform/platform.h"
#include "cutlass/subbyte_reference.h"

namespace cutlass {


template <int Rank>
class IdentityTensorLayout {
public:
    static int const kRank = Rank;

    static int const kStrideRank = Rank;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Coord<kRank, Index>;

    using Stride = Coord<kStrideRank, Index>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    IdentityTensorLayout(Stride const& stride = Stride()) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    LongIndex operator()(Coord<Rank> const& coord) const {
        return coord.dot(stride_);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& size) const {
        int idx = stride_.max_dim_index();
        return stride_[idx] * size[idx];
    }
};


template <
        typename Element_,
        typename Layout_>
class TensorRef {
public:
    using Element = Element_;

    using Layout = Layout_;

    using Reference =
            typename platform::conditional<sizeof_bits<Element>::value >= 8,
                                           Element&,
                                           SubbyteReference<Element> >::type;

    static int const kRank = Layout::kRank;

    using Index = typename Layout::Index;

    using LongIndex = typename Layout::LongIndex;

    using TensorCoord = typename Layout::TensorCoord;

    using Stride = typename Layout::Stride;

    using ConstTensorRef =
            TensorRef<typename platform::remove_const<Element>::type const,
                      Layout>;

    using NonConstTensorRef =
            TensorRef<typename platform::remove_const<Element>::type, Layout>;

    static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

private:
    Element* ptr_;

    Layout layout_;

public:

    CUTLASS_HOST_DEVICE
    TensorRef(Element* ptr = nullptr,
              Layout const& layout = Layout()
              )
            : ptr_(ptr), layout_(layout) {}

    CUTLASS_HOST_DEVICE
    TensorRef(NonConstTensorRef const& ref
              )
            : ptr_(ref.data()), layout_(ref.layout()) {}

    CUTLASS_HOST_DEVICE
    ConstTensorRef const_ref() const { return ConstTensorRef(ptr_, layout_); }

    CUTLASS_HOST_DEVICE
    NonConstTensorRef non_const_ref() const {
        return NonConstTensorRef(
                const_cast<typename platform::remove_const<Element>::type*>(
                        ptr_),
                layout_);
    }

    CUTLASS_HOST_DEVICE
    void reset(Element* ptr = nullptr) { ptr_ = ptr; }

    CUTLASS_HOST_DEVICE
    void reset(Element* ptr, Layout const& layout) {
        ptr_ = ptr;
        layout_ = layout;
    }

    CUTLASS_HOST_DEVICE
    bool good() const { return ptr_ != nullptr; }

    CUTLASS_HOST_DEVICE
    Element* data() const { return ptr_; }

    CUTLASS_HOST_DEVICE
    Reference data(LongIndex idx) const {
        return ReferenceFactory<typename platform::remove_const<Element>::type,
                                (sizeof_bits<Element>::value < 8)>::get(ptr_,
                                                                        idx);
    }

    CUTLASS_HOST_DEVICE
    Layout& layout() { return layout_; }

    CUTLASS_HOST_DEVICE
    Layout layout() const { return layout_; }

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
    TensorRef& add_pointer_offset(LongIndex offset_) {
        ptr_ += offset_;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TensorRef& add_coord_offset(TensorCoord const& coord) {
        add_pointer_offset(offset(coord));
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TensorRef operator+(TensorCoord const& b) const {
        TensorRef result(*this);
        result.add_coord_offset(b);
        return result;
    }

    CUTLASS_HOST_DEVICE
    TensorRef& operator+=(TensorCoord const& b) {
        add_coord_offset(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TensorRef operator-(TensorCoord const& b) const {
        TensorRef result(*this);
        result.add_pointer_offset(-offset(b));
        return result;
    }

    CUTLASS_HOST_DEVICE
    TensorRef& operator-=(TensorCoord const& b) {
        add_pointer_offset(-offset(b));
        return *this;
    }
};

template <typename Element, typename Layout>
CUTLASS_HOST_DEVICE TensorRef<Element, Layout> make_TensorRef(
        Element* ptr, Layout const& layout) {
    return TensorRef<Element, Layout>(ptr, layout);
}


template <typename Element, typename Layout>
CUTLASS_HOST_DEVICE bool TensorRef_aligned(
        TensorRef<Element, Layout> const& ref, int alignment) {
    int const kStrideRank = Layout::kStrideRank;

    if (reinterpret_cast<uintptr_t>(ref.data()) % alignment) {
        return false;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kStrideRank; ++i) {
        if (ref.stride(i) % alignment) {
            return false;
        }
    }

    return true;
}


}
