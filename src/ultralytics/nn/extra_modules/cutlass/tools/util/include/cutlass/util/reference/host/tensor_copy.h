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
#include "tensor_foreach.h"

namespace cutlass {
namespace reference {
namespace host {


namespace detail {

template <typename DstElement, typename SrcElement>
struct TrivialConvert {
    TrivialConvert() {}

    DstElement operator()(SrcElement src) const { return DstElement(src); }
};

template <typename DstElement, typename DstLayout, typename SrcElement,
          typename SrcLayout, typename F>
struct TensorCopyIf {
    using DstTensorView = TensorView<DstElement, DstLayout>;
    using SrcTensorView = TensorView<SrcElement, SrcLayout>;


    DstTensorView dst;
    SrcTensorView src;
    F convert;


    TensorCopyIf() {}

    TensorCopyIf(DstTensorView const& dst_, SrcTensorView const& src_,
                 F const& convert_)
            : dst(dst_), src(src_), convert(convert_) {}

    void operator()(Coord<DstLayout::kRank> const& coord) {
        if (dst.contains(coord) && src.contains(coord)) {
            dst.at(coord) = convert(src.at(coord));
        }
    }
};

}


template <typename DstElement,
          typename DstLayout,
          typename SrcElement,
          typename SrcLayout,
          typename F
          >
void TensorCopy(TensorView<DstElement, DstLayout> dst,
                TensorView<SrcElement, SrcLayout> src, F const& transform) {
    using CopyIf = detail::TensorCopyIf<DstElement, DstLayout, SrcElement,
                                        SrcLayout, F>;

    CopyIf copy_if(dst, src, transform);

    TensorForEach(dst.extent(), copy_if);
}


template <typename DstElement,
          typename DstLayout,
          typename SrcElement,
          typename SrcLayout,
          typename F
          >
void TensorCopy(TensorView<DstElement, DstLayout> dst,
                TensorRef<SrcElement, SrcLayout> src, F const& transform) {
    using CopyIf = detail::TensorCopyIf<DstElement, DstLayout, SrcElement,
                                        SrcLayout, F>;

    TensorView<SrcElement, SrcLayout> src_view(src, dst.extent());

    CopyIf copy_if(dst, src_view, transform);

    TensorForEach(dst.extent(), copy_if);
}

template <typename DstElement,
          typename DstLayout,
          typename SrcElement,
          typename SrcLayout,
          typename F
          >
void TensorCopy(TensorRef<DstElement, DstLayout> dst,
                TensorView<SrcElement, SrcLayout> src, F const& transform) {
    using CopyIf = detail::TensorCopyIf<DstElement, DstLayout, SrcElement,
                                        SrcLayout, F>;

    TensorView<DstElement, DstLayout> dst_view(dst, src.extent());

    CopyIf copy_if(dst_view, src, transform);

    TensorForEach(src.extent(), copy_if);
}


template <typename DstElement,
          typename DstLayout,
          typename SrcElement,
          typename SrcLayout
          >
void TensorCopy(TensorView<DstElement, DstLayout> dst,
                TensorView<SrcElement, SrcLayout> src) {
    detail::TrivialConvert<DstElement, SrcElement> convert;

    TensorCopy(dst, src, convert);
}


template <typename DstElement,
          typename DstLayout,
          typename SrcElement,
          typename SrcLayout,
          typename F
          >
void TensorCopy(TensorView<DstElement, DstLayout> dst,
                TensorRef<SrcElement, SrcLayout> src) {
    detail::TrivialConvert<DstElement, SrcElement> convert;

    TensorCopy(dst, src, convert);
}


template <typename DstElement,
          typename DstLayout,
          typename SrcElement,
          typename SrcLayout
          >
void TensorCopy(TensorRef<DstElement, DstLayout> dst,
                TensorView<SrcElement, SrcLayout> src) {
    detail::TrivialConvert<DstElement, SrcElement> convert;

    TensorCopy(dst, src, convert);
}


}
}
}
