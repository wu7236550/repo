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
#include "cutlass/tensor_view.h"

#include "cutlass/util/reference/device/tensor_foreach.h"


namespace cutlass {
namespace reference {
namespace device {


namespace detail {

template <typename Element,
          typename Layout>
struct TensorReLuFunc {
    using TensorView = TensorView<Element, Layout>;

    using TensorCoord = typename TensorView::TensorCoord;

    struct Params {

        TensorView view;
        Element threshold;


        Params(TensorView view_ = TensorView(), Element threshold_ = Element(0))
                : view(view_), threshold(threshold_) {}
    };


    Params params;


    CUTLASS_DEVICE
    TensorReLuFunc(Params const& params) : params(params) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        Element const& value = params.view.at(coord);
        params.view.at(coord) =
                (value < params.threshold) ? params.threshold : value;
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorReLu(
  TensorView<Element, Layout> view,
  Element threshold = Element(0)) {

    using Func = detail::TensorReLuFunc<Element, Layout>;
    using Params = typename Func::Params;

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, threshold));
}


}
}
}
