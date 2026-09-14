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

namespace cutlass {
namespace layout {

class PackedVectorLayout {
public:
    static int const kRank = 1;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Coord<kRank, Index>;

    using Stride = Coord<kStrideRank, Index>;

private:

public:

    CUTLASS_HOST_DEVICE
    PackedVectorLayout() {}

    CUTLASS_HOST_DEVICE
    static PackedVectorLayout packed(TensorCoord const& size) {
        return PackedVectorLayout();
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const { return coord[0]; }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return make_Coord(1); }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& size) const { return size[0]; }
};

}
}
