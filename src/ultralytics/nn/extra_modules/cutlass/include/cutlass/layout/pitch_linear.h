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


template <int Contiguous, int Strided>
struct PitchLinearShape {
    static int const kContiguous = Contiguous;
    static int const kStrided = Strided;
    static int const kCount = Contiguous * Strided;
};


struct PitchLinearCoord : public Coord<2, int> {
public:
    using Index = int;

    using Base = Coord<2, Index>;

private:
    static int const kContiguous = 0;

    static int const kStrided = 1;

public:

    CUTLASS_HOST_DEVICE
    PitchLinearCoord() {}

    CUTLASS_HOST_DEVICE
    PitchLinearCoord(Coord<2, Index> const& coord) : Base(coord) {}

    CUTLASS_HOST_DEVICE
    PitchLinearCoord(Index contiguous_, Index strided_)
            : Base(make_Coord(contiguous_, strided_)) {}

    CUTLASS_HOST_DEVICE
    Index const& contiguous() const { return this->at(kContiguous); }

    CUTLASS_HOST_DEVICE
    Index& contiguous() { return this->at(kContiguous); }

    CUTLASS_HOST_DEVICE
    Index const& strided() const { return this->at(kStrided); }

    CUTLASS_HOST_DEVICE
    Index& strided() { return this->at(kStrided); }


    CUTLASS_HOST_DEVICE
    PitchLinearCoord operator+(Base const& b) const {
        return PitchLinearCoord(Base::operator+(b));
    }

    CUTLASS_HOST_DEVICE
    PitchLinearCoord operator-(Base const& b) const {
        return PitchLinearCoord(Base::operator-(b));
    }

    CUTLASS_HOST_DEVICE
    PitchLinearCoord operator-() const {
        return PitchLinearCoord(-at(0), -at(1));
    }

    CUTLASS_HOST_DEVICE
    PitchLinearCoord operator*(Base const& b) const {
        return PitchLinearCoord(Base::operator*(b));
    }

    CUTLASS_HOST_DEVICE
    PitchLinearCoord operator/(Base const& b) const {
        return PitchLinearCoord(Base::operator/(b));
    }

    CUTLASS_HOST_DEVICE
    PitchLinearCoord& operator+=(Base const& b) {
        Base::operator+=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PitchLinearCoord& operator-=(Base const& b) {
        Base::operator-=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PitchLinearCoord& operator*=(Base const& b) {
        Base::operator*=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    PitchLinearCoord& operator/=(Base const& b) {
        Base::operator/=(b);
        return *this;
    }
};


class PitchLinear {
public:
    static int const kRank = 2;

    static int const kStrideRank = 1;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = PitchLinearCoord;

    using Stride = Coord<kStrideRank, Index>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    PitchLinear(Index ldm = 0) : stride_(ldm) {}

    CUTLASS_HOST_DEVICE
    PitchLinear(Stride _stride) : stride_(_stride) {}

    CUTLASS_HOST_DEVICE
    static PitchLinear packed(TensorCoord const& extent) {
        return PitchLinear(extent.contiguous());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return LongIndex(coord.contiguous()) +
               LongIndex(coord.strided()) * LongIndex(stride_[0]);
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex index) const {
        return make_Coord(Index(index % stride_[0]), Index(index / stride_[0]));
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    Index stride(int rank) const { return stride_[rank]; }

    CUTLASS_HOST_DEVICE
    Index& stride(int rank) { return stride_[rank]; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return extent.strided() * stride_[0];
    }
};


}
}
