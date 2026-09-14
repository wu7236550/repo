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


struct Tensor4DCoord : public Coord<4> {
    using Base = Coord<4>;

    using Index = typename Base::Index;

    using LongIndex = typename Base::LongIndex;

    static int const kN = 0;

    static int const kH = 1;

    static int const kW = 2;

    static int const kC = 3;


    CUTLASS_HOST_DEVICE
    Tensor4DCoord() {}

    CUTLASS_HOST_DEVICE
    Tensor4DCoord(Coord<4> const& coord) : Base(coord) {}

    CUTLASS_HOST_DEVICE
    Tensor4DCoord(Index n, Index h, Index w, Index c)
            : Base(make_Coord(n, h, w, c)) {}

    CUTLASS_HOST_DEVICE
    Index const& n() const { return this->at(kN); }

    CUTLASS_HOST_DEVICE
    Index& n() { return this->at(kN); }

    CUTLASS_HOST_DEVICE
    Index const& h() const { return this->at(kH); }

    CUTLASS_HOST_DEVICE
    Index& h() { return this->at(kH); }

    CUTLASS_HOST_DEVICE
    Index const& w() const { return this->at(kW); }

    CUTLASS_HOST_DEVICE
    Index& w() { return this->at(kW); }

    CUTLASS_HOST_DEVICE
    Index const& c() const { return this->at(kC); }

    CUTLASS_HOST_DEVICE
    Index& c() { return this->at(kC); }


    CUTLASS_HOST_DEVICE
    Tensor4DCoord operator+(Base const& b) const {
        return Tensor4DCoord(Base::operator+(b));
    }

    CUTLASS_HOST_DEVICE
    Tensor4DCoord operator-(Base const& b) const {
        return Tensor4DCoord(Base::operator-(b));
    }

    CUTLASS_HOST_DEVICE
    Tensor4DCoord operator*(Base const& b) const {
        return Tensor4DCoord(Base::operator*(b));
    }

    CUTLASS_HOST_DEVICE
    Tensor4DCoord operator/(Base const& b) const {
        return Tensor4DCoord(Base::operator/(b));
    }

    CUTLASS_HOST_DEVICE
    Tensor4DCoord& operator+=(Base const& b) {
        Base::operator+=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Tensor4DCoord& operator-=(Base const& b) {
        Base::operator-=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Tensor4DCoord& operator*=(Base const& b) {
        Base::operator*=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Tensor4DCoord& operator/=(Base const& b) {
        Base::operator/=(b);
        return *this;
    }
};


struct Tensor5DCoord : public Coord<5> {
    using Base = Coord<5>;

    using Index = typename Base::Index;

    using LongIndex = typename Base::LongIndex;

    static int const kN = 0;

    static int const kD = 1;

    static int const kH = 2;

    static int const kW = 3;

    static int const kC = 4;


    CUTLASS_HOST_DEVICE
    Tensor5DCoord() {}

    CUTLASS_HOST_DEVICE
    Tensor5DCoord(Coord<5> const& coord) : Base(coord) {}

    CUTLASS_HOST_DEVICE
    Tensor5DCoord(Index n, Index d, Index h, Index w, Index c)
            : Base(make_Coord(n, d, h, w, c)) {}

    CUTLASS_HOST_DEVICE
    Index const& n() const { return this->at(kN); }

    CUTLASS_HOST_DEVICE
    Index& n() { return this->at(kN); }

    CUTLASS_HOST_DEVICE
    Index const& d() const { return this->at(kD); }

    CUTLASS_HOST_DEVICE
    Index& d() { return this->at(kD); }

    CUTLASS_HOST_DEVICE
    Index const& h() const { return this->at(kH); }

    CUTLASS_HOST_DEVICE
    Index& h() { return this->at(kH); }

    CUTLASS_HOST_DEVICE
    Index const& w() const { return this->at(kW); }

    CUTLASS_HOST_DEVICE
    Index& w() { return this->at(kW); }

    CUTLASS_HOST_DEVICE
    Index const& c() const { return this->at(kC); }

    CUTLASS_HOST_DEVICE
    Index& c() { return this->at(kC); }


    CUTLASS_HOST_DEVICE
    Tensor5DCoord operator+(Base const& b) const {
        return Tensor5DCoord(Base::operator+(b));
    }

    CUTLASS_HOST_DEVICE
    Tensor5DCoord operator-(Base const& b) const {
        return Tensor5DCoord(Base::operator-(b));
    }

    CUTLASS_HOST_DEVICE
    Tensor5DCoord operator*(Base const& b) const {
        return Tensor5DCoord(Base::operator*(b));
    }

    CUTLASS_HOST_DEVICE
    Tensor5DCoord operator/(Base const& b) const {
        return Tensor5DCoord(Base::operator/(b));
    }

    CUTLASS_HOST_DEVICE
    Tensor5DCoord& operator+=(Base const& b) {
        Base::operator+=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Tensor5DCoord& operator-=(Base const& b) {
        Base::operator-=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Tensor5DCoord& operator*=(Base const& b) {
        Base::operator*=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    Tensor5DCoord& operator/=(Base const& b) {
        Base::operator/=(b);
        return *this;
    }
};


}
