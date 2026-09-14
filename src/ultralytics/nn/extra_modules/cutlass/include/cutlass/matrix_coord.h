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


struct MatrixCoord : public Coord<2, int> {
public:
    using Index = int;

    using Base = Coord<2, Index>;

private:
    static int const kRow = 0;

    static int const kColumn = 1;

public:

    CUTLASS_HOST_DEVICE
    MatrixCoord() {}

    CUTLASS_HOST_DEVICE
    MatrixCoord(Coord<2, Index> const& coord) : Base(coord) {}

    CUTLASS_HOST_DEVICE
    MatrixCoord(Index row, Index column) : Base(make_Coord(row, column)) {}

    CUTLASS_HOST_DEVICE
    Index const& row() const { return this->at(kRow); }

    CUTLASS_HOST_DEVICE
    Index& row() { return this->at(kRow); }

    CUTLASS_HOST_DEVICE
    Index const& column() const { return this->at(kColumn); }

    CUTLASS_HOST_DEVICE
    Index& column() { return this->at(kColumn); }


    CUTLASS_HOST_DEVICE
    MatrixCoord operator+(Base const& b) const {
        return MatrixCoord(Base::operator+(b));
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord operator-(Base const& b) const {
        return MatrixCoord(Base::operator-(b));
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord operator*(Base const& b) const {
        return MatrixCoord(Base::operator*(b));
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord operator/(Base const& b) const {
        return MatrixCoord(Base::operator/(b));
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord& operator+=(Base const& b) {
        Base::operator+=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord& operator-=(Base const& b) {
        Base::operator-=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord& operator*=(Base const& b) {
        Base::operator*=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    MatrixCoord& operator/=(Base const& b) {
        Base::operator/=(b);
        return *this;
    }
};


}
