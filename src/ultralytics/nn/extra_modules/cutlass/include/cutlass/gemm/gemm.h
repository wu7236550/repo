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
namespace gemm {


enum class Operand {
    kA,
    kB,
    kC,
    kD
};


template <
        int M = 1,
        int N = 1,
        int K = 1>
struct GemmShape {
    static int const kM = M;
    static int const kN = N;
    static int const kK = K;

    static int const kMN = M * N;
    static int const kMK = M * K;
    static int const kKN = N * K;
    static int const kMNK = M * N * K;

    static int const kCount = kMNK;


    CUTLASS_HOST_DEVICE
    static Coord<3> toCoord() { return make_Coord(kM, kN, kK); }
};


template <
        typename Shape>
using GemmShapeTranspose = GemmShape<Shape::kN, Shape::kM, Shape::kK>;


/// GemmCoord is a structure derived from Coord<3> that specifies a location
struct GemmCoord : public Coord<3, int> {
    typedef int Index;

    typedef Coord<3, Index> Base;

    static int const kM = 0;

    static int const kN = 1;

    static int const kK = 2;


    CUTLASS_HOST_DEVICE
    GemmCoord() {}

    CUTLASS_HOST_DEVICE
    GemmCoord(Coord<3, Index> const& coord)
            : Base(make_Coord(coord[0], coord[1], coord[2])) {}

    CUTLASS_HOST_DEVICE
    GemmCoord(Index m, Index n, Index k) : Base(make_Coord(m, n, k)) {}

    CUTLASS_HOST_DEVICE
    Index const& m() const { return this->at(kM); }

    CUTLASS_HOST_DEVICE
    Index& m() { return this->at(kM); }

    CUTLASS_HOST_DEVICE
    Index const& n() const { return this->at(kN); }

    CUTLASS_HOST_DEVICE
    Index& n() { return this->at(kN); }

    CUTLASS_HOST_DEVICE
    Index const& k() const { return this->at(kK); }

    CUTLASS_HOST_DEVICE
    Index& k() { return this->at(kK); }

    CUTLASS_HOST_DEVICE
    Coord<3> mnk() const { return make_Coord(m(), n(), k()); }

    CUTLASS_HOST_DEVICE
    Coord<3> knm() const { return make_Coord(k(), n(), m()); }

    CUTLASS_HOST_DEVICE
    Coord<2> nm() const { return make_Coord(n(), m()); }

    CUTLASS_HOST_DEVICE
    Coord<2> mn() const { return make_Coord(m(), n()); }

    CUTLASS_HOST_DEVICE
    Coord<2> mk() const { return make_Coord(m(), k()); }

    CUTLASS_HOST_DEVICE
    Coord<2> km() const { return make_Coord(k(), m()); }

    CUTLASS_HOST_DEVICE
    Coord<2> nk() const { return make_Coord(n(), k()); }

    CUTLASS_HOST_DEVICE
    Coord<2> kn() const { return make_Coord(k(), n()); }


    CUTLASS_HOST_DEVICE
    GemmCoord operator+(Base const& b) const {
        return GemmCoord(Base::operator+(b));
    }

    CUTLASS_HOST_DEVICE
    GemmCoord operator-(Base const& b) const {
        return GemmCoord(Base::operator-(b));
    }

    CUTLASS_HOST_DEVICE
    GemmCoord operator*(Base const& b) const {
        return GemmCoord(Base::operator*(b));
    }

    CUTLASS_HOST_DEVICE
    GemmCoord operator/(Base const& b) const {
        return GemmCoord(Base::operator/(b));
    }

    CUTLASS_HOST_DEVICE
    GemmCoord& operator+=(Base const& b) {
        Base::operator+=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    GemmCoord& operator-=(Base const& b) {
        Base::operator-=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    GemmCoord& operator*=(Base const& b) {
        Base::operator*=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    GemmCoord& operator/=(Base const& b) {
        Base::operator/=(b);
        return *this;
    }
};


/// BatchedGemmCoord is a structure derived from Coord<4> that specifies a
struct BatchedGemmCoord : public Coord<4, int> {
    typedef int Index;

    typedef Coord<4, Index> Base;

    static int const kM = 0;

    static int const kN = 1;

    static int const kK = 2;

    static int const kBatch = 3;


    CUTLASS_HOST_DEVICE
    BatchedGemmCoord() {}

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord(Base const& coord) : Base(coord) {}

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord(Index m, Index n, Index k, Index b)
            : Base(make_Coord(m, n, k, b)) {}

    CUTLASS_HOST_DEVICE
    Index const& m() const { return this->at(kM); }

    CUTLASS_HOST_DEVICE
    Index& m() { return this->at(kM); }

    CUTLASS_HOST_DEVICE
    Index const& n() const { return this->at(kN); }

    CUTLASS_HOST_DEVICE
    Index& n() { return this->at(kN); }

    CUTLASS_HOST_DEVICE
    Index const& k() const { return this->at(kK); }

    CUTLASS_HOST_DEVICE
    Index& k() { return this->at(kK); }

    CUTLASS_HOST_DEVICE
    Index const& batch() const { return this->at(kBatch); }

    CUTLASS_HOST_DEVICE
    Index& batch() { return this->at(kBatch); }

    CUTLASS_HOST_DEVICE
    GemmCoord mnk() const { return GemmCoord(m(), n(), k()); }

    CUTLASS_HOST_DEVICE
    Coord<4> mnkb() const { return make_Coord(m(), n(), k(), batch()); }


    CUTLASS_HOST_DEVICE
    BatchedGemmCoord operator+(Base const& b) const {
        return BatchedGemmCoord(Base::operator+(b));
    }

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord operator-(Base const& b) const {
        return BatchedGemmCoord(Base::operator-(b));
    }

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord operator*(Base const& b) const {
        return BatchedGemmCoord(Base::operator*(b));
    }

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord operator/(Base const& b) const {
        return BatchedGemmCoord(Base::operator/(b));
    }

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord& operator+=(Base const& b) {
        Base::operator+=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord& operator-=(Base const& b) {
        Base::operator-=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord& operator*=(Base const& b) {
        Base::operator*=(b);
        return *this;
    }

    CUTLASS_HOST_DEVICE
    BatchedGemmCoord& operator/=(Base const& b) {
        Base::operator/=(b);
        return *this;
    }
};


enum class GemmUniversalMode {
    kGemm,
    kGemmSplitKParallel,
    kBatched,
    kArray,
    kInvalid
};


}
}
