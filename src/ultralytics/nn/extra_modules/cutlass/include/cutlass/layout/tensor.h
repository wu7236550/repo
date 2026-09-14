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
#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include "assert.h"
#endif
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/coord.h"
#include "cutlass/tensor_coord.h"

namespace cutlass {
namespace layout {


class TensorNHWC {
public:
    static int const kRank = 4;

    static int const kStrideRank = 3;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Tensor4DCoord;

    using Stride = Coord<kStrideRank>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorNHWC(Stride const& stride = Stride(0)) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    TensorNHWC(typename Stride::Index stride_w,
               typename Stride::Index stride_h,
               typename Stride::Index stride_n
               )
            : stride_(make_Coord(stride_w, stride_h, stride_n)) {}

    CUTLASS_HOST_DEVICE
    static TensorNHWC packed(TensorCoord const& extent) {
        return TensorNHWC(make_Coord(extent.c(), extent.w() * extent.c(),
                                     extent.h() * extent.w() * extent.c()));
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return coord.c() + LongIndex(stride_[0] * coord.w()) +
               LongIndex(stride_[1] * coord.h()) +
               LongIndex(stride_[2] * coord.n());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(PitchLinearCoord coord) const {
        return coord.contiguous() + LongIndex(coord.strided() * stride_[2]);
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex index) const {
        int n = 0, h = 0, w = 0, c = 0;

#if defined(__CUDA_ARCH__)
        int tmp = 0;
        c = int(index % static_cast<int>(stride_[0]));

        unsigned int hw_mul, hw_shr, w_mul, w_shr, c_mul, c_shr;

        find_divisor(hw_mul, hw_shr, stride_[2]);
        find_divisor(w_mul, w_shr, stride_[1]);
        find_divisor(c_mul, c_shr, stride_[0]);

        fast_divmod(n, tmp, index, int(stride_[2]), hw_mul, hw_shr);
        fast_divmod(h, w, tmp, int(stride_[1]), w_mul, w_shr);
        fast_divmod(w, tmp, w, int(stride_[0]), c_mul, c_shr);
#else

        n = int(index / (stride_[0] * stride_[1] * stride_[2]));
        LongIndex residual = index % (stride_[0] * stride_[1] * stride_[2]);

        h = int(residual / (stride_[0] * stride_[1]));
        residual = (residual % (stride_[0] * stride_[1]));

        w = int(residual / stride_[0]);
        c = int(residual % stride_[0]);

#endif
        return TensorCoord(n, h, w, c);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        if ((extent.c() > stride_[0]) ||
            (extent.w() * stride_[0] > stride_[1]) ||
            (extent.h() * stride_[1] > stride_[2])) {
            assert(0);
        }
        return extent.n() * stride_[2];
    }
};

class TensorCHWN {
public:
    static int const kRank = 4;

    static int const kStrideRank = 3;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Tensor4DCoord;

    using Stride = Coord<kStrideRank>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorCHWN(Stride const& stride = Stride(0)) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    TensorCHWN(typename Stride::Index stride_w,
               typename Stride::Index stride_h,
               typename Stride::Index stride_c
               )
            : stride_(make_Coord(stride_w, stride_h, stride_c)) {}

    CUTLASS_HOST_DEVICE
    static TensorCHWN packed(TensorCoord const& extent) {
        return TensorCHWN(make_Coord(extent.n(), extent.w() * extent.n(),
                                     extent.h() * extent.w() * extent.n()));
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return coord.n() + LongIndex(stride_[0] * coord.w()) +
               LongIndex(stride_[1] * coord.h()) +
               LongIndex(stride_[2] * coord.c());
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(PitchLinearCoord coord) const {
        return coord.contiguous() + LongIndex(coord.strided() * stride_[2]);
    }

    CUTLASS_HOST_DEVICE
    TensorCoord inverse(LongIndex index) const {
        int n = 0, h = 0, w = 0, c = 0;

#if defined(__CUDA_ARCH__)
        int tmp = 0;
        n = int(index % static_cast<int>(stride_[0]));

        unsigned int hw_mul, hw_shr, w_mul, w_shr, n_mul, n_shr;

        find_divisor(hw_mul, hw_shr, stride_[2]);
        find_divisor(w_mul, w_shr, stride_[1]);
        find_divisor(n_mul, n_shr, stride_[0]);

        fast_divmod(c, tmp, index, int(stride_[2]), hw_mul, hw_shr);
        fast_divmod(h, w, tmp, int(stride_[1]), w_mul, w_shr);
        fast_divmod(w, tmp, w, int(stride_[0]), n_mul, n_shr);
#else

        c = int(index / (stride_[0] * stride_[1] * stride_[2]));
        LongIndex residual = index % (stride_[0] * stride_[1] * stride_[2]);

        h = int(residual / (stride_[0] * stride_[1]));
        residual = (residual % (stride_[0] * stride_[1]));

        w = int(residual / stride_[0]);
        n = int(residual % stride_[0]);

#endif
        return TensorCoord(n, h, w, c);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        if ((extent.n() > stride_[0]) ||
            (extent.w() * stride_[0] > stride_[1]) ||
            (extent.h() * stride_[1] > stride_[2])) {
            assert(0);
        }
        return extent.c() * stride_[2];
    }
};


class TensorNCHW {
public:
    static int const kRank = 4;

    static int const kStrideRank = 3;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Tensor4DCoord;

    using Stride = Coord<kStrideRank>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorNCHW(Stride const& stride = Stride(0)) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorNCHW packed(TensorCoord const& extent) {
        return TensorNCHW(make_Coord(extent.w(), extent.w() * extent.h(),
                                     extent.h() * extent.w() * extent.c()));
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return coord.w() + LongIndex(stride_[0] * coord.h()) +
               LongIndex(stride_[1] * coord.c()) +
               LongIndex(stride_[2] * coord.n());
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return extent.n() * stride_[2];
    }
};


template <int Interleave>
class TensorNCxHWx {
public:
    static int const kInterleave = Interleave;

    static int const kRank = 4;

    static int const kStrideRank = 3;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Tensor4DCoord;

    using Stride = Coord<kStrideRank>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorNCxHWx(Stride const& stride = Stride(0)) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorNCxHWx packed(TensorCoord const& extent) {
        return TensorNCxHWx(make_Coord(kInterleave * extent.w(),
                                       kInterleave * extent.w() * extent.h(),
                                       extent.h() * extent.w() * extent.c()));
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        Index c_minor = (coord.c() % kInterleave);
        Index c_major = (coord.c() / kInterleave);

        return c_minor + LongIndex(kInterleave * coord.w()) +
               LongIndex(stride_[0] * coord.h()) +
               LongIndex(stride_[1] * c_major) +
               LongIndex(stride_[2] * coord.n());
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return extent.n() * stride_[2];
    }
};


template <int Interleave>
class TensorCxRSKx {
public:
    static int const kInterleave = Interleave;

    static int const kRank = 4;

    static int const kStrideRank = 3;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Tensor4DCoord;

    using Stride = Coord<kStrideRank>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorCxRSKx(Stride const& stride = Stride(0)) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorCxRSKx packed(TensorCoord const& extent) {
        return TensorCxRSKx(make_Coord(
                kInterleave * extent.n(), kInterleave * extent.n() * extent.w(),
                kInterleave * extent.n() * extent.w() * extent.h()));
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        Index c_minor = (coord.c() % kInterleave);
        Index c_major = (coord.c() / kInterleave);

        return c_minor + LongIndex(kInterleave * coord.n()) +
               LongIndex(stride_[0] * coord.w()) +
               LongIndex(stride_[1] * coord.h()) +
               LongIndex(stride_[2] * c_major);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(PitchLinearCoord const& coord) const {
        return (coord.contiguous() % kInterleave) +
               LongIndex((coord.contiguous() / kInterleave) * stride_[2]) +
               LongIndex(coord.strided() * kInterleave);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return (extent.c() / kInterleave * stride_[2]);
    }
};

template <int Interleave>
class TensorKxRSCx {
public:
    static int const kInterleave = Interleave;

    static int const kRank = 4;

    static int const kStrideRank = 3;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Tensor4DCoord;

    using Stride = Coord<kStrideRank>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorKxRSCx(Stride const& stride = Stride(0)) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorKxRSCx packed(TensorCoord const& extent) {
        return TensorKxRSCx(make_Coord(
                kInterleave * extent.c(), kInterleave * extent.c() * extent.w(),
                kInterleave * extent.c() * extent.w() * extent.h()));
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        Index n_minor = (coord.n() % kInterleave);
        Index n_major = (coord.n() / kInterleave);

        return n_minor + LongIndex(kInterleave * coord.c()) +
               LongIndex(stride_[0] * coord.w()) +
               LongIndex(stride_[1] * coord.h()) +
               LongIndex(stride_[2] * n_major);
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(PitchLinearCoord const& coord) const {
        return (coord.contiguous() % kInterleave) +
               LongIndex((coord.contiguous() / kInterleave) * stride_[2]) +
               LongIndex(coord.strided() * kInterleave);
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return (extent.n() / kInterleave * stride_[2]);
    }
};

template <int Interleave>
class TensorCKxRSx {
public:
    static int const kInterleave = Interleave;

    static int const kRank = 4;

    static int const kStrideRank = 3;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Tensor4DCoord;

    using Stride = Coord<kStrideRank>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorCKxRSx(Stride const& stride = Stride(0)) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    static TensorCKxRSx packed(TensorCoord const& extent) {
        return TensorCKxRSx(make_Coord(kInterleave * extent.w(),
                                       kInterleave * extent.w() * extent.h(),
                                       extent.w() * extent.h() * extent.n()));
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        Index n_minor = (coord.n() % kInterleave);
        Index n_major = (coord.n() / kInterleave);

        return n_minor + LongIndex(kInterleave * coord.w()) +
               LongIndex(stride_[0] * coord.h()) +
               LongIndex(stride_[1] * n_major) +
               LongIndex(stride_[2] * coord.c());
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        return (extent.c() * stride_[2]);
    }
};


class TensorNDHWC {
public:
    static int const kRank = 5;

    static int const kStrideRank = 4;

    using Index = int32_t;

    using LongIndex = int64_t;

    using TensorCoord = Tensor5DCoord;

    using Stride = Coord<kStrideRank>;

private:

    Stride stride_;

public:

    CUTLASS_HOST_DEVICE
    TensorNDHWC(Stride const& stride = Stride(0)) : stride_(stride) {}

    CUTLASS_HOST_DEVICE
    TensorNDHWC(typename Stride::Index c, typename Stride::Index wc,
                typename Stride::Index hwc, typename Stride::Index dhwc)
            : stride_(make_Coord(c, wc, hwc, dhwc)) {}

    CUTLASS_HOST_DEVICE
    static TensorNDHWC packed(TensorCoord const& extent) {
        return TensorNDHWC(
                make_Coord(extent.c(), extent.w() * extent.c(),
                           extent.h() * extent.w() * extent.c(),
                           extent.d() * extent.h() * extent.w() * extent.c()));
    }

    CUTLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const& coord) const {
        return coord.c() + LongIndex(stride_[0] * coord.w()) +
               LongIndex(stride_[1] * coord.h()) +
               LongIndex(stride_[2] * coord.d()) +
               LongIndex(stride_[3] * coord.n());
    }

    CUTLASS_HOST_DEVICE
    Stride stride() const { return stride_; }

    CUTLASS_HOST_DEVICE
    Stride& stride() { return stride_; }

    CUTLASS_HOST_DEVICE
    LongIndex capacity(TensorCoord const& extent) const {
        if ((extent.c() > stride_[0]) ||
            (extent.w() * stride_[0] > stride_[1]) ||
            (extent.h() * stride_[1] > stride_[2]) ||
            (extent.d() * stride_[2] > stride_[3])) {
            assert(0);
        }
        return extent.n() * stride_[3];
    }
};


}
}
