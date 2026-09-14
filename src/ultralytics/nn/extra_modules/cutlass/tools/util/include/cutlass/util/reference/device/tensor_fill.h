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

#if !defined(__CUDACC_RTC__)

#include <utility>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <cstdint>

#endif

#include <curand_kernel.h>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/complex.h"
#include "cutlass/tensor_view.h"

#include "cutlass/util/reference/device/tensor_foreach.h"
#include "cutlass/util/distribution.h"


namespace cutlass {
namespace reference {
namespace device {


namespace detail {

template <typename FloatType>
CUTLASS_DEVICE FloatType random_normal_float(curandState_t* state) {
    return curand_normal(state);
}

template <>
CUTLASS_DEVICE double random_normal_float<double>(curandState_t* state) {
    return curand_normal_double(state);
}

template <typename FloatType>
CUTLASS_DEVICE FloatType random_uniform_float(curandState_t* state) {
    return curand_uniform(state);
}

template <>
CUTLASS_DEVICE double random_uniform_float<double>(curandState_t* state) {
    return curand_uniform_double(state);
}

template <typename Element>
struct RandomGaussianFunc {
    using FloatType = typename std::conditional<(sizeof(Element) > 4), double,
                                                float>::type;
    using IntType = typename std::conditional<(sizeof(Element) > 4), int64_t,
                                              int>::type;

    struct Params {

        uint64_t seed;
        FloatType mean;
        FloatType stddev;
        int int_scale;
        FloatType float_scale_up;
        FloatType float_scale_down;


        Params(uint64_t seed_ = 0, Element mean_ = 0, Element stddev_ = 1,
               int int_scale_ = -1)
                : seed(seed_),
                  mean(static_cast<FloatType>(mean_)),
                  stddev(static_cast<FloatType>(stddev_)),
                  int_scale(int_scale_) {
            float_scale_up = FloatType(IntType(1) << int_scale);
            float_scale_up += FloatType(0.5) * float_scale_up;
            float_scale_down =
                    FloatType(1) / FloatType(IntType(1) << int_scale);
        }
    };


    Params params;

    curandState_t rng_state;


    CUTLASS_DEVICE
    RandomGaussianFunc(Params const& params) : params(params) {
        uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

        curand_init(params.seed, gtid, 0, &rng_state);
    }

    CUTLASS_DEVICE
    Element operator()() {
        FloatType rnd = random_normal_float<FloatType>(&rng_state);
        rnd = params.mean + params.stddev * rnd;

        Element result;
        if (params.int_scale >= 0) {
            rnd = FloatType(IntType(rnd * params.float_scale_up));
            result = Element(rnd * params.float_scale_down);
        } else {
            result = Element(rnd);
        }

        return result;
    }
};

template <typename Real>
struct RandomGaussianFunc<complex<Real>> {
    using Element = complex<Real>;
    using FloatType =
            typename std::conditional<(sizeof(Real) > 4), double, float>::type;
    using IntType =
            typename std::conditional<(sizeof(Real) > 4), int64_t, int>::type;

    struct Params {

        uint64_t seed;
        FloatType mean;
        FloatType stddev;
        int int_scale;
        FloatType float_scale_up;
        FloatType float_scale_down;


        Params(uint64_t seed_ = 0, Real mean_ = 0, Real stddev_ = 1,
               int int_scale_ = -1)
                : seed(seed_),
                  mean(static_cast<FloatType>(mean_)),
                  stddev(static_cast<FloatType>(stddev_)),
                  int_scale(int_scale_) {
            float_scale_up = FloatType(IntType(1) << int_scale);
            float_scale_up += FloatType(0.5) * float_scale_up;
            float_scale_down =
                    FloatType(1) / FloatType(IntType(1) << int_scale);
        }
    };


    Params params;

    curandState_t rng_state;


    CUTLASS_DEVICE
    RandomGaussianFunc(Params const& params) : params(params) {
        uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

        curand_init(params.seed, gtid, 0, &rng_state);
    }

    CUTLASS_DEVICE
    Element operator()() {
        FloatType rnd_r = random_normal_float<FloatType>(&rng_state);
        FloatType rnd_i = random_normal_float<FloatType>(&rng_state);
        rnd_r = params.mean + params.stddev * rnd_r;
        rnd_i = params.mean + params.stddev * rnd_i;

        Element result;
        if (params.int_scale >= 0) {
            rnd_r = FloatType(IntType(rnd_r * params.float_scale_up));
            rnd_i = FloatType(IntType(rnd_i * params.float_scale_down));

            result = {Real(rnd_r * params.float_scale_down),
                      Real(rnd_i * params.float_scale_down)};
        } else {
            result = Element(Real(rnd_r), Real(rnd_i));
        }

        return result;
    }
};

template <typename Element,
          typename Layout>
struct TensorFillRandomGaussianFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    using RandomFunc = RandomGaussianFunc<Element>;

    struct Params {

        TensorView view;
        typename RandomFunc::Params random;


        Params(TensorView view_ = TensorView(),
               typename RandomFunc::Params random_ =
                       typename RandomFunc::Params())
                : view(view_), random(random_) {}
    };


    Params params;
    RandomFunc random;


    CUTLASS_DEVICE
    TensorFillRandomGaussianFunc(Params const& params)
            : params(params), random(params.random) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        params.view.at(coord) = random();
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorFillRandomGaussian(
  TensorView<Element, Layout> view,
  uint64_t seed,
  Element mean = Element(0),
  Element stddev = Element(1),
  int bits = -1) {

    using RandomFunc = detail::RandomGaussianFunc<Element>;
    using Func = detail::TensorFillRandomGaussianFunc<Element, Layout>;
    using Params = typename Func::Params;

    TensorForEach<Func, Layout::kRank, Params>(
            view.extent(), Params(view, typename RandomFunc::Params(
                                                seed, mean, stddev, bits)));
}


template <typename Element>
void BlockFillRandomGaussian(
        Element* ptr, size_t capacity,
        uint64_t seed,
        typename RealType<Element>::Type
                mean,
        typename RealType<Element>::Type
                stddev,
        int bits = -1) {

    using RandomFunc = detail::RandomGaussianFunc<Element>;

    typename RandomFunc::Params params(seed, mean, stddev, bits);

    BlockForEach<Element, RandomFunc>(ptr, capacity, params);
}


namespace detail {

template <typename Element>
struct RandomUniformFunc {
    using FloatType = typename std::conditional<(sizeof(Element) > 4), double,
                                                float>::type;

    using IntType = typename std::conditional<(sizeof(Element) > 4), int64_t,
                                              int>::type;

    struct Params {

        uint64_t seed;
        FloatType range;
        FloatType max;
        int int_scale;
        FloatType float_scale_up;
        FloatType float_scale_down;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(uint64_t seed_ = 0, Element max_ = 1, Element min = 0,
               int int_scale_ = -1)
                : seed(seed_),
                  range(static_cast<FloatType>(max_ - min)),
                  max(static_cast<FloatType>(max_)),
                  int_scale(int_scale_) {
            float_scale_up = FloatType(IntType(1) << int_scale);
            float_scale_up += FloatType(0.5) * float_scale_up;
            float_scale_down =
                    FloatType(1) / FloatType(IntType(1) << int_scale);
        }
    };


    Params params;

    curandState_t rng_state;


    CUTLASS_DEVICE
    RandomUniformFunc(Params const& params) : params(params) {
        uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

        curand_init(params.seed, gtid, 0, &rng_state);
    }

    CUTLASS_DEVICE
    Element operator()() {
        FloatType rnd = random_uniform_float<FloatType>(&rng_state);
        rnd = params.max - params.range * rnd;

        Element result;

        if (params.int_scale >= 0) {
            rnd = FloatType(IntType(rnd * params.float_scale_up));
            result = Element(rnd * params.float_scale_down);
        } else {
            result = Element(rnd);
        }

        return result;
    }
};

template <typename Real>
struct RandomUniformFunc<complex<Real>> {
    using Element = complex<Real>;

    using FloatType =
            typename std::conditional<(sizeof(Real) > 4), double, float>::type;

    using IntType =
            typename std::conditional<(sizeof(Real) > 4), int64_t, int>::type;

    struct Params {

        uint64_t seed;
        FloatType range;
        FloatType min;
        int int_scale;
        FloatType float_scale_up;
        FloatType float_scale_down;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(uint64_t seed_ = 0, FloatType max = 1, FloatType min_ = 0,
               int int_scale_ = -1)
                : seed(seed_),
                  range(static_cast<FloatType>(max - min_)),
                  min(static_cast<FloatType>(min_)),
                  int_scale(int_scale_) {
            float_scale_up = FloatType(IntType(1) << int_scale);
            float_scale_up += FloatType(0.5) * float_scale_up;
            float_scale_down =
                    FloatType(1) / FloatType(IntType(1) << int_scale);
        }
    };


    Params params;

    curandState_t rng_state;


    CUTLASS_DEVICE
    RandomUniformFunc(Params const& params) : params(params) {
        uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

        curand_init(params.seed, gtid, 0, &rng_state);
    }

    CUTLASS_DEVICE
    Element operator()() {
        FloatType rnd_r = random_uniform_float<FloatType>(&rng_state);
        FloatType rnd_i = random_uniform_float<FloatType>(&rng_state);

        rnd_r = params.min + params.range * rnd_r;
        rnd_i = params.min + params.range * rnd_i;

        Element result;

        if (params.int_scale >= 0) {
            rnd_r = FloatType(IntType(rnd_r * params.float_scale_up));
            rnd_i = FloatType(IntType(rnd_i * params.float_scale_up));

            result = {Real(rnd_r * params.float_scale_down),
                      Real(rnd_i * params.float_scale_down)};
        } else {
            result = Element(Real(rnd_r), Real(rnd_i));
        }

        return result;
    }
};

template <typename Element,
          typename Layout>
struct TensorFillRandomUniformFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    using RandomFunc = RandomUniformFunc<Element>;

    struct Params {

        TensorView view;
        typename RandomFunc::Params random;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(TensorView view_ = TensorView(),
               typename RandomFunc::Params random_ = RandomFunc::Params())
                : view(view_), random(random_) {}
    };


    Params params;
    RandomFunc random;


    CUTLASS_DEVICE
    TensorFillRandomUniformFunc(Params const& params)
            : params(params), random(params.random) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        params.view.at(coord) = random();
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorFillRandomUniform(
  TensorView<Element, Layout> view,
  uint64_t seed,
  Element max = Element(1),
  Element min = Element(0),
  int bits = -1) {

    using RandomFunc = detail::RandomUniformFunc<Element>;
    using Func = detail::TensorFillRandomUniformFunc<Element, Layout>;
    using Params = typename Func::Params;

    typename RandomFunc::Params random(seed, max, min, bits);

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, random));
}


template <typename Element>
void BlockFillRandomUniform(
        Element* ptr, size_t capacity,
        uint64_t seed,
        typename RealType<Element>::Type max,
        typename RealType<Element>::Type min,
        int bits = -1) {

    using RandomFunc = detail::RandomUniformFunc<Element>;

    typename RandomFunc::Params params(seed, max, min, bits);

    BlockForEach<Element, RandomFunc>(ptr, capacity, params);
}


namespace detail {

template <typename Element>
struct RandomSparseMetaFunc {
    using FloatType = float;

    using IntType = int32_t;

    struct Params {

        uint64_t seed;
        FloatType range;
        int MetaSizeInBits;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(uint64_t seed_ = 0, int MetaSizeInBits_ = 2)
                : seed(seed_), MetaSizeInBits(MetaSizeInBits_) {
            if (MetaSizeInBits_ == 2) {
                range = 6;
            } else if (MetaSizeInBits_ == 4) {
                range = 2;
            }
        }
    };


    Params params;

    curandState_t rng_state;


    CUTLASS_DEVICE
    RandomSparseMetaFunc(Params const& params) : params(params) {
        uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

        curand_init(params.seed, gtid, 0, &rng_state);
    }

    CUTLASS_DEVICE
    Element operator()() {
        Element FourToTwoMeta[6] = {0x4, 0x8, 0x9, 0xc, 0xd, 0xe};
        Element TwoToOneMeta[2] = {0x4, 0xe};

        Element* MetaArray =
                (params.MetaSizeInBits == 2) ? FourToTwoMeta : TwoToOneMeta;

        Element result = 0x0;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cutlass::sizeof_bits<Element>::value / 4; ++i) {
            FloatType rnd = random_uniform_float<FloatType>(&rng_state);
            rnd = params.range * rnd;
            Element meta = MetaArray[(int)rnd];

            result = (Element)(result | ((Element)(meta << (i * 4))));
        }

        return result;
    }
};

template <typename Element,
          typename Layout>
struct TensorFillRandomSparseMetaFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    using RandomFunc = RandomSparseMetaFunc<Element>;

    struct Params {

        TensorView view;
        typename RandomFunc::Params random;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(TensorView view_ = TensorView(),
               typename RandomFunc::Params random_ = RandomFunc::Params())
                : view(view_), random(random_) {}
    };


    Params params;
    RandomFunc random;


    CUTLASS_DEVICE
    TensorFillRandomSparseMetaFunc(Params const& params)
            : params(params), random(params.random) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        params.view.at(coord) = random();
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorFillRandomSparseMeta(
  TensorView<Element, Layout> view,
  uint64_t seed,
  int MetaSizeInBits = 2) {

    using RandomFunc = detail::RandomSparseMetaFunc<Element>;
    using Func = detail::TensorFillRandomUniformFunc<Element, Layout>;
    using Params = typename Func::Params;

    typename RandomFunc::Params random(seed, MetaSizeInBits);

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, random));
}


template <typename Element>
void BlockFillRandomSparseMeta(Element* ptr, size_t capacity,
                               uint64_t seed,
                               int MetaSizeInBits = 2) {

    using RandomFunc = detail::RandomSparseMetaFunc<Element>;

    typename RandomFunc::Params params(seed, MetaSizeInBits);

    BlockForEach<Element, RandomFunc>(ptr, capacity, params);
}


namespace detail {

template <typename Element,
          typename Layout>
struct TensorFillDiagonalFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    struct Params {

        TensorView view;
        Element diag;
        Element other;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(TensorView view_ = TensorView(), Element diag_ = Element(1),
               Element other_ = Element(0))
                : view(view_), diag(diag_), other(other_) {}
    };


    Params params;


    CUTLASS_DEVICE
    TensorFillDiagonalFunc(Params const& params) : params(params) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        bool is_diag = true;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < Layout::kRank; ++i) {
            if (coord[i] != coord[i - 1]) {
                is_diag = false;
                break;
            }
        }

        params.view.at(coord) = (is_diag ? params.diag : params.other);
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorFillDiagonal(
  TensorView<Element, Layout> view,
  Element diag = Element(1),
  Element other = Element(0)) {

    typedef detail::TensorFillDiagonalFunc<Element, Layout> Func;
    typedef typename Func::Params Params;

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, diag, other));
}


template <
  typename Element,
  typename Layout>
void TensorFill(
  TensorView<Element, Layout> view,
  Element val = Element(0)) {

    TensorFillDiagonal(view, val, val);
}


template <
  typename Element,
  typename Layout>
void TensorFillIdentity(
  TensorView<Element, Layout> view) {

    TensorFillDiagonal(view, Element(1), Element(0));
}


namespace detail {

template <typename Element,
          typename Layout>
struct TensorUpdateDiagonalFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    struct Params {

        TensorView view;
        Element diag;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(TensorView view_ = TensorView(), Element diag_ = Element(1))
                : view(view_), diag(diag_) {}
    };


    Params params;


    CUTLASS_DEVICE
    TensorUpdateDiagonalFunc(Params const& params) : params(params) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        bool is_diag = true;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < Layout::kRank; ++i) {
            if (coord[i] != coord[i - 1]) {
                is_diag = false;
                break;
            }
        }

        if (is_diag) {
            params.view.at(coord) = params.diag;
        }
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorUpdateDiagonal(
  TensorView<Element, Layout> view,
  Element diag = Element(1)) {
    typedef detail::TensorUpdateDiagonalFunc<Element, Layout> Func;
    typedef typename Func::Params Params;

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, diag));
}


namespace detail {

template <typename Element,
          typename Layout>
struct TensorUpdateOffDiagonalFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    struct Params {

        TensorView view;
        Element other;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(TensorView view_ = TensorView(), Element other_ = Element(0))
                : view(view_), other(other_) {}
    };


    Params params;


    CUTLASS_DEVICE
    TensorUpdateOffDiagonalFunc(Params const& params) : params(params) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        bool is_diag = true;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < Layout::kRank; ++i) {
            if (coord[i] != coord[i - 1]) {
                is_diag = false;
                break;
            }
        }

        if (!is_diag) {
            params.view.at(coord) = params.other;
        }
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorUpdateOffDiagonal(
  TensorView<Element, Layout> view,
  Element other = Element(1)) {
    typedef detail::TensorUpdateOffDiagonalFunc<Element, Layout> Func;
    typedef typename Func::Params Params;

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, other));
}


namespace detail {

template <typename Element,
          typename Layout>
struct TensorFillLinearFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    struct Params {

        TensorView view;
        Array<Element, Layout::kRank> v;
        Element s;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(TensorView view_,
               Array<Element, Layout::kRank> const& v_, Element s_ = Element(0))
                : view(view_), v(v_), s(s_) {}
    };


    Params params;


    CUTLASS_DEVICE
    TensorFillLinearFunc(Params const& params) : params(params) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        Element sum = params.s;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Layout::kRank; ++i) {
            sum += params.v[i] * Element(coord[i]);
        }

        params.view.at(coord) = sum;
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorFillLinear(
  TensorView<Element, Layout> view,
  Array<Element, Layout::kRank> const & v,
  Element s = Element(0)) {
    using Func = detail::TensorFillLinearFunc<Element, Layout>;
    using Params = typename Func::Params;

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, v, s));
}


template <typename Element>
void BlockFillSequential(Element* ptr, int64_t capacity, Element v = Element(1),
                         Element s = Element(0)) {}


template <typename Element>
void BlockFillRandom(Element* ptr, size_t capacity, uint64_t seed,
                     Distribution dist) {
    using Real = typename RealType<Element>::Type;

    if (dist.kind == Distribution::Gaussian) {
        BlockFillRandomGaussian<Element>(
                ptr, capacity, seed, static_cast<Real>(dist.gaussian.mean),
                static_cast<Real>(dist.gaussian.stddev), dist.int_scale);
    } else if (dist.kind == Distribution::Uniform) {
        BlockFillRandomUniform<Element>(
                ptr, capacity, seed, static_cast<Real>(dist.uniform.max),
                static_cast<Real>(dist.uniform.min), dist.int_scale);
    }
}


namespace detail {

template <typename Element,
          typename Layout>
struct TensorCopyDiagonalInFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    struct Params {

        TensorView view;
        Element const* ptr;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(TensorView view_,
               Element const* ptr_)
                : view(view_), ptr(ptr_) {}
    };


    Params params;


    CUTLASS_DEVICE
    TensorCopyDiagonalInFunc(Params const& params) : params(params) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        bool is_diagonal = true;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < Layout::kRank; ++i) {
            if (coord[i] != coord[0]) {
                is_diagonal = false;
            }
        }
        if (is_diagonal) {
            params.view.at(coord) = params.ptr[coord[0]];
        }
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorCopyDiagonalIn(
  TensorView<Element, Layout> view,
  Element const *ptr) {

    using Func = detail::TensorCopyDiagonalInFunc<Element, Layout>;
    using Params = typename Func::Params;

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, ptr));
}


namespace detail {

template <typename Element,
          typename Layout>
struct TensorCopyDiagonalOutFunc {
    using TensorView = TensorView<Element, Layout>;

    typedef typename TensorView::Element T;

    typedef typename TensorView::TensorCoord TensorCoord;

    struct Params {

        TensorView view;
        Element* ptr;

        CUTLASS_HOST_DEVICE
        Params() {}


        Params(TensorView view_,
               Element* ptr_)
                : view(view_), ptr(ptr_) {}
    };


    Params params;


    CUTLASS_DEVICE
    TensorCopyDiagonalOutFunc(Params const& params) : params(params) {}

    CUTLASS_DEVICE
    void operator()(TensorCoord const& coord) {
        bool is_diagonal = true;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < Layout::kRank; ++i) {
            if (coord[i] != coord[0]) {
                is_diagonal = false;
            }
        }
        if (is_diagonal) {
            params.ptr[coord[0]] = params.view.at(coord);
        }
    }
};

}


template <
  typename Element,
  typename Layout>
void TensorCopyDiagonalOut(
  Element *ptr,
  TensorView<Element, Layout> view) {

    using Func = detail::TensorCopyDiagonalOutFunc<Element, Layout>;
    using Params = typename Func::Params;

    TensorForEach<Func, Layout::kRank, Params>(view.extent(),
                                               Params(view, ptr));
}


}
}
}
