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
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"


namespace cutlass {
namespace transform {


template <typename Shape_, int Threads, int ElementsPerAccess = 1>
struct PitchLinearStripminedThreadMap {
    using TensorCoord = layout::PitchLinearCoord;

    using Shape = Shape_;

    static int const kThreads = Threads;

    static int const kElementsPerAccess = ElementsPerAccess;

    using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

    struct Detail {
        static_assert(!(Shape::kContiguous % kElementsPerAccess), "");

        static_assert(!((Shape::kContiguous * Shape::kStrided) %
                        (kThreads * kElementsPerAccess)),
                      "Shape must be divisible thread count.");

        using ShapeVec = layout::PitchLinearShape<
                Shape::kContiguous / kElementsPerAccess, Shape::kStrided>;

        static_assert((Threads < ShapeVec::kContiguous &&
                       !(ShapeVec::kContiguous % kThreads)) ||
                              (!(kThreads % ShapeVec::kContiguous) &&
                               !(ShapeVec::kStrided %
                                 (kThreads / ShapeVec::kContiguous))),
                      "Shape must be divisible by number of iterations of each "
                      "thread.");
    };

    using Iterations = typename platform::conditional<
            Threads >= Detail::ShapeVec::kContiguous,
            layout::PitchLinearShape<
                    1,
                    (Threads >= Detail::ShapeVec::kContiguous
                             ? Detail::ShapeVec::kStrided /
                                       (kThreads /
                                        Detail::ShapeVec::kContiguous)
                             : 0)>,
            layout::PitchLinearShape<Detail::ShapeVec::kContiguous / kThreads,
                                     Detail::ShapeVec::kStrided>>::type;

    using Delta = typename platform::conditional<
            Threads >= Detail::ShapeVec::kContiguous,
            layout::PitchLinearShape<1,
                                     kThreads / Detail::ShapeVec::kContiguous>,
            layout::PitchLinearShape<kThreads * kElementsPerAccess, 1>>::type;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        return TensorCoord((thread_id % Detail::ShapeVec::kContiguous) *
                                   kElementsPerAccess,
                           thread_id / Detail::ShapeVec::kContiguous);
    }
};

template <typename Shape, int Threads, int ElementsPerAccess = 1>
struct PitchLinearTilePolicyStripminedThreadContiguous {
    static_assert((Shape::kContiguous % (Threads * ElementsPerAccess)) == 0,
                  "Contiguous shape must divide number of threads");

    using TensorCoord = layout::PitchLinearCoord;

    static int const kThreads = Threads;
    static int const kElementsPerAccess = ElementsPerAccess;

    using Iterations =
            layout::PitchLinearShape<Shape::kContiguous /
                                             (kThreads * kElementsPerAccess),
                                     Shape::kStrided>;

    using Delta = layout::PitchLinearShape<1, 1>;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        return TensorCoord(
                thread_id * Iterations::kContiguous * kElementsPerAccess, 0);
    }
};

template <typename Shape, int Threads, int ElementsPerAccess = 1>
struct PitchLinearTilePolicyStripminedThreadStrided {
    static_assert((Shape::kStrided % Threads == 0),
                  "Strided shape must divide number of threads");

    using TensorCoord = layout::PitchLinearCoord;

    static int const kThreads = Threads;
    static int const kElementsPerAccess = ElementsPerAccess;

    using Iterations =
            layout::PitchLinearShape<Shape::kContiguous / kElementsPerAccess,
                                     Shape::kStrided / kThreads>;

    using Delta = layout::PitchLinearShape<1, 1>;

    using ShapeVec = Shape;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        return TensorCoord(0, thread_id * Iterations::kStrided);
    }
};


template <typename Shape, typename ThreadArrangement, int ElementsPerAccess = 1>
struct PitchLinear2DTilePolicyStripminedThreadContiguous {
    static_assert((Shape::kContiguous %
                   (ThreadArrangement::kContiguous * ElementsPerAccess)) == 0 &&
                          (Shape::kStrided % ThreadArrangement::kStrided) == 0,
                  "Shape must divide thread shape");

    using TensorCoord = layout::PitchLinearCoord;

    static int const kThreads = ThreadArrangement::kCount;
    static int const kElementsPerAccess = ElementsPerAccess;

    using Iterations = layout::PitchLinearShape<
            Shape::kContiguous /
                    (ThreadArrangement::kContiguous * kElementsPerAccess),
            Shape::kStrided / ThreadArrangement::kStrided>;

    using Delta = layout::PitchLinearShape<1, 1>;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        return TensorCoord(thread_id * kElementsPerAccess,
                           threadIdx.y * Iterations::kStrided);
    }
};


template <typename Shape_, int Threads, typename WarpThreadArrangement_,
          int ElementsPerAccess = 1>
struct PitchLinearWarpRakedThreadMap {
    using TensorCoord = layout::PitchLinearCoord;

    using Shape = Shape_;

    static int const kThreads = Threads;

    static int const kElementsPerAccess = ElementsPerAccess;

    using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

    struct Detail {
        using WarpThreadArrangement = WarpThreadArrangement_;

        static int const kWarpSize = WarpThreadArrangement::kCount;

        static int const kWarpCount = kThreads / kWarpSize;

        static_assert(!(Shape::kContiguous % kElementsPerAccess),
                      "Shape must be divisible by vector length.");

        using ShapeInAccesses = layout::PitchLinearShape<
                Shape::kContiguous / kElementsPerAccess, Shape::kStrided>;

        static_assert(
                !(ShapeInAccesses::kContiguous %
                  WarpThreadArrangement::kContiguous),
                "ShapeInAccesses must be divisible by WarpThreadArrangement.");

        static_assert(
                !(ShapeInAccesses::kStrided % WarpThreadArrangement::kStrided),
                "ShapeInAccesses must be divisible by WarpThreadArrangement.");

        using WarpAccessIterations = layout::PitchLinearShape<
                ShapeInAccesses::kContiguous /
                        WarpThreadArrangement::kContiguous,
                ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided>;

        static int const kWarpsStrided =
                (WarpAccessIterations::kStrided >= kWarpCount
                         ? kWarpCount
                         : WarpAccessIterations::kStrided);

        static int const kWarpsContiguous =
                (kWarpCount > WarpAccessIterations::kStrided
                         ? kWarpCount / kWarpsStrided
                         : 1);

        using WarpArrangement =
                layout::PitchLinearShape<kWarpsContiguous, kWarpsStrided>;
    };

    using Iterations =
            layout::PitchLinearShape<Detail::WarpAccessIterations::kContiguous /
                                             Detail::kWarpsContiguous,
                                     Detail::WarpAccessIterations::kStrided /
                                             Detail::kWarpsStrided>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    using Delta = layout::PitchLinearShape<
            Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
            Detail::WarpThreadArrangement::kStrided>;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        int warp_id = (thread_id / Detail::kWarpSize);
        int lane_id = (thread_id % Detail::kWarpSize);


        layout::PitchLinearCoord warp_footprint{
                Detail::WarpThreadArrangement::kContiguous *
                        Iterations::kContiguous,
                Detail::WarpThreadArrangement::kStrided * Iterations::kStrided};

        layout::PitchLinearCoord warp_offset{
                (warp_id % Detail::kWarpsContiguous),
                (warp_id / Detail::kWarpsContiguous)};

        layout::PitchLinearCoord thread_offset_in_warp{
                lane_id % Detail::WarpThreadArrangement::kContiguous,
                lane_id / Detail::WarpThreadArrangement::kContiguous};

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
                warp_footprint * warp_offset + thread_offset_in_warp;

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
                thread_offset_in_threadblock_tile_vec.contiguous() *
                        kElementsPerAccess,
                thread_offset_in_threadblock_tile_vec.strided()};

        return thread_offset_in_threadblock_tile_base;
    }
};


template <typename Shape_, int Threads, typename WarpThreadArrangement_,
          int ElementsPerAccess = 1>
struct PitchLinearStridedWarpRakedThreadMap {
    using TensorCoord = layout::PitchLinearCoord;

    using Shape = Shape_;

    static int const kThreads = Threads;

    using WarpThreadArrangement = WarpThreadArrangement_;

    static int const kElementsPerAccess = ElementsPerAccess;

    using BaseThreadMap = PitchLinearWarpRakedThreadMap<
            Shape, kThreads, WarpThreadArrangement, kElementsPerAccess>;

    using ThreadAccessShape = typename BaseThreadMap::ThreadAccessShape;

    struct Detail {
        using WarpThreadArrangement = WarpThreadArrangement_;

        using WarpAccessIterations =
                typename BaseThreadMap::Detail::WarpAccessIterations;

        static int const kWarpSize = BaseThreadMap::Detail::kWarpSize;

        static int const kWarpCount = BaseThreadMap::Detail::kWarpCount;

        using ShapeInAccesses = typename BaseThreadMap::Detail::ShapeInAccesses;

        static int const kWarpsContiguous =
                (WarpAccessIterations::kContiguous >= kWarpCount
                         ? kWarpCount
                         : WarpAccessIterations::kContiguous);

        static int const kWarpsStrided =
                (kWarpCount > WarpAccessIterations::kContiguous
                         ? kWarpCount / kWarpsContiguous
                         : 1);

        using WarpArrangement =
                layout::PitchLinearShape<kWarpsContiguous, kWarpsStrided>;
    };

    using Iterations =
            layout::PitchLinearShape<Detail::WarpAccessIterations::kContiguous /
                                             Detail::kWarpsContiguous,
                                     Detail::WarpAccessIterations::kStrided /
                                             Detail::kWarpsStrided>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    using Delta = typename BaseThreadMap::Delta;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        int warp_id = (thread_id / Detail::kWarpSize);
        int lane_id = (thread_id % Detail::kWarpSize);


        layout::PitchLinearCoord warp_footprint{
                Detail::WarpThreadArrangement::kContiguous *
                        Iterations::kContiguous,
                Detail::WarpThreadArrangement::kStrided * Iterations::kStrided};

        layout::PitchLinearCoord warp_offset{
                (warp_id % Detail::kWarpsContiguous),
                (warp_id / Detail::kWarpsContiguous)};

        layout::PitchLinearCoord thread_offset_in_warp{
                lane_id % Detail::WarpThreadArrangement::kContiguous,
                lane_id / Detail::WarpThreadArrangement::kContiguous};

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
                warp_footprint * warp_offset + thread_offset_in_warp;

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
                thread_offset_in_threadblock_tile_vec.contiguous() *
                        kElementsPerAccess,
                thread_offset_in_threadblock_tile_vec.strided()};

        return thread_offset_in_threadblock_tile_base;
    }
};



template <typename ThreadMap_, typename WarpThreadArrangement_>
struct TransposePitchLinearThreadMap {
    using ThreadMap = ThreadMap_;

    using TensorCoord = typename ThreadMap::TensorCoord;

    using Shape = typename ThreadMap::Shape;

    static int const kThreads = ThreadMap::kThreads;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

    struct Detail {
        using WarpThreadArrangement = WarpThreadArrangement_;

        static int const kWarpSize = WarpThreadArrangement::kCount;

        static int const kWarpCount = kThreads / kWarpSize;

        static_assert(!(Shape::kContiguous % kElementsPerAccess),
                      "Shape must be divisible by vector length.");

        using WarpArrangement =
                layout::PitchLinearShape<ThreadMap::Detail::kWarpsStrided,
                                         ThreadMap::Detail::kWarpsContiguous>;
    };

    using Iterations =
            layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
                                     ThreadMap::Iterations::kContiguous>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    using Delta = layout::PitchLinearShape<
            Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
            Detail::WarpThreadArrangement::kStrided>;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        int warp_id = (thread_id / Detail::kWarpSize);
        int lane_id = (thread_id % Detail::kWarpSize);


        layout::PitchLinearCoord warp_footprint{
                Detail::WarpThreadArrangement::kContiguous *
                        Iterations::kContiguous,
                Detail::WarpThreadArrangement::kStrided * Iterations::kStrided};

        layout::PitchLinearCoord warp_offset{
                (warp_id / Detail::WarpArrangement::kStrided),
                (warp_id % Detail::WarpArrangement::kStrided)};

        layout::PitchLinearCoord thread_offset_in_warp{
                lane_id % Detail::WarpThreadArrangement::kContiguous,
                lane_id / Detail::WarpThreadArrangement::kContiguous};

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
                warp_footprint * warp_offset + thread_offset_in_warp;

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
                thread_offset_in_threadblock_tile_vec.contiguous() *
                        kElementsPerAccess,
                thread_offset_in_threadblock_tile_vec.strided()};

        return thread_offset_in_threadblock_tile_base;
    }
};

template <typename ThreadMap_>
struct TransposePitchLinearThreadMapSimt {
    using ThreadMap = ThreadMap_;

    using TensorCoord = typename ThreadMap::TensorCoord;

    using Shape = typename ThreadMap::Shape;

    static int const kThreads = ThreadMap::kThreads;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static_assert(kElementsPerAccess == 1,
                  "Simt transpose requires elements per access to be 1");
    using Iterations =
            layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
                                     ThreadMap::Iterations::kContiguous>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    using ThreadAccessShape = typename ThreadMap::ThreadAccessShape;

    using Delta = layout::PitchLinearShape<ThreadMap::Delta::kStrided,
                                           ThreadMap::Delta::kContiguous>;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        TensorCoord coord = ThreadMap::initial_offset(thread_id);

        return TensorCoord(coord.strided(), coord.contiguous());
    }
};


template <typename Shape_,
          int Threads,
          typename WarpThreadArrangement_,
          int ElementsPerAccess =
                  1
          >
struct PitchLinearWarpStripedThreadMap {
    using TensorCoord = layout::PitchLinearCoord;

    using Shape = Shape_;

    static int const kThreads = Threads;

    static int const kElementsPerAccess = ElementsPerAccess;

    using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

    struct Detail {
        using WarpThreadArrangement = WarpThreadArrangement_;

        static int const kWarpSize = WarpThreadArrangement::kCount;

        static int const kWarpCount = kThreads / kWarpSize;

        static_assert(!(Shape::kContiguous % kElementsPerAccess),
                      "Shape must be divisible by vector length.");

        using ShapeInAccesses = layout::PitchLinearShape<
                Shape::kContiguous / kElementsPerAccess, Shape::kStrided>;

        using WarpAccessIterations = layout::PitchLinearShape<
                ShapeInAccesses::kContiguous /
                        WarpThreadArrangement::kContiguous,
                ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided>;

        static int const kWarpsStrided =
                (WarpAccessIterations::kStrided >= kWarpCount
                         ? kWarpCount
                         : (kWarpCount / WarpAccessIterations::kStrided));

        static int const kWarpsContiguous =
                (kWarpCount > WarpAccessIterations::kStrided
                         ? WarpAccessIterations::kContiguous / kWarpsStrided
                         : 1);

        using WarpArrangement =
                layout::PitchLinearShape<kWarpsContiguous, kWarpsStrided>;
    };

    using Iterations =
            layout::PitchLinearShape<Detail::WarpAccessIterations::kContiguous /
                                             Detail::kWarpsContiguous,
                                     Detail::WarpAccessIterations::kStrided /
                                             Detail::kWarpsStrided>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    using Delta = layout::PitchLinearShape<
            Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
            Detail::WarpThreadArrangement::kStrided *
                    Detail::WarpArrangement::kStrided>;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        int warp_id = (thread_id / Detail::kWarpSize);
        int lane_id = (thread_id % Detail::kWarpSize);


        layout::PitchLinearCoord warp_footprint{
                Detail::WarpThreadArrangement::kContiguous *
                        Iterations::kContiguous,
                Detail::WarpThreadArrangement::kStrided};

        layout::PitchLinearCoord warp_offset{
                (warp_id % Detail::kWarpsContiguous),
                (warp_id / Detail::kWarpsContiguous)};

        layout::PitchLinearCoord thread_offset_in_warp{
                lane_id % Detail::WarpThreadArrangement::kContiguous,
                lane_id / Detail::WarpThreadArrangement::kContiguous};

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
                warp_footprint * warp_offset + thread_offset_in_warp;

        layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
                thread_offset_in_threadblock_tile_vec.contiguous() *
                        kElementsPerAccess,
                thread_offset_in_threadblock_tile_vec.strided()};

        return thread_offset_in_threadblock_tile_base;
    }
};

template <typename Shape_, int Threads, typename ThreadTileShape>
struct PitchLinear2DThreadTileStripminedThreadMap;

template <typename Shape_, int Threads>
struct PitchLinear2DThreadTileStripminedThreadMap<
        Shape_, Threads, cutlass::layout::PitchLinearShape<4, 4>> {
    using TensorCoord = layout::PitchLinearCoord;

    using Shape = Shape_;

    using ThreadAccessShape = cutlass::layout::PitchLinearShape<4, 4>;

    static int const kThreads = Threads;

    static int const kElementsPerAccess = ThreadAccessShape::kContiguous;

    static_assert(!(kElementsPerAccess % 4),
                  "kElementsPerAccess, needs to be multiple of 4 (32bits)");

    struct Detail {
        static_assert(!(ThreadAccessShape::kContiguous % 4),
                      "ThreadAccessShape, needs to be multiple of 4");

        static_assert(!(Shape::kContiguous % ThreadAccessShape::kContiguous),
                      "");

        static_assert(
                !((Shape::kContiguous * Shape::kStrided) %
                  (kThreads * ThreadAccessShape::kCount)),
                "Shape must be divisible thread count * accesses per thread.");

        using ShapeVec = layout::PitchLinearShape<
                Shape::kContiguous / ThreadAccessShape::kContiguous,
                Shape::kStrided / ThreadAccessShape::kStrided>;

        static_assert((Threads < ShapeVec::kContiguous &&
                       !(ShapeVec::kContiguous % kThreads)) ||
                              (!(kThreads % ShapeVec::kContiguous) &&
                               !(ShapeVec::kStrided %
                                 (kThreads / ShapeVec::kContiguous))),
                      "Shape must be divisible by number of iterations of each "
                      "thread.");
    };

    using Iterations = typename platform::conditional<
            Threads >= Detail::ShapeVec::kContiguous,
            layout::PitchLinearShape<
                    1,
                    (Threads >= Detail::ShapeVec::kContiguous
                             ? Detail::ShapeVec::kStrided /
                                       (kThreads /
                                        Detail::ShapeVec::kContiguous)
                             : 0)>,
            layout::PitchLinearShape<Detail::ShapeVec::kContiguous / kThreads,
                                     Detail::ShapeVec::kStrided>>::type;

    using Delta = typename platform::conditional<
            Threads >= Detail::ShapeVec::kContiguous,
            layout::PitchLinearShape<Shape::kContiguous,
                                     kThreads * ThreadAccessShape::kStrided /
                                             Detail::ShapeVec::kContiguous>,
            layout::PitchLinearShape<kThreads * ThreadAccessShape::kContiguous,
                                     1>>::type;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        return TensorCoord((thread_id % Detail::ShapeVec::kContiguous) *
                                   ThreadAccessShape::kContiguous,
                           (thread_id / Detail::ShapeVec::kContiguous) *
                                   ThreadAccessShape::kStrided);
    }
};

template <typename ThreadMap_>
struct TransposePitchLinearThreadMap2DThreadTile {
    using ThreadMap = ThreadMap_;

    using TensorCoord = typename ThreadMap::TensorCoord;

    using Shape = typename ThreadMap::Shape;

    static int const kThreads = ThreadMap::kThreads;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static_assert(kElementsPerAccess > 1,
                  "Simt transpose requires elements per access to be 1");
    using Iterations =
            layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
                                     ThreadMap::Iterations::kContiguous>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    using ThreadAccessShape = typename ThreadMap::ThreadAccessShape;

    using Delta = layout::PitchLinearShape<ThreadMap::Delta::kStrided,
                                           ThreadMap::Delta::kContiguous>;

    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        TensorCoord coord = ThreadMap::initial_offset(thread_id);
        return TensorCoord(coord.strided(), coord.contiguous());
    }
};


}
}

