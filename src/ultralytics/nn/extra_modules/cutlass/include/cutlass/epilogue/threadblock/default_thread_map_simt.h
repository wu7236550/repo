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

#include "predicated_tile_iterator.h"
#include "cutlass/gemm/gemm.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <typename ThreadblockShape_, typename WarpShape_,
          typename MmaSimtPolicy_, int PartitionsK, typename Element_,
          int ElementsPerAccess>
struct DefaultThreadMapSimt {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using MmaSimtPolicy = MmaSimtPolicy_;
    static int const kPartitionsK = PartitionsK;
    using Element = Element_;
    static int const kElementsPerAccess = ElementsPerAccess;


    struct Detail {
        static int const kWarpSize = 32;

        static_assert(!(ThreadblockShape::kM % WarpShape::kM) &&
                              !(ThreadblockShape::kN % WarpShape::kN),
                      "Divisibility");

        using WarpCount = gemm::GemmShape<ThreadblockShape::kM / WarpShape::kM,
                                          ThreadblockShape::kN / WarpShape::kN,
                                          kPartitionsK>;

        static int const kGroupCount =
                WarpShape::kM / (MmaSimtPolicy::WarpShape::kRow *
                                 MmaSimtPolicy::LaneMmaShape::kM);

        static int const kThreads = WarpCount::kCount * kWarpSize;

        static int const kIterations =
                MmaSimtPolicy::LaneMmaShape::kM * kGroupCount;
    };


    using Type = OutputTileOptimalThreadMap<
            OutputTileShape<
                    ThreadblockShape::kN, 1, MmaSimtPolicy::WarpShape::kRow,
                    Detail::WarpCount::kM, 1>,
            OutputTileShape<
                    1, MmaSimtPolicy::LaneMmaShape::kM, Detail::kGroupCount, 1,
                    Detail::kIterations>,
            Detail::kThreads, kElementsPerAccess, sizeof_bits<Element>::value>;
};


}
}
}

