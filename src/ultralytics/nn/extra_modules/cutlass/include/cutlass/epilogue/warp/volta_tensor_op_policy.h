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

#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape,
          typename InterleavedTileShape,
          typename ElementC,
          typename Layout
          >
struct VoltaTensorOpPolicy;


template <typename WarpShape_
          >
struct VoltaTensorOpPolicy<WarpShape_, gemm::GemmShape<32, 32, 4>, half_t,
                           layout::RowMajor> {
    using WarpShape = WarpShape_;
    using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
    using ElementC = half_t;
    using Layout = layout::RowMajor;

    using InstructionShape = gemm::GemmShape<16, 16, 4>;

    using MmaIterations =
            MatrixShape<InterleavedTileShape::kM / InstructionShape::kM,
                        InterleavedTileShape::kN / InstructionShape::kN>;

    using TileIterations =
            MatrixShape<WarpShape::kM / InterleavedTileShape::kM,
                        WarpShape::kN / InterleavedTileShape::kN>;

    static int const kElementsPerMma = 8;
    static int const kRowsPerIteration = 16;


    static int const kElementsPerAccess = 4;

    static int const kAccessesPerInterleavedTile = 4;

    static int const kIterations = TileIterations::kRow * 2;


    using AccessType = AlignedArray<ElementC, kElementsPerAccess>;

    using Fragment =
            Array<ElementC, kElementsPerAccess * kAccessesPerInterleavedTile *
                                    TileIterations::kColumn>;

    using AccumulatorTile =
            Array<ElementC, TileIterations::kCount * MmaIterations::kCount *
                                    kElementsPerMma>;
};


template <typename WarpShape_
          >
struct VoltaTensorOpPolicy<WarpShape_, gemm::GemmShape<32, 32, 4>, float,
                           layout::RowMajor> {
    using WarpShape = WarpShape_;
    using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
    using ElementC = float;
    using Layout = layout::RowMajor;

    using InstructionShape = gemm::GemmShape<16, 16, 4>;

    using MmaIterations =
            MatrixShape<InterleavedTileShape::kM / InstructionShape::kM,
                        InterleavedTileShape::kN / InstructionShape::kN>;

    using TileIterations =
            MatrixShape<WarpShape::kM / InterleavedTileShape::kM,
                        WarpShape::kN / InterleavedTileShape::kN>;

    static int const kElementsPerMma = 8;
    static int const kRowsPerIteration = 16;


    static int const kElementsPerAccess = 2;

    static int const kAccessesPerInterleavedTile = 8;

    static int const kRowsPerMmaTile = 2;

    static int const kIterations = TileIterations::kRow * MmaIterations::kRow;


    using AccessType = AlignedArray<ElementC, kElementsPerAccess>;

    using Fragment =
            Array<ElementC, kElementsPerAccess * kAccessesPerInterleavedTile *
                                    TileIterations::kColumn>;

    using AccumulatorTile =
            Array<ElementC, TileIterations::kCount * MmaIterations::kCount *
                                    kElementsPerMma>;
};


}
}
}

