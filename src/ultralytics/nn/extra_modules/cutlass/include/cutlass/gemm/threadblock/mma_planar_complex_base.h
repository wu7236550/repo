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

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

namespace cutlass {
namespace gemm {
namespace threadblock {


template <
        typename Shape_,
        typename Policy_,
        int Stages,
        typename Enable = bool>
class MmaPlanarComplexBase {
public:
    using Shape = Shape_;

    using Policy = Policy_;


    using Operator = typename Policy::Operator;

    using WarpGemm = typename Policy::Operator::Shape;

    using WarpCount =
            GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN,
                      Shape::kK / WarpGemm::kK>;

    static int const kWarpGemmIterations =
            (WarpGemm::kK / Operator::Policy::MmaShape::kK);

    static int const kStages = Stages;

    using TensorRefA =
            TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

    using TensorRefB =
            TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;


    class SharedStorage {
    public:

        using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                                   Shape::kK * kStages +
                                           Policy::SmemPaddingA::kColumn>;

        static int const kImaginaryStrideA = ShapeA::kCount;

        using ShapeB =
                MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                            Shape::kN + Policy::SmemPaddingB::kColumn>;

        static int const kImaginaryStrideB = ShapeB::kCount;

    public:

        AlignedBuffer<typename Operator::ElementA,
                      ShapeA::kCount + kImaginaryStrideA>
                operand_A;

        AlignedBuffer<typename Operator::ElementB,
                      ShapeB::kCount + kImaginaryStrideB>
                operand_B;

    public:

        CUTLASS_DEVICE
        static typename Operator::LayoutA LayoutA() {
            return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
        }

        CUTLASS_HOST_DEVICE
        static typename Operator::LayoutB LayoutB() {
            return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
        }

        CUTLASS_HOST_DEVICE
        TensorRefA operand_A_ref() {
            return TensorRefA{operand_A.data(), LayoutA()};
        }

        CUTLASS_HOST_DEVICE
        TensorRefB operand_B_ref() {
            return TensorRefB{operand_B.data(), LayoutB()};
        }
    };

protected:

    typename Operator::IteratorA warp_tile_iterator_A_;

    typename Operator::IteratorB warp_tile_iterator_B_;

public:
    CUTLASS_DEVICE
    MmaPlanarComplexBase(
            SharedStorage& shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx)
            : warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
              warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx) {}
};


}
}
}

