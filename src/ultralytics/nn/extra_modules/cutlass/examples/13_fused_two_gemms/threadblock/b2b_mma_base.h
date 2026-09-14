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
        typename Shape0_,
        typename Shape1_,
        typename Policy0_,
        typename Policy1_,
        int Stages,
        typename Enable = bool>
class B2bMmaBase {
public:
    using Shape0 = Shape0_;
    using Shape1 = Shape1_;

    using Policy0 = Policy0_;
    using Policy1 = Policy1_;


    using Operator0 = typename Policy0::Operator;
    using Operator1 = typename Policy1::Operator;

    using WarpGemm0 = typename Policy0::Operator::Shape;
    using WarpGemm1 = typename Policy1::Operator::Shape;

    using WarpCount0 =
            GemmShape<Shape0::kM / WarpGemm0::kM, Shape0::kN / WarpGemm0::kN,
                      Shape0::kK / WarpGemm0::kK>;
    using WarpCount1 =
            GemmShape<Shape1::kM / WarpGemm1::kM, Shape1::kN / WarpGemm1::kN,
                      Shape1::kK / WarpGemm1::kK>;

    static int const kWarpGemmIterations0 =
            (WarpGemm0::kK / Operator0::Policy::MmaShape::kK);
    static int const kWarpGemmIterations1 =
            (WarpGemm1::kK / Operator1::Policy::MmaShape::kK);

    static int const kStages = Stages;


    template <typename Shape_, typename Policy_>
    class SharedStorage {
    public:
        using Shape = Shape_;
        using Policy = Policy_;
        using Operator = typename Policy::Operator;

        using TensorRefA = TensorRef<typename Operator::ElementA,
                                     typename Operator::LayoutA>;

        using TensorRefB = TensorRef<typename Operator::ElementB,
                                     typename Operator::LayoutB>;

        using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                                   Shape::kK * kStages +
                                           Policy::SmemPaddingA::kColumn>;

        using ShapeB =
                MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                            Shape::kN + Policy::SmemPaddingB::kColumn>;

    public:

        AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

        AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

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

    using SharedStorage0 = SharedStorage<Shape0, Policy0>;
    using SharedStorage1 = SharedStorage<Shape1, Policy1>;
    union B2bMmaSharedStorage {
        SharedStorage0 sharedStorage0;
        SharedStorage1 sharedStorage1;
    };

protected:

    typename Operator0::IteratorA warp_tile_iterator_A0_;

    typename Operator0::IteratorB warp_tile_iterator_B0_;

    typename Operator1::IteratorB warp_tile_iterator_B1_;

public:
    CUTLASS_DEVICE
    B2bMmaBase(
            B2bMmaSharedStorage& shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx)
            : warp_tile_iterator_A0_(
                      shared_storage.sharedStorage0.operand_A_ref(), lane_idx),
              warp_tile_iterator_B0_(
                      shared_storage.sharedStorage0.operand_B_ref(), lane_idx),
              warp_tile_iterator_B1_(
                      shared_storage.sharedStorage1.operand_B_ref(), lane_idx) {

    }
};


}
}
}

