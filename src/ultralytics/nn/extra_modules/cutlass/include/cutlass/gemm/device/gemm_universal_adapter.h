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
#include "cutlass/gemm/device/gemm_universal_base.h"


namespace cutlass {
namespace gemm {
namespace device {


namespace detail {

template <typename ElementA_, typename LayoutA_, ComplexTransform TransformA,
          int AlignmentA, typename ElementB_, typename LayoutB_,
          ComplexTransform TransformB, int AlignmentB, typename LayoutC_,
          bool Transpose>
struct MapArguments {
    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    static ComplexTransform const kTransformA = TransformA;
    static int const kAlignmentA = AlignmentA;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    static ComplexTransform const kTransformB = TransformB;
    static int const kAlignmentB = AlignmentB;
    using LayoutC = LayoutC_;
};

template <typename ElementA_, typename LayoutA_, ComplexTransform TransformA,
          int AlignmentA, typename ElementB_, typename LayoutB_,
          ComplexTransform TransformB, int AlignmentB, typename LayoutC_>
struct MapArguments<ElementA_, LayoutA_, TransformA, AlignmentA, ElementB_,
                    LayoutB_, TransformB, AlignmentB, LayoutC_, true> {
    using ElementA = ElementB_;
    using LayoutA = typename layout::LayoutTranspose<LayoutB_>::type;
    static ComplexTransform const kTransformA = TransformB;
    static int const kAlignmentA = AlignmentB;
    using ElementB = ElementA_;
    using LayoutB = typename layout::LayoutTranspose<LayoutA_>::type;
    static ComplexTransform const kTransformB = TransformA;
    static int const kAlignmentB = AlignmentA;
    using LayoutC = typename layout::LayoutTranspose<LayoutC_>::type;
};

};


template <typename GemmKernel_>
class GemmUniversalAdapter {
public:
    using GemmKernel = GemmKernel_;

    static bool const kInternalTranspose =
            std::is_same<typename GemmKernel::LayoutC,
                         cutlass::layout::RowMajor>::value;

    using ThreadblockShape = typename GemmKernel::Mma::Shape;
    using WarpShape = typename GemmKernel::WarpShape;
    using InstructionShape = typename GemmKernel::InstructionShape;

    using WarpMmaOperator = typename GemmKernel::Mma::Policy::Operator;
    using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
    using MathOperator = typename ArchMmaOperator::Operator;

    using OperatorClass = typename WarpMmaOperator::OperatorClass;
    using ArchTag = typename WarpMmaOperator::ArchTag;

    using MapArguments = detail::MapArguments<
            typename GemmKernel::ElementA, typename GemmKernel::LayoutA,
            GemmKernel::kTransformA, GemmKernel::kAlignmentA,
            typename GemmKernel::ElementB, typename GemmKernel::LayoutB,
            GemmKernel::kTransformB, GemmKernel::kAlignmentB,
            typename GemmKernel::LayoutC, kInternalTranspose>;

    using ElementA = typename MapArguments::ElementA;
    using LayoutA = typename MapArguments::LayoutA;
    static ComplexTransform const kTransformA = MapArguments::kTransformA;
    static int const kAlignmentA = GemmKernel::kAlignmentA;

    using ElementB = typename MapArguments::ElementB;
    using LayoutB = typename MapArguments::LayoutB;
    static ComplexTransform const kTransformB = MapArguments::kTransformB;
    static int const kAlignmentB = GemmKernel::kAlignmentB;

    using ElementC = typename GemmKernel::ElementC;
    using LayoutC = typename MapArguments::LayoutC;
    static int const kAlignmentC = GemmKernel::kAlignmentC;

    using TensorRefA = TensorRef<ElementA const, LayoutA>;
    using TensorRefB = TensorRef<ElementB const, LayoutB>;
    using TensorRefC = TensorRef<ElementC const, LayoutC>;
    using TensorRefD = TensorRef<ElementC, LayoutC>;

    using ElementAccumulator =
            typename GemmKernel::Mma::Policy::Operator::ElementC;

    static int const kStages = GemmKernel::Mma::kStages;

    using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
    using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
    using Operator = typename GemmKernel::Operator;

    using UnderlyingOperator = GemmUniversalBase<GemmKernel>;
    using Arguments = typename UnderlyingOperator::Arguments;

private:
    UnderlyingOperator underlying_operator_;

public:
    GemmUniversalAdapter() {}

    static Arguments to_underlying_arguments(Arguments const& args) {
        if (kInternalTranspose) {
            return args.transposed_problem();
        } else {
            return args;
        }
    }

    static Status can_implement(Arguments const& args) {
        return UnderlyingOperator::can_implement(to_underlying_arguments(args));
    }

    static size_t get_workspace_size(Arguments const& args) {
        return UnderlyingOperator::get_workspace_size(
                to_underlying_arguments(args));
    }

    static dim3 get_grid_shape(Arguments const& args) {
        return UnderlyingOperator::get_grid_shape(
                to_underlying_arguments(args));
    }

    static int maximum_active_blocks(int smem_capacity = -1) {
        return UnderlyingOperator::maximum_active_blocks(smem_capacity);
    }

    Status initialize(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        return underlying_operator_.initialize(to_underlying_arguments(args),
                                               workspace, stream);
    }

    Status update(Arguments const& args, void* workspace = nullptr) {
        return underlying_operator_.update(to_underlying_arguments(args),
                                           workspace);
    }

    Status run(cudaStream_t stream = nullptr) {
        return underlying_operator_.run(stream);
    }

    Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

    Status operator()(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        Status status = initialize(args, workspace, stream);

        if (status == Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
};


}
}
}

