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
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_universal_base.h"


namespace cutlass {
namespace gemm {
namespace device {


template <
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_,
        typename ElementAccumulator_ = ElementC_,
        typename OperatorClass_ = arch::OpClassSimt,
        typename ArchTag_ = arch::Sm70,
        typename ThreadblockShape_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::ThreadblockShape,
        typename WarpShape_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::WarpShape,
        typename InstructionShape_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::InstructionShape,
        typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::EpilogueOutputOp,
        typename ThreadblockSwizzle_ =
                threadblock::GemmIdentityThreadblockSwizzle<>,
        int Stages = DefaultGemmConfiguration<OperatorClass_, ArchTag_,
                                              ElementA_, ElementB_, ElementC_,
                                              ElementAccumulator_>::kStages,
        int AlignmentA = DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::kAlignmentA,
        int AlignmentB = DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::kAlignmentB,
        typename Operator_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::Operator,
        ComplexTransform TransformA = ComplexTransform::kNone,
        ComplexTransform TransformB = ComplexTransform::kNone>
class GemmUniversal
        : GemmUniversalBase<typename kernel::DefaultGemmUniversal<
                  ElementA_, LayoutA_, TransformA, AlignmentA, ElementB_,
                  LayoutB_, TransformB, AlignmentB, ElementC_, LayoutC_,
                  ElementAccumulator_, OperatorClass_, ArchTag_,
                  ThreadblockShape_, WarpShape_, InstructionShape_,
                  EpilogueOutputOp_, ThreadblockSwizzle_, Stages,
                  Operator_>::GemmKernel> {
public:
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    static int const kStages = Stages;
    static int const kAlignmentA = AlignmentA;
    static int const kAlignmentB = AlignmentB;
    static int const kAlignmentC = EpilogueOutputOp::kCount;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;

    using Base = GemmUniversalBase<typename kernel::DefaultGemmUniversal<
            ElementA_, LayoutA_, TransformA, AlignmentA, ElementB_, LayoutB_,
            TransformB, AlignmentB, ElementC_, LayoutC_, ElementAccumulator_,
            OperatorClass_, ArchTag_, ThreadblockShape_, WarpShape_,
            InstructionShape_, EpilogueOutputOp_, ThreadblockSwizzle_, Stages,
            Operator_>::GemmKernel>;

    using Arguments = typename Base::Arguments;
    using GemmKernel = typename Base::GemmKernel;
};


template <
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename ElementAccumulator_,
        typename OperatorClass_,
        typename ArchTag_,
        typename ThreadblockShape_,
        typename WarpShape_,
        typename InstructionShape_,
        typename EpilogueOutputOp_,
        typename ThreadblockSwizzle_,
        int Stages,
        int AlignmentA,
        int AlignmentB,
        typename Operator_,
        ComplexTransform TransformA,
        ComplexTransform TransformB>
class GemmUniversal<ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
                    layout::ColumnMajor,
                    ElementAccumulator_, OperatorClass_, ArchTag_,
                    ThreadblockShape_, WarpShape_, InstructionShape_,
                    EpilogueOutputOp_, ThreadblockSwizzle_, Stages, AlignmentA,
                    AlignmentB, Operator_, TransformA, TransformB> {
public:
    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    using TensorRefA = TensorRef<ElementA const, LayoutA>;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    using TensorRefB = TensorRef<ElementB const, LayoutB>;
    using ElementC = ElementC_;
    using LayoutC = layout::ColumnMajor;
    using TensorRefC = TensorRef<ElementC const, LayoutC>;
    using TensorRefD = TensorRef<ElementC, LayoutC>;
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    static int const kStages = Stages;
    static int const kAlignmentA = AlignmentA;
    static int const kAlignmentB = AlignmentB;
    static ComplexTransform const kTransformA = TransformA;
    static ComplexTransform const kTransformB = TransformB;

    using UnderlyingOperator = typename GemmUniversal<
            ElementB, typename layout::LayoutTranspose<LayoutB>::type, ElementA,
            typename layout::LayoutTranspose<LayoutA>::type, ElementC,
            layout::RowMajor, ElementAccumulator, OperatorClass, ArchTag,
            ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
            ThreadblockSwizzle, Stages, kAlignmentB, kAlignmentA, Operator,
            kTransformB, kTransformA>::Base;

    using GemmKernel = typename UnderlyingOperator::GemmKernel;
    static int const kAlignmentC = EpilogueOutputOp::kCount;

    using Arguments = typename UnderlyingOperator::Arguments;

private:
    UnderlyingOperator underlying_operator_;

public:
    GemmUniversal() {}

    static Arguments to_underlying_arguments(Arguments const& args) {
        return args.transposed_problem();
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

