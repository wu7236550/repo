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

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"


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
                typename threadblock::GemmIdentityThreadblockSwizzle<>,
        int Stages = DefaultGemmConfiguration<OperatorClass_, ArchTag_,
                                              ElementA_, ElementB_, ElementC_,
                                              ElementAccumulator_>::kStages,
        int AlignmentA = DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::kAlignmentA,
        int AlignmentB = DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::kAlignmentB,
        bool SplitKSerial = false,
        typename Operator_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::Operator,
        bool IsBetaZero = false>
class Gemm {
public:
    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    using TensorRefA = TensorRef<ElementA const, LayoutA>;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    using TensorRefB = TensorRef<ElementB const, LayoutB>;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
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
    static int const kAlignmentC = EpilogueOutputOp::kCount;
    static bool const kSplitKSerial = SplitKSerial;
    static bool const kIsBetaZero = IsBetaZero;
    static ComplexTransform const kTransformA = ComplexTransform::kNone;
    static ComplexTransform const kTransformB = ComplexTransform::kNone;

    using GemmKernel = typename kernel::DefaultGemm<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
            ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
            ThreadblockSwizzle, kStages, kSplitKSerial, Operator,
            kIsBetaZero>::GemmKernel;

    struct Arguments {

        GemmCoord problem_size;
        TensorRef<ElementA const, LayoutA> ref_A;
        TensorRef<ElementB const, LayoutB> ref_B;
        TensorRef<ElementC const, LayoutC> ref_C;
        TensorRef<ElementC, LayoutC> ref_D;
        typename EpilogueOutputOp::Params epilogue;
        int split_k_slices;


        CUTLASS_HOST_DEVICE
        Arguments() : problem_size(0, 0, 0), split_k_slices(1) {}

        CUTLASS_HOST_DEVICE
        Arguments(GemmCoord problem_size_,
                  TensorRef<ElementA const, LayoutA> ref_A_,
                  TensorRef<ElementB const, LayoutB> ref_B_,
                  TensorRef<ElementC const, LayoutC> ref_C_,
                  TensorRef<ElementC, LayoutC> ref_D_,
                  typename EpilogueOutputOp::Params epilogue_ =
                          typename EpilogueOutputOp::Params(),
                  int split_k_slices = 1)
                : problem_size(problem_size_),
                  ref_A(ref_A_),
                  ref_B(ref_B_),
                  ref_C(ref_C_),
                  ref_D(ref_D_),
                  epilogue(epilogue_),
                  split_k_slices(split_k_slices) {}
    };

private:
    typename GemmKernel::Params params_;

public:
    Gemm() {}

    static Status can_implement(Arguments const& args) {
        if (!kSplitKSerial && args.split_k_slices > 1) {
            return Status::kErrorInvalidProblem;
        }

        Status status = GemmKernel::can_implement(
                args.problem_size, args.ref_A.non_const_ref(),
                args.ref_B.non_const_ref(), args.ref_C.non_const_ref(),
                args.ref_D);

        if (status != Status::kSuccess) {
            return status;
        }

        return Status::kSuccess;
    }

    static size_t get_workspace_size(Arguments const& args) {
        size_t bytes = 0;

        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord tiled_shape =
                threadblock_swizzle.get_tiled_shape(
                        args.problem_size,
                        {ThreadblockShape::kM, ThreadblockShape::kN,
                         ThreadblockShape::kK},
                        args.split_k_slices);

        if (kSplitKSerial && args.split_k_slices > 1) {
            bytes += sizeof(int) * size_t(tiled_shape.m()) *
                     size_t(tiled_shape.n());
        }

        return bytes;
    }

    Status initialize(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord grid_shape =
                threadblock_swizzle.get_tiled_shape(
                        args.problem_size,
                        {ThreadblockShape::kM, ThreadblockShape::kN,
                         ThreadblockShape::kK},
                        args.split_k_slices);

        if (kSplitKSerial) {
            if (args.split_k_slices > 1) {
                if (!workspace) {
                    return Status::kErrorWorkspaceNull;
                }

                size_t bytes = get_workspace_size(args);

                cudaError_t result =
                        cudaMemsetAsync(workspace, 0, bytes, stream);

                if (result != cudaSuccess) {
                    return Status::kErrorInternal;
                }
            }
        } else {
            if (args.split_k_slices > 1) {
                return Status::kErrorInvalidProblem;
            }
        }

        params_ = typename GemmKernel::Params{args.problem_size,
                                              grid_shape,
                                              args.ref_A.non_const_ref(),
                                              args.ref_B.non_const_ref(),
                                              args.ref_C.non_const_ref(),
                                              args.ref_D,
                                              args.epilogue,
                                              static_cast<int*>(workspace)};

        return Status::kSuccess;
    }

    Status update(Arguments const& args, void* workspace = nullptr) {
        if (kSplitKSerial && args.split_k_slices > 1) {
            if (!workspace) {
                return Status::kErrorWorkspaceNull;
            }
        }

        params_.ref_A.reset(args.ref_A.non_const_ref().data());
        params_.ref_B.reset(args.ref_B.non_const_ref().data());
        params_.ref_C.reset(args.ref_C.non_const_ref().data());
        params_.ref_D.reset(args.ref_D.data());
        params_.output_op = args.epilogue;
        params_.semaphore = static_cast<int*>(workspace);

        return Status::kSuccess;
    }

    Status run(cudaStream_t stream = nullptr) {
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid =
                threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(GemmKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
        if (smem_size >= (48 << 10)) {
            result = cudaFuncSetAttribute(
                    Kernel<GemmKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(
                    Kernel<GemmKernel>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }
        }

        cutlass::Kernel<GemmKernel>
                <<<grid, block, smem_size, stream>>>(params_);

        result = cudaGetLastError();

        return result == cudaSuccess ? Status::kSuccess
                                     : Status::kErrorInternal;
    }

    Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

    Status operator()(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        Status status = initialize(args, workspace);

        if (status == Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
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
        bool SplitKSerial,
        typename Operator_,
        bool IsBetaZero>
class Gemm<ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
           layout::ColumnMajor,
           ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
           WarpShape_, InstructionShape_, EpilogueOutputOp_,
           ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial,
           Operator_, IsBetaZero> {
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
    static ComplexTransform const kTransformA = ComplexTransform::kNone;
    static ComplexTransform const kTransformB = ComplexTransform::kNone;
    static bool const kSplitKSerial = SplitKSerial;
    static bool const kIsBetaZero = IsBetaZero;

    using UnderlyingOperator =
            Gemm<ElementB, typename layout::LayoutTranspose<LayoutB>::type,
                 ElementA, typename layout::LayoutTranspose<LayoutA>::type,
                 ElementC, layout::RowMajor, ElementAccumulator, OperatorClass,
                 ArchTag, ThreadblockShape, WarpShape, InstructionShape,
                 EpilogueOutputOp, ThreadblockSwizzle, Stages, kAlignmentB,
                 kAlignmentA, SplitKSerial, Operator, kIsBetaZero>;

    using UnderlyingArguments = typename UnderlyingOperator::Arguments;
    using GemmKernel = typename UnderlyingOperator::GemmKernel;
    static int const kAlignmentC = UnderlyingOperator::kAlignmentC;

    struct Arguments {

        GemmCoord problem_size;
        TensorRef<ElementA const, LayoutA> ref_A;
        TensorRef<ElementB const, LayoutB> ref_B;
        TensorRef<ElementC const, LayoutC> ref_C;
        TensorRef<ElementC, LayoutC> ref_D;
        typename EpilogueOutputOp::Params epilogue;
        int split_k_slices;


        CUTLASS_HOST_DEVICE
        Arguments() {}

        CUTLASS_HOST_DEVICE
        Arguments(GemmCoord problem_size_,
                  TensorRef<ElementA const, LayoutA> ref_A_,
                  TensorRef<ElementB const, LayoutB> ref_B_,
                  TensorRef<ElementC const, LayoutC> ref_C_,
                  TensorRef<ElementC, LayoutC> ref_D_,
                  typename EpilogueOutputOp::Params epilogue_ =
                          typename EpilogueOutputOp::Params(),
                  int split_k_slices = 1)
                : problem_size(problem_size_),
                  ref_A(ref_A_),
                  ref_B(ref_B_),
                  ref_C(ref_C_),
                  ref_D(ref_D_),
                  epilogue(epilogue_),
                  split_k_slices(split_k_slices) {}
    };

private:
    UnderlyingOperator underlying_operator_;

public:
    Gemm() {}

    static UnderlyingArguments to_underlying_arguments(Arguments const& args) {
        return UnderlyingArguments(
                {args.problem_size.n(), args.problem_size.m(),
                 args.problem_size.k()},
                {args.ref_B.data(), args.ref_B.stride(0)},
                {args.ref_A.data(), args.ref_A.stride(0)},
                {args.ref_C.data(), args.ref_C.stride(0)},
                {args.ref_D.data(), args.ref_D.stride(0)}, args.epilogue,
                args.split_k_slices);
    }

    static Status can_implement(Arguments const& args) {
        return UnderlyingOperator::can_implement(to_underlying_arguments(args));
    }

    static size_t get_workspace_size(Arguments const& args) {
        return UnderlyingOperator::get_workspace_size(
                to_underlying_arguments(args));
    }

    Status initialize(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        return underlying_operator_.initialize(to_underlying_arguments(args),
                                               workspace);
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
        Status status = initialize(args, workspace);

        if (status == Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
};


}
}
}

