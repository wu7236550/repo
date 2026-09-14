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

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"

#include "kernel/b2b_gemm.h"
#include "kernel/default_b2b_gemm.h"


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
        typename ThreadblockShape0_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::ThreadblockShape,
        typename ThreadblockShape1_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::ThreadblockShape,
        typename WarpShape0_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::WarpShape,
        typename WarpShape1_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::WarpShape,
        typename InstructionShape_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::InstructionShape,
        typename EpilogueOutputOp0_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::EpilogueOutputOp,
        typename EpilogueOutputOp1_ = typename DefaultGemmConfiguration<
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
        bool SplitKSerial = false,
        typename Operator_ = typename DefaultGemmConfiguration<
                OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                ElementAccumulator_>::Operator,
        bool IsBetaZero = false>
class B2bGemm {
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
    using ThreadblockShape0 = ThreadblockShape0_;
    using ThreadblockShape1 = ThreadblockShape1_;
    using WarpShape0 = WarpShape0_;
    using WarpShape1 = WarpShape1_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp0 = EpilogueOutputOp0_;
    using EpilogueOutputOp1 = EpilogueOutputOp1_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    static int const kStages = Stages;
    static int const kAlignmentA = AlignmentA;
    static int const kAlignmentB = AlignmentB;
    static int const kAlignmentC = EpilogueOutputOp1::kCount;
    static bool const kSplitKSerial = SplitKSerial;
    static bool const kIsBetaZero = IsBetaZero;
    static ComplexTransform const kTransformA = ComplexTransform::kNone;
    static ComplexTransform const kTransformB = ComplexTransform::kNone;

    using B2bGemmKernel = typename kernel::DefaultB2bGemm<
            ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
            ElementC, LayoutC, ElementAccumulator, OperatorClass, ArchTag,
            ThreadblockShape0, ThreadblockShape1, WarpShape0, WarpShape1,
            InstructionShape, EpilogueOutputOp0, EpilogueOutputOp1,
            ThreadblockSwizzle, kStages, kSplitKSerial, Operator,
            kIsBetaZero>::B2bGemmKernel;

    struct Arguments {

        GemmCoord problem_size_0;
        GemmCoord problem_size_1;
        TensorRef<ElementA const, LayoutA> ref_A0;
        TensorRef<ElementB const, LayoutB> ref_B0;
        TensorRef<ElementC const, LayoutC> ref_C0;
        TensorRef<ElementB const, LayoutB> ref_B1;
        TensorRef<ElementC const, LayoutC> ref_C1;
        TensorRef<ElementC, LayoutC> ref_D1;
        typename EpilogueOutputOp0::Params epilogue0;
        typename EpilogueOutputOp1::Params epilogue1;
        int split_k_slices;


        CUTLASS_HOST_DEVICE
        Arguments()
                : problem_size_0(0, 0, 0),
                  problem_size_1(0, 0, 0),
                  split_k_slices(1) {}

        CUTLASS_HOST_DEVICE
        Arguments(GemmCoord problem_size_0_, GemmCoord problem_size_1_,
                  TensorRef<ElementA const, LayoutA> ref_A0_,
                  TensorRef<ElementB const, LayoutB> ref_B0_,
                  TensorRef<ElementC const, LayoutC> ref_C0_,
                  TensorRef<ElementB const, LayoutB> ref_B1_,
                  TensorRef<ElementC const, LayoutC> ref_C1_,
                  TensorRef<ElementC, LayoutC> ref_D1_,
                  typename EpilogueOutputOp0::Params epilogue0_ =
                          typename EpilogueOutputOp0::Params(),
                  typename EpilogueOutputOp1::Params epilogue1_ =
                          typename EpilogueOutputOp1::Params(),
                  int split_k_slices_ = 1)
                : problem_size_0(problem_size_0_),
                  problem_size_1(problem_size_1_),
                  ref_A0(ref_A0_),
                  ref_B0(ref_B0_),
                  ref_C0(ref_C0_),
                  ref_B1(ref_B1_),
                  ref_C1(ref_C1_),
                  ref_D1(ref_D1_),
                  epilogue0(epilogue0_),
                  epilogue1(epilogue1_),
                  split_k_slices(split_k_slices_) {}
    };

private:
    typename B2bGemmKernel::Params params_;

public:
    B2bGemm() {}

    static Status can_implement(Arguments const& args) {
        if (!kSplitKSerial && args.split_k_slices > 1) {
            return Status::kErrorInvalidProblem;
        }

        Status status = B2bGemmKernel::can_implement(
                args.problem_size_0, args.problem_size_1,
                args.ref_A0.non_const_ref(), args.ref_B0.non_const_ref(),
                args.ref_C0.non_const_ref(), args.ref_B1.non_const_ref(),
                args.ref_C1.non_const_ref(), args.ref_D1);

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
                        args.problem_size_0,
                        {ThreadblockShape0::kM, ThreadblockShape0::kN,
                         ThreadblockShape0::kK},
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
                        args.problem_size_0,
                        {ThreadblockShape0::kM, ThreadblockShape0::kN,
                         ThreadblockShape0::kK},
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

        params_ = typename B2bGemmKernel::Params{
                args.problem_size_0,
                args.problem_size_1,
                grid_shape,
                args.ref_A0.non_const_ref(),
                args.ref_B0.non_const_ref(),
                args.ref_C0.non_const_ref(),
                args.ref_B1.non_const_ref(),
                args.ref_C1.non_const_ref(),
                args.ref_D1,
                args.epilogue0,
                args.epilogue1,
                static_cast<int*>(workspace),
        };

        return Status::kSuccess;
    }

    Status update(Arguments const& args, void* workspace = nullptr) {
        if (kSplitKSerial && args.split_k_slices > 1) {
            if (!workspace) {
                return Status::kErrorWorkspaceNull;
            }
        }

        params_.ref_A0.reset(args.ref_A.non_const_ref().data());
        params_.ref_B0.reset(args.ref_B.non_const_ref().data());
        params_.ref_C0.reset(args.ref_C.non_const_ref().data());
        params_.ref_B1.reset(args.ref_B.non_const_ref().data());
        params_.ref_C1.reset(args.ref_C.non_const_ref().data());
        params_.ref_D1.reset(args.ref_D.data());
        params_.output_op_0 = args.epilogue0;
        params_.output_op_1 = args.epilogue1;
        params_.semaphore = static_cast<int*>(workspace);

        return Status::kSuccess;
    }

    Status run(cudaStream_t stream = nullptr) {
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid =
                threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(B2bGemmKernel::kThreadCount, 1, 1);

        cudaError_t result;

        int smem_size = int(sizeof(typename B2bGemmKernel::SharedStorage));
        if (smem_size >= (48 << 10)) {
            result = cudaFuncSetAttribute(
                    Kernel<B2bGemmKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }

            result = cudaFuncSetAttribute(
                    Kernel<B2bGemmKernel>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }
        }

        cutlass::Kernel<B2bGemmKernel>
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

}
}
}

