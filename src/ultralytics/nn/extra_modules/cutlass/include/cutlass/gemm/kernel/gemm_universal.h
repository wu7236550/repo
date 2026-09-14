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
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/trace.h"


namespace cutlass {
namespace gemm {
namespace kernel {


template <typename Mma_,
          typename Epilogue_,
          typename ThreadblockSwizzle_
          >
struct GemmUniversal {
public:
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;

    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;

    static ComplexTransform const kTransformA = Mma::kTransformA;
    static ComplexTransform const kTransformB = Mma::kTransformB;
    using Operator = typename Mma::Operator;

    using OperatorClass = typename Mma::Operator::OperatorClass;
    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename Mma::Operator::Shape;
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    using ArchTag = typename Mma::ArchTag;

    static int const kStages = Mma::kStages;
    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC =
            Epilogue::OutputTileIterator::kElementsPerAccess;

    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    static int const kSplitKAlignment =
            const_max(128 / sizeof_bits<ElementA>::value,
                      128 / sizeof_bits<ElementB>::value);


    struct Arguments {

        GemmUniversalMode mode;
        GemmCoord problem_size;
        int batch_count;

        typename EpilogueOutputOp::Params epilogue;

        void const* ptr_A;
        void const* ptr_B;
        void const* ptr_C;
        void* ptr_D;

        int64_t batch_stride_A;
        int64_t batch_stride_B;
        int64_t batch_stride_C;
        int64_t batch_stride_D;

        int lda;
        int ldb;
        int ldc;
        int ldd;


        Arguments()
                : mode(GemmUniversalMode::kGemm),
                  batch_count(1),
                  ptr_A(nullptr),
                  ptr_B(nullptr),
                  ptr_C(nullptr),
                  ptr_D(nullptr) {}

        Arguments(GemmUniversalMode mode, GemmCoord problem_size,
                  int batch_count, typename EpilogueOutputOp::Params epilogue,
                  void const* ptr_A, void const* ptr_B, void const* ptr_C,
                  void* ptr_D, int64_t batch_stride_A, int64_t batch_stride_B,
                  int64_t batch_stride_C, int64_t batch_stride_D, int lda,
                  int ldb, int ldc, int ldd)
                : mode(mode),
                  problem_size(problem_size),
                  batch_count(batch_count),
                  epilogue(epilogue),
                  ptr_A(ptr_A),
                  ptr_B(ptr_B),
                  ptr_C(ptr_C),
                  ptr_D(ptr_D),
                  batch_stride_A(batch_stride_A),
                  batch_stride_B(batch_stride_B),
                  batch_stride_C(batch_stride_C),
                  batch_stride_D(batch_stride_D),
                  lda(lda),
                  ldb(ldb),
                  ldc(ldc),
                  ldd(ldd) {
            CUTLASS_TRACE_HOST(
                    "GemmUniversal::Arguments::Arguments() - problem_size: "
                    << problem_size);
        }

        Arguments transposed_problem() const {
            Arguments args(*this);

            std::swap(args.problem_size.m(), args.problem_size.n());
            std::swap(args.ptr_A, args.ptr_B);
            std::swap(args.lda, args.ldb);
            std::swap(args.batch_stride_A, args.batch_stride_B);

            return args;
        }
    };


    struct Params {
        cutlass::gemm::GemmCoord problem_size;
        cutlass::gemm::GemmCoord grid_tiled_shape;

        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorB::Params params_B;
        typename Epilogue::OutputTileIterator::Params params_C;
        typename Epilogue::OutputTileIterator::Params params_D;

        typename EpilogueOutputOp::Params output_op;

        GemmUniversalMode mode;
        int batch_count;
        int gemm_k_size;

        void* ptr_A;
        void* ptr_B;
        void* ptr_C;
        void* ptr_D;

        int64_t batch_stride_A;
        int64_t batch_stride_B;
        int64_t batch_stride_C;
        int64_t batch_stride_D;

        int* semaphore;


        CUTLASS_HOST_DEVICE
        Params()
                : params_A(0),
                  params_B(0),
                  params_C(0),
                  params_D(0),
                  batch_count(0),
                  gemm_k_size(0),
                  mode(cutlass::gemm::GemmUniversalMode::kGemm),
                  ptr_A(nullptr),
                  ptr_B(nullptr),
                  ptr_C(nullptr),
                  ptr_D(nullptr),
                  batch_stride_A(0),
                  batch_stride_B(0),
                  batch_stride_C(0),
                  batch_stride_D(0),
                  semaphore(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(Arguments const& args,
               cutlass::gemm::GemmCoord const& grid_tiled_shape,
               int gemm_k_size, void* workspace = nullptr)
                : problem_size(args.problem_size),
                  grid_tiled_shape(grid_tiled_shape),
                  params_A(args.lda),
                  params_B(args.ldb),
                  params_C(args.ldc),
                  params_D(args.ldd),
                  output_op(args.epilogue),
                  mode(args.mode),
                  batch_count(args.batch_count),
                  gemm_k_size(gemm_k_size),
                  ptr_A(const_cast<void*>(args.ptr_A)),
                  ptr_B(const_cast<void*>(args.ptr_B)),
                  ptr_C(const_cast<void*>(args.ptr_C)),
                  ptr_D(args.ptr_D),
                  batch_stride_A(args.batch_stride_A),
                  batch_stride_B(args.batch_stride_B),
                  batch_stride_C(args.batch_stride_C),
                  batch_stride_D(args.batch_stride_D),
                  semaphore(static_cast<int*>(workspace)) {
            CUTLASS_TRACE_HOST(
                    "GemmUniversal::Params::Params() - problem_size: "
                    << problem_size);
        }

        CUTLASS_HOST_DEVICE
        void update(Arguments const& args, void* workspace = nullptr) {
            ptr_A = const_cast<void*>(args.ptr_A);
            ptr_B = const_cast<void*>(args.ptr_B);
            ptr_C = const_cast<void*>(args.ptr_C);
            ptr_D = args.ptr_D;

            batch_stride_A = args.batch_stride_A;
            batch_stride_B = args.batch_stride_B;
            batch_stride_C = args.batch_stride_C;
            batch_stride_D = args.batch_stride_D;

            output_op = args.epilogue;

            semaphore = static_cast<int*>(workspace);

            CUTLASS_TRACE_HOST("GemmUniversal::Params::update()");
        }
    };

    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

public:

    CUTLASS_DEVICE
    GemmUniversal() {}

    static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) {
        CUTLASS_TRACE_HOST("GemmUniversal::can_implement()");

        static int const kAlignmentA =
                (platform::is_same<typename Mma::IteratorA::Layout,
                                   layout::ColumnMajorInterleaved<32>>::value)
                        ? 32
                        : (platform::is_same<
                                  typename Mma::IteratorA::Layout,
                                  layout::ColumnMajorInterleaved<64>>::value)
                                  ? 64
                                  : Mma::IteratorA::AccessType::kElements;
        static int const kAlignmentB =
                (platform::is_same<typename Mma::IteratorB::Layout,
                                   layout::RowMajorInterleaved<32>>::value)
                        ? 32
                        : (platform::is_same<
                                  typename Mma::IteratorB::Layout,
                                  layout::RowMajorInterleaved<64>>::value)
                                  ? 64
                                  : Mma::IteratorB::AccessType::kElements;
        static int const kAlignmentC =
                Epilogue::OutputTileIterator::kElementsPerAccess;

        if ((problem_size.m() % kAlignmentA) ||
            (problem_size.k() % kAlignmentA) ||
            (problem_size.n() % kAlignmentB) ||
            (problem_size.k() % kAlignmentB) ||
            (problem_size.m() % kAlignmentC) ||
            (problem_size.n() % kAlignmentC)) {
            CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand");
            return Status::kErrorMisalignedOperand;
        }

        CUTLASS_TRACE_HOST("  returning kSuccess");

        return Status::kSuccess;
    }

    static Status can_implement(Arguments const& args) {
        return can_implement(args.problem_size);
    }

    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage) {
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_offset =
                threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

        if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
            params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
            return;
        }

        int offset_k = 0;
        int problem_size_k = params.problem_size.k();

        ElementA* ptr_A = static_cast<ElementA*>(params.ptr_A);
        ElementB* ptr_B = static_cast<ElementB*>(params.ptr_B);

        if (params.mode == GemmUniversalMode::kGemm ||
            params.mode == GemmUniversalMode::kGemmSplitKParallel) {
            if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {
                problem_size_k =
                        (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
            }

            offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
        } else if (params.mode == GemmUniversalMode::kBatched) {
            ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
            ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
        } else if (params.mode == GemmUniversalMode::kArray) {
            ptr_A = static_cast<ElementA* const*>(
                    params.ptr_A)[threadblock_tile_offset.k()];
            ptr_B = static_cast<ElementB* const*>(
                    params.ptr_B)[threadblock_tile_offset.k()];
        }

        __syncthreads();

        cutlass::MatrixCoord tb_offset_A{
                threadblock_tile_offset.m() * Mma::Shape::kM,
                offset_k,
        };

        cutlass::MatrixCoord tb_offset_B{
                offset_k, threadblock_tile_offset.n() * Mma::Shape::kN};

        int thread_idx = threadIdx.x;

        typename Mma::IteratorA iterator_A(
                params.params_A, ptr_A,
                {params.problem_size.m(), problem_size_k}, thread_idx,
                tb_offset_A);

        typename Mma::IteratorB iterator_B(
                params.params_B, ptr_B,
                {problem_size_k, params.problem_size.n()}, thread_idx,
                tb_offset_B);

        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

        int lane_idx = threadIdx.x % 32;


        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        int gemm_k_iterations =
                (problem_size_k - offset_k + Mma::Shape::kK - 1) /
                Mma::Shape::kK;

        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
            accumulators);


        EpilogueOutputOp output_op(params.output_op);


        threadblock_tile_offset =
                threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

        MatrixCoord threadblock_offset(
                threadblock_tile_offset.m() * Mma::Shape::kM,
                threadblock_tile_offset.n() * Mma::Shape::kN);

        int block_idx =
                threadblock_tile_offset.m() +
                threadblock_tile_offset.n() * params.grid_tiled_shape.m();

        ElementC* ptr_C = static_cast<ElementC*>(params.ptr_C);
        ElementC* ptr_D = static_cast<ElementC*>(params.ptr_D);


        Semaphore semaphore(params.semaphore + block_idx, thread_idx);

        if (params.mode == GemmUniversalMode::kGemm) {
            if (params.grid_tiled_shape.k() > 1) {
                semaphore.fetch();

                output_op.set_k_partition(threadblock_tile_offset.k(),
                                          params.grid_tiled_shape.k());
            }
        } else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
            ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
        } else if (params.mode == GemmUniversalMode::kBatched) {
            ptr_C += threadblock_tile_offset.k() * params.batch_stride_C;
            ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
        } else if (params.mode == GemmUniversalMode::kArray) {
            ptr_C = static_cast<ElementC* const*>(
                    params.ptr_C)[threadblock_tile_offset.k()];
            ptr_D = static_cast<ElementC* const*>(
                    params.ptr_D)[threadblock_tile_offset.k()];
        }

        typename Epilogue::OutputTileIterator iterator_C(
                params.params_C, ptr_C, params.problem_size.mn(), thread_idx,
                threadblock_offset);

        typename Epilogue::OutputTileIterator iterator_D(
                params.params_D, ptr_D, params.problem_size.mn(), thread_idx,
                threadblock_offset);

        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);

        if (params.mode == GemmUniversalMode::kGemm &&
            params.grid_tiled_shape.k() > 1) {
            if (threadblock_tile_offset.k()) {
                iterator_C = iterator_D;
            }

            semaphore.wait(threadblock_tile_offset.k());

            __threadfence();
        }

        epilogue(output_op, iterator_D, accumulators, iterator_C);


        if (params.mode == GemmUniversalMode::kGemm &&
            params.grid_tiled_shape.k() > 1) {
            int lock = 0;
            if (params.grid_tiled_shape.k() ==
                threadblock_tile_offset.k() + 1) {
                lock = 0;
            } else {
                lock = threadblock_tile_offset.k() + 1;
            }

            semaphore.release(lock);
        }
    }
};


}
}
}

