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

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"


namespace cutlass {
namespace gemm {
namespace kernel {


template <typename Mma_,
          typename Epilogue_,
          typename ThreadblockSwizzle_,
          bool SplitKSerial
          >
struct Gemm {
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using OutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    static bool const kSplitKSerial = SplitKSerial;

    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    struct Params {
        cutlass::gemm::GemmCoord problem_size;
        cutlass::gemm::GemmCoord grid_tiled_shape;
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::Params params_B;
        typename Mma::IteratorB::TensorRef ref_B;
        typename Epilogue::OutputTileIterator::Params params_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        typename Epilogue::OutputTileIterator::Params params_D;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;
        typename OutputOp::Params output_op;
        int* semaphore;
        int gemm_k_iterations;
        int gemm_k_size;


        CUTLASS_HOST_DEVICE
        Params() : semaphore(0), gemm_k_iterations(0), gemm_k_size(0) {}

        CUTLASS_HOST_DEVICE
        Params(cutlass::gemm::GemmCoord const& problem_size,
               cutlass::gemm::GemmCoord const& grid_tiled_shape,
               typename Mma::IteratorA::TensorRef ref_A,
               typename Mma::IteratorB::TensorRef ref_B,
               typename Epilogue::OutputTileIterator::TensorRef ref_C,
               typename Epilogue::OutputTileIterator::TensorRef ref_D,
               typename OutputOp::Params output_op =
                       typename OutputOp::Params(),
               int* workspace = nullptr)
                : problem_size(problem_size),
                  grid_tiled_shape(grid_tiled_shape),
                  params_A(ref_A.layout()),
                  ref_A(ref_A),
                  params_B(ref_B.layout()),
                  ref_B(ref_B),
                  params_C(ref_C.layout()),
                  ref_C(ref_C),
                  params_D(ref_D.layout()),
                  ref_D(ref_D),
                  output_op(output_op) {
            int total_gemm_k_iterations =
                    (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
            int gemm_k_iterations =
                    (total_gemm_k_iterations + grid_tiled_shape.k() - 1) /
                    grid_tiled_shape.k();

            gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

            semaphore = workspace;
        }
    };

    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };


    CUTLASS_HOST_DEVICE
    Gemm() {}

    static Status can_implement(
            cutlass::gemm::GemmCoord const& problem_size,
            typename Mma::IteratorA::TensorRef ref_A,
            typename Mma::IteratorB::TensorRef ref_B,
            typename Epilogue::OutputTileIterator::TensorRef ref_C,
            typename Epilogue::OutputTileIterator::TensorRef ref_D) {
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

        if (!TensorRef_aligned(ref_A, kAlignmentA)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_B, kAlignmentB)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_C, kAlignmentC)) {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(ref_D, kAlignmentC)) {
            return Status::kErrorMisalignedOperand;
        }

        if ((problem_size.m() % kAlignmentA) ||
            (problem_size.k() % kAlignmentA) ||
            (problem_size.n() % kAlignmentB) ||
            (problem_size.k() % kAlignmentB) ||
            (problem_size.m() % kAlignmentC) ||
            (problem_size.n() % kAlignmentC)) {
            return Status::kErrorMisalignedOperand;
        }

        return Status::kSuccess;
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

        cutlass::MatrixCoord tb_offset_A{
                threadblock_tile_offset.m() * Mma::Shape::kM,
                threadblock_tile_offset.k() * params.gemm_k_size,
        };

        cutlass::MatrixCoord tb_offset_B{
                threadblock_tile_offset.k() * params.gemm_k_size,
                threadblock_tile_offset.n() * Mma::Shape::kN};

        int problem_size_k =
                min(params.problem_size.k(),
                    (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

        int gemm_k_iterations =
                (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) /
                Mma::Shape::kK;

        int thread_idx = threadIdx.x;

        typename Mma::IteratorA iterator_A(
                params.params_A, params.ref_A.data(),
                {params.problem_size.m(), problem_size_k}, thread_idx,
                tb_offset_A);

        typename Mma::IteratorB iterator_B(
                params.params_B, params.ref_B.data(),
                {problem_size_k, params.problem_size.n()}, thread_idx,
                tb_offset_B);

        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;


        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        if (!kSplitKSerial || gemm_k_iterations > 0) {
            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
                accumulators);
        }


        OutputOp output_op(params.output_op);


        threadblock_tile_offset =
                threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

        MatrixCoord threadblock_offset(
                threadblock_tile_offset.m() * Mma::Shape::kM,
                threadblock_tile_offset.n() * Mma::Shape::kN);

        int block_idx =
                threadblock_tile_offset.m() +
                threadblock_tile_offset.n() * params.grid_tiled_shape.m();

        Semaphore semaphore(params.semaphore + block_idx, thread_idx);

        if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
            semaphore.fetch();

            output_op.set_k_partition(threadblock_tile_offset.k(),
                                      params.grid_tiled_shape.k());
        }

        typename Epilogue::OutputTileIterator iterator_C(
                params.params_C, params.ref_C.data(), params.problem_size.mn(),
                thread_idx, threadblock_offset);

        typename Epilogue::OutputTileIterator iterator_D(
                params.params_D, params.ref_D.data(), params.problem_size.mn(),
                thread_idx, threadblock_offset);

        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);

        if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
            if (threadblock_tile_offset.k()) {
                iterator_C = iterator_D;
            }

            semaphore.wait(threadblock_tile_offset.k());

            __threadfence();
        }

        epilogue(output_op, iterator_D, accumulators, iterator_C);


        if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
            int lock = 0;
            if (params.grid_tiled_shape.k() ==
                threadblock_tile_offset.k() + 1) {
                lock = 0;
            } else {
                lock = threadblock_tile_offset.k() + 1;
            }

            __threadfence();
            semaphore.release(lock);
        }
    }
};


}
}
}
