/***************************************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"


namespace cutlass {
namespace gemm {
namespace threadblock {

namespace {
template <typename Op, typename T, int Threads>
struct Reduce;

template <typename T, int N, int Threads>
struct Reduce<plus<T>, Array<T, N>, Threads> {
    using AccessType = Array<T, N>;
    CUTLASS_DEVICE void operator()(int t, T* x, int stride = 1) const {
        int DIM_X = blockDim.x;
        AccessType* pointer = reinterpret_cast<AccessType*>(x);
        plus<Array<T, N>> _op;
        __syncthreads();
        CUTLASS_PRAGMA_UNROLL
        for (int i = (Threads >> 1); i >= 1; i >>= 1) {
            if (t < i) {
                pointer[t * stride] =
                        _op(pointer[t * stride], pointer[(t + i) * stride]);
            }
            if (i * DIM_X > 16)
                __syncthreads();
            else {
#if __CUDACC_VER_MAJOR__ >= 9
                __syncwarp();
#endif
            }
        }
    }
};
}


template <class Core_
          >
class Gemv {
public:
    using Shape = typename Core_::Shape;

    using Operator = typename Core_::Operator;

    using IteratorA = typename Core_::IteratorA;

    using IteratorB = typename Core_::IteratorB;

    using IteratorC = typename Core_::IteratorC;

    using FragmentA = typename IteratorA::Fragment;

    using FragmentB = typename IteratorB::Fragment;

    using FragmentC = typename Operator::FragmentC;

    using ThreadShape = typename Core_::ThreadShape;

public:
    CUTLASS_DEVICE
    Gemv() {}

    CUTLASS_DEVICE
    void operator()(
            GemmCoord const& problem_size,
            FragmentC& accum,
            IteratorA iterator_A,
            IteratorB iterator_B,
            FragmentC const& src_accum) {


        FragmentA frag_A;
        FragmentB frag_B;
        frag_A.clear();
        frag_B.clear();

        iterator_A.load(frag_A);
        iterator_B.load(frag_B);
        ++iterator_A;
        ++iterator_B;

        Operator thread_mma;
        int gemm_k = problem_size.k();

        CUTLASS_GEMM_LOOP
        for (; gemm_k > 0; gemm_k -= Shape::kK) {
            thread_mma(accum, frag_A, frag_B, accum);

            if (gemm_k > Shape::kK) {
                iterator_A.load(frag_A);
                iterator_B.load(frag_B);
                ++iterator_A;
                ++iterator_B;
            }
        }
    }
};


template <class Core_
          >
class GemvBatchedReduction {
public:
    using Shape = typename Core_::Shape;

    using Operator = typename Core_::Operator;

    using IteratorA = typename Core_::IteratorA;

    using IteratorB = typename Core_::IteratorB;

    using IteratorC = typename Core_::IteratorC;

    using FragmentA = typename IteratorA::Fragment;

    using FragmentB = typename IteratorB::Fragment;

    using FragmentC = typename Operator::FragmentC;

    using ThreadShape = typename Core_::ThreadShape;

    using TensorRefAccumulator =
            TensorRef<typename Core_::ElementC, typename Core_::LayoutC>;

    using ReductionThreadMap = cutlass::transform::
            PitchLinear2DTilePolicyStripminedThreadContiguous<
                    layout::PitchLinearShape<Shape::kN,
                                             Shape::kK / ThreadShape::kK>,
                    typename Core_::ThreadArrangement, ThreadShape::kN>;

    using SmemTileIteratorAccumulator =
            cutlass::transform::threadblock::RegularTileIterator<
                    MatrixShape<Shape::kK / ThreadShape::kK, Shape::kN>,
                    typename Core_::ElementC, layout::RowMajor, 0,
                    ReductionThreadMap>;



    class SharedStorage {
    public:

        using ShapeAccumulator =
                MatrixShape<Shape::kK / ThreadShape::kK, Shape::kN>;

    public:

        AlignedBuffer<typename Core_::ElementC, ShapeAccumulator::kCount>
                buffer_accumulator;

    public:

        CUTLASS_HOST_DEVICE
        static typename Core_::LayoutC LayoutAccumulator() {
            return Core_::LayoutC::packed(
                    {ShapeAccumulator::kRow, ShapeAccumulator::kColumn});
        }

        CUTLASS_HOST_DEVICE
        TensorRefAccumulator buffer_ref() {
            return TensorRefAccumulator{buffer_accumulator.data(),
                                        LayoutAccumulator()};
        }
    };

public:
    CUTLASS_DEVICE
    GemvBatchedReduction() {}

    CUTLASS_DEVICE
    void operator()(
            GemmCoord const& problem_size,
            FragmentC& accum,
            IteratorA iterator_A,
            IteratorB iterator_B,
            FragmentC const& src_accum) {


        FragmentA frag_A;
        FragmentB frag_B;
        frag_A.clear();
        frag_B.clear();

        iterator_A.load(frag_A);
        iterator_B.load(frag_B);
        ++iterator_A;
        ++iterator_B;

        Operator thread_mma;
        int gemm_k = problem_size.k();

        CUTLASS_GEMM_LOOP
        for (; gemm_k > 0; gemm_k -= Shape::kK) {
            thread_mma(accum, frag_A, frag_B, accum);

            if (gemm_k > Shape::kK) {
                iterator_A.load(frag_A);
                iterator_B.load(frag_B);
                ++iterator_A;
                ++iterator_B;
            }
        }

        if (Shape::kK > ThreadShape::kK) {
            extern __shared__ int SharedStorageBase[];

            SharedStorage& shared_storage =
                    *(reinterpret_cast<SharedStorage*>(SharedStorageBase));
            SmemTileIteratorAccumulator smem_tile_iterator(
                    shared_storage.buffer_ref(), threadIdx.x);
            smem_tile_iterator.store(accum);

            using Reduction = Reduce<plus<typename Core_::ElementC>, FragmentC,
                                     Core_::kThreadsPerK>;
            Reduction reduce_op;
            auto stride = SharedStorage::LayoutAccumulator().stride(0) /
                          FragmentC::kElements;
            typename Core_::ElementC* pointer =
                    shared_storage.buffer_ref().data();
            reduce_op(threadIdx.y, &pointer[threadIdx.x * FragmentC::kElements],
                      stride);
            if (threadIdx.y == 0)
                smem_tile_iterator.load(accum);
        }
    }
};


}
}
}
