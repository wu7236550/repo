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

/**
 * \file include/cutlass/epilogue/threadblock/convolution_epilogue.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/vector.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <typename Shape_,
          typename Layout_,
          int PartitionsK,
          typename WarpMmaOperator_,
          typename OutputTileIterator_,
          typename AccumulatorFragmentIterator_,
          typename WarpTileIterator_,
          typename SharedLoadIterator_,
          typename BiasTileIterator_,
          typename OutputOp_,
          typename Padding_,
          bool UseSyncWarp = false>
class ConvolutionEpilogue
        : public EpilogueBase<Shape_, typename WarpMmaOperator_::Shape,
                              PartitionsK, AccumulatorFragmentIterator_,
                              WarpTileIterator_, Padding_> {
public:
    using Base = EpilogueBase<Shape_, typename WarpMmaOperator_::Shape,
                              PartitionsK, AccumulatorFragmentIterator_,
                              WarpTileIterator_, Padding_>;

    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using WarpTileIterator = WarpTileIterator_;
    using SharedLoadIterator = SharedLoadIterator_;
    using BiasTileIterator = BiasTileIterator_;
    using OutputOp = OutputOp_;
    using Padding = Padding_;

    using Layout = Layout_;
    using LongIndex = typename Layout::LongIndex;

    using AccumulatorTile = typename Base::AccumulatorTile;

    using ElementAccumulator = typename WarpTileIterator::Element;

    using ElementOutput = typename OutputTileIterator::Element;

    using ElementBias = typename BiasTileIterator::Element;

    static int const kElementsPerAccess =
            OutputTileIterator::kElementsPerAccess;

    using TensorRef = typename OutputTileIterator::TensorRef;

    using SyncTensorRef =
            typename cutlass::TensorRef<int,
                                        cutlass::layout::PackedVectorLayout>;

    using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

    using OutputAccessType = Array<typename OutputTileIterator::Element,
                                   OutputTileIterator::kElementsPerAccess>;

    using AccumulatorAccessType = Array<typename WarpTileIterator::Element,
                                        OutputTileIterator::kElementsPerAccess>;

    using BiasAccessType = Array<typename BiasTileIterator::Element,
                                 OutputTileIterator::kElementsPerAccess>;

    using WarpCount = typename Base::WarpCount;

public:
    static_assert(
            SharedLoadIterator::Fragment::kElements ==
                    OutputTileIterator::Fragment::kElements,
            "Mismatch between shared load iterator and output tile iterator.");

    static_assert((!(OutputTileIterator::kElementsPerAccess % 4) ||
                   OutputTileIterator::kElementsPerAccess == 1),
                  "OutputTileIterator::kElementsPerAccess must be 1 or a "
                  "multiple of 4.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    OutputTileIterator::kElementsPerAccess),
                  "Divisibility");

    static_assert(kPartitionsK == 1,
                  "Split K algorithm for convolution not supported.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    BiasTileIterator::Fragment::kElements),
                  "Divisibility");

private:
    SharedLoadIterator shared_load_iterator_;

public:
    CUTLASS_DEVICE
    ConvolutionEpilogue(
            typename Base::SharedStorage&
                    shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx
            )
            : Base(shared_storage, thread_idx, warp_idx, lane_idx),
              shared_load_iterator_(shared_storage.reference(), thread_idx) {}

    CUTLASS_DEVICE
    void operator()(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators,
            BiasTileIterator bias_iterator,
            OutputTileIterator
                    source_iterator) {
        if (output_op.is_bias_needed() && (!output_op.is_source_needed())) {
            compute_with_bias(output_op, destination_iterator, accumulators,
                              bias_iterator);
        } else if (output_op.is_bias_needed() && output_op.is_source_needed()) {
            compute_with_bias_add_source(output_op, destination_iterator,
                                         accumulators, bias_iterator,
                                         source_iterator);
        } else if ((!output_op.is_bias_needed()) &&
                   (!output_op.is_source_needed())) {
            compute_without_bias(output_op, destination_iterator, accumulators);

        } else {
            compute_add_source(output_op, destination_iterator, accumulators,
                               source_iterator);
        }
    }

private:
    CUTLASS_DEVICE
    void compute_with_bias(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const& accumulators,
            BiasTileIterator
                    bias_iterator) {

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            if (iter >= 1) {
                if (UseSyncWarp)
                    __syncwarp();
                else
                    __syncthreads();
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            if (UseSyncWarp)
                __syncwarp();
            else
                __syncthreads();


            typename SharedLoadIterator::Fragment aligned_accum_fragment;

            shared_load_iterator_.load(aligned_accum_fragment);


            typename BiasTileIterator::Fragment bias_fragment;
            if (bias_iterator.valid()) {
                bias_iterator.load(bias_fragment);
            }
            ++bias_iterator;


            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &aligned_accum_fragment);

            BiasAccessType const* bias_frag_ptr =
                    reinterpret_cast<BiasAccessType const*>(&bias_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            static int const kBiasAdvanceIterations =
                    OutputTileIterator::Fragment::kElements /
                    BiasTileIterator::Fragment::kElements;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                output_frag_ptr[i] = output_op.apply_add_bias(
                        compute_frag_ptr[i],
                        bias_frag_ptr[i / kBiasAdvanceIterations]);
            }


            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    CUTLASS_DEVICE
    void compute_with_bias_add_source(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators,
            BiasTileIterator bias_iterator,
            OutputTileIterator
                    source_iterator) {
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();


        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);


        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            source_iterator.load(source_fragment);
            ++source_iterator;

            if (iter >= 1) {
                if (UseSyncWarp)
                    __syncwarp();
                else
                    __syncthreads();
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            if (UseSyncWarp)
                __syncwarp();
            else
                __syncthreads();


            typename SharedLoadIterator::Fragment aligned_accum_fragment;

            shared_load_iterator_.load(aligned_accum_fragment);


            typename BiasTileIterator::Fragment bias_fragment;
            if (bias_iterator.valid()) {
                bias_iterator.load(bias_fragment);
            }
            ++bias_iterator;


            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &aligned_accum_fragment);

            OutputAccessType const* source_frag_ptr =
                    reinterpret_cast<OutputAccessType const*>(&source_fragment);

            BiasAccessType const* bias_frag_ptr =
                    reinterpret_cast<BiasAccessType const*>(&bias_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            static int const kBiasAdvanceIterations =
                    OutputTileIterator::Fragment::kElements /
                    BiasTileIterator::Fragment::kElements;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                output_frag_ptr[i] = output_op.apply_add_bias_source(
                        compute_frag_ptr[i],
                        bias_frag_ptr[i / kBiasAdvanceIterations],
                        source_frag_ptr[i]);
            }


            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    CUTLASS_DEVICE
    void compute_add_source(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators,
            OutputTileIterator
                    source_iterator) {
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();


        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);


        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            source_iterator.load(source_fragment);
            ++source_iterator;

            if (iter >= 1) {
                if (UseSyncWarp)
                    __syncwarp();
                else
                    __syncthreads();
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            if (UseSyncWarp)
                __syncwarp();
            else
                __syncthreads();


            typename SharedLoadIterator::Fragment aligned_accum_fragment;

            shared_load_iterator_.load(aligned_accum_fragment);


            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &aligned_accum_fragment);

            OutputAccessType const* source_frag_ptr =
                    reinterpret_cast<OutputAccessType const*>(&source_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                output_frag_ptr[i] = output_op.apply_add_source(
                        compute_frag_ptr[i], source_frag_ptr[i]);
            }


            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    CUTLASS_DEVICE
    void compute_without_bias(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators) {

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            if (iter >= 1) {
                if (UseSyncWarp)
                    __syncwarp();
                else
                    __syncthreads();
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            if (UseSyncWarp)
                __syncwarp();
            else
                __syncthreads();


            typename SharedLoadIterator::Fragment aligned_accum_fragment;

            shared_load_iterator_.load(aligned_accum_fragment);


            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &aligned_accum_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                output_frag_ptr[i] = output_op.apply(compute_frag_ptr[i]);
            }


            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }
};

template <typename Shape_,
          typename Layout_,
          int PartitionsK,
          typename WarpMmaOperator_,
          typename OutputTileIterator_,
          typename AccumulatorFragmentIterator_,
          typename BiasTileIterator_,
          typename OutputOp_
          >
class ConvolutionEpilogueWithoutSharedLoad {
public:
    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using BiasTileIterator = BiasTileIterator_;
    using OutputOp = OutputOp_;

    using AccumulatorTile =
            typename AccumulatorFragmentIterator::AccumulatorTile;

    using ElementAccumulator = typename AccumulatorTile::Element;

    using Layout = Layout_;
    using LongIndex = typename Layout::LongIndex;

    using ElementOutput = typename OutputTileIterator::Element;

    using ElementBias = typename BiasTileIterator::Element;

    static int const kElementsPerAccess =
            OutputTileIterator::kElementsPerAccess;

    static int const kOutputElements =
            OutputTileIterator::ThreadMap::Iterations::kColumn *
            OutputTileIterator::ThreadMap::Iterations::kRow *
            kElementsPerAccess;

    using TensorRef = typename OutputTileIterator::TensorRef;

    using SyncTensorRef =
            typename cutlass::TensorRef<int,
                                        cutlass::layout::PackedVectorLayout>;

    using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

    using OutputAccessType = Array<typename OutputTileIterator::Element,
                                   OutputTileIterator::kElementsPerAccess>;

    using AccumulatorAccessType =
            Array<ElementAccumulator, OutputTileIterator::kElementsPerAccess>;

    using BiasAccessType = Array<typename BiasTileIterator::Element,
                                 OutputTileIterator::kElementsPerAccess>;

    using WarpCount = gemm::GemmShape<Shape::kM / WarpMmaOperator::Shape::kM,
                                      Shape::kN / WarpMmaOperator::Shape::kN,
                                      kPartitionsK>;

public:
    static_assert((!(OutputTileIterator::kElementsPerAccess % 4) ||
                   OutputTileIterator::kElementsPerAccess == 1),
                  "OutputTileIterator::kElementsPerAccess must be 1 or a "
                  "multiple of 4.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    OutputTileIterator::kElementsPerAccess),
                  "Divisibility");

    static_assert(kPartitionsK == 1,
                  "Split K algorithm for convolution not supported.");

    static_assert(!(kOutputElements % BiasTileIterator::Fragment::kElements),
                  "Divisibility");

    struct SharedStorage {};

public:
    CUTLASS_DEVICE
    ConvolutionEpilogueWithoutSharedLoad(
            SharedStorage& shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx
    ) {}

    CUTLASS_DEVICE
    void operator()(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators,
            BiasTileIterator bias_iterator,
            OutputTileIterator
                    source_iterator) {
        if (output_op.is_bias_needed() && (!output_op.is_source_needed())) {
            compute_with_bias(output_op, destination_iterator, accumulators,
                              bias_iterator);
        } else if (output_op.is_bias_needed() && output_op.is_source_needed()) {
            compute_with_bias_add_source(output_op, destination_iterator,
                                         accumulators, bias_iterator,
                                         source_iterator);
        } else if ((!output_op.is_bias_needed()) &&
                   (!output_op.is_source_needed())) {
            compute_without_bias(output_op, destination_iterator, accumulators);

        } else {
            compute_add_source(output_op, destination_iterator, accumulators,
                               source_iterator);
        }
    }

private:
    CUTLASS_DEVICE
    void compute_with_bias(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const& accumulators,
            BiasTileIterator
                    bias_iterator) {

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);


        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;


            typename BiasTileIterator::Fragment bias_fragment;
            if (bias_iterator.valid()) {
                bias_iterator.load(bias_fragment);
            }
            ++bias_iterator;


            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &accum_fragment);

            BiasAccessType const* bias_frag_ptr =
                    reinterpret_cast<BiasAccessType const*>(&bias_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            static int const kBiasAdvanceIterations =
                    kOutputElements / BiasTileIterator::Fragment::kElements;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                output_frag_ptr[i] = output_op.apply_add_bias(
                        compute_frag_ptr[i],
                        bias_frag_ptr[i / kBiasAdvanceIterations]);
            }

            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    CUTLASS_DEVICE
    void compute_with_bias_add_source(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators,
            BiasTileIterator bias_iterator,
            OutputTileIterator
                    source_iterator) {
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();


        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);


        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            source_iterator.set_iteration_index(iter);
            source_iterator.load(source_fragment);
            ++source_iterator;

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;


            typename BiasTileIterator::Fragment bias_fragment;
            if (bias_iterator.valid()) {
                bias_iterator.load(bias_fragment);
            }
            ++bias_iterator;


            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &accum_fragment);

            OutputAccessType const* source_frag_ptr =
                    reinterpret_cast<OutputAccessType const*>(&source_fragment);

            BiasAccessType const* bias_frag_ptr =
                    reinterpret_cast<BiasAccessType const*>(&bias_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            static int const kBiasAdvanceIterations =
                    kOutputElements / BiasTileIterator::Fragment::kElements;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                output_frag_ptr[i] = output_op.apply_add_bias_source(
                        compute_frag_ptr[i],
                        bias_frag_ptr[i / kBiasAdvanceIterations],
                        source_frag_ptr[i]);
            }

            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    CUTLASS_DEVICE
    void compute_add_source(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators,
            OutputTileIterator
                    source_iterator) {
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();


        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
            source_iterator.set_iteration_index(iter);
            source_iterator.load(source_fragment);
            ++source_iterator;

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;


            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &accum_fragment);

            OutputAccessType const* source_frag_ptr =
                    reinterpret_cast<OutputAccessType const*>(&source_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                output_frag_ptr[i] = output_op.apply_add_source(
                        compute_frag_ptr[i], source_frag_ptr[i]);
            }

            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    CUTLASS_DEVICE
    void compute_without_bias(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators) {

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);


        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;


            typename OutputTileIterator::Fragment output_fragment;

            OutputAccessType* output_frag_ptr =
                    reinterpret_cast<OutputAccessType*>(&output_fragment);

            AccumulatorAccessType const* compute_frag_ptr =
                    reinterpret_cast<AccumulatorAccessType const*>(
                            &accum_fragment);

            static int const kOutputOpIterations =
                    OutputTileIterator::Fragment::kElements /
                    OutputTileIterator::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kOutputOpIterations; ++i) {
                output_frag_ptr[i] = output_op.apply(compute_frag_ptr[i]);
            }

            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }
};


}
}
}

