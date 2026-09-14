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

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <typename Shape_,
          typename WarpMmaOperator_,
          int PartitionsK,
          typename OutputTileIterator_,
          typename AccumulatorFragmentIterator_,
          typename WarpTileIterator_,
          typename SharedLoadIterator_,
          typename OutputOp_,
          typename Padding_
          >
class Epilogue : public EpilogueBase<Shape_, typename WarpMmaOperator_::Shape,
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
    using OutputOp = OutputOp_;
    using Padding = Padding_;

    using Layout = layout::RowMajor;
    using LongIndex = typename Layout::LongIndex;

    using AccumulatorTile = typename Base::AccumulatorTile;

    using ElementAccumulator = typename WarpTileIterator::Element;

    using ElementOutput = typename OutputTileIterator::Element;

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

    using WarpCount = typename Base::WarpCount;

public:
    static_assert(
            SharedLoadIterator::Fragment::kElements ==
                    OutputTileIterator::Fragment::kElements,
            "Mismatch between shared load iterator and output tile iterator.");

    static_assert(OutputTileIterator::kElementsPerAccess,
                  "OutputTileIterator::kElementsPerAccess must not be zero.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    OutputTileIterator::kElementsPerAccess),
                  "Divisibility");

private:
    SharedLoadIterator shared_load_iterator_;

public:
    CUTLASS_DEVICE
    Epilogue(typename Base::SharedStorage&
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
            OutputTileIterator
                    source_iterator) {

        if (!output_op.is_source_needed()) {
            compute_source_not_needed_(output_op, destination_iterator,
                                       accumulators);
        } else {
            compute_source_needed_(output_op, destination_iterator,
                                   accumulators, source_iterator);
        }
    }

private:
    CUTLASS_DEVICE
    void compute_source_not_needed_(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators
    ) {

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);


        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            __syncthreads();

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            __syncthreads();


            typename SharedLoadIterator::Fragment
                    aligned_accum_fragment[kPartitionsK];

            shared_load_iterator_.load(aligned_accum_fragment[0]);

            if (kPartitionsK > 1) {
                plus<typename SharedLoadIterator::Fragment> add_fragments;
                const int tile_row_offset =
                        Base::SharedStorage::StorageShape::kRow / PartitionsK;

                CUTLASS_PRAGMA_UNROLL
                for (int i = 1; i < kPartitionsK; ++i) {
                    shared_load_iterator_.add_tile_offset({tile_row_offset, 0});
                    shared_load_iterator_.load(aligned_accum_fragment[i]);
                    aligned_accum_fragment[0] =
                            add_fragments(aligned_accum_fragment[0],
                                          aligned_accum_fragment[i]);
                }

                shared_load_iterator_.add_tile_offset(
                        {-1 * (kPartitionsK - 1) * tile_row_offset, 0});
            }


            typename OutputTileIterator::Fragment output_fragment;

            apply_output_operator_source_not_needed_(output_fragment, output_op,
                                                     aligned_accum_fragment[0]);


            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    CUTLASS_DEVICE
    void compute_source_needed_(
            OutputOp const& output_op,
            OutputTileIterator
                    destination_iterator,
            AccumulatorTile const&
                    accumulators,
            OutputTileIterator
                    source_iterator
    ) {
        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();


        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);


        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            source_iterator.load(source_fragment);
            ++source_iterator;


            __syncthreads();

            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;

            this->warp_tile_iterator_.store(accum_fragment);

            __syncthreads();


            typename SharedLoadIterator::Fragment
                    aligned_accum_fragment[kPartitionsK];

            shared_load_iterator_.load(aligned_accum_fragment[0]);

            if (kPartitionsK > 1) {
                plus<typename SharedLoadIterator::Fragment> add_fragments;
                const int tile_row_offset =
                        Base::SharedStorage::StorageShape::kRow / PartitionsK;

                CUTLASS_PRAGMA_UNROLL
                for (int i = 1; i < kPartitionsK; ++i) {
                    shared_load_iterator_.add_tile_offset({tile_row_offset, 0});
                    shared_load_iterator_.load(aligned_accum_fragment[i]);
                    aligned_accum_fragment[0] =
                            add_fragments(aligned_accum_fragment[0],
                                          aligned_accum_fragment[i]);
                }

                shared_load_iterator_.add_tile_offset(
                        {-1 * (kPartitionsK - 1) * tile_row_offset, 0});
            }


            typename OutputTileIterator::Fragment output_fragment;

            apply_output_operator_(output_fragment, output_op,
                                   aligned_accum_fragment[0], source_fragment);


            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

    CUTLASS_DEVICE
    void apply_output_operator_(
            typename OutputTileIterator::Fragment& output_fragment,
            OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const& aligned_accum_fragment,
            typename OutputTileIterator::Fragment const& source_fragment) {
        OutputAccessType* output_frag_ptr =
                reinterpret_cast<OutputAccessType*>(&output_fragment);

        AccumulatorAccessType const* compute_frag_ptr =
                reinterpret_cast<AccumulatorAccessType const*>(
                        &aligned_accum_fragment);

        OutputAccessType const* source_frag_ptr =
                reinterpret_cast<OutputAccessType const*>(&source_fragment);

        int const kOutputOpIterations =
                OutputTileIterator::Fragment::kElements /
                OutputTileIterator::kElementsPerAccess;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kOutputOpIterations; ++i) {
            output_frag_ptr[i] =
                    output_op(compute_frag_ptr[i], source_frag_ptr[i]);
        }
    }

    CUTLASS_DEVICE
    void apply_output_operator_source_not_needed_(
            typename OutputTileIterator::Fragment& output_fragment,
            OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const&
                    aligned_accum_fragment) {
        OutputAccessType* output_frag_ptr =
                reinterpret_cast<OutputAccessType*>(&output_fragment);

        AccumulatorAccessType const* compute_frag_ptr =
                reinterpret_cast<AccumulatorAccessType const*>(
                        &aligned_accum_fragment);

        int const kOutputOpIterations =
                OutputTileIterator::Fragment::kElements /
                OutputTileIterator::kElementsPerAccess;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kOutputOpIterations; ++i) {
            output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
        }
    }
};


}
}
}

