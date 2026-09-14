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

#include <assert.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <
        typename Shape_,
        typename WarpMmaOperator_,
        int PartitionsK,
        typename OutputTileIterator_,
        typename AccumulatorFragmentIterator_,
        typename OutputOp_,
        int InterleavedK,
        bool IsBetaZero = false>
class InterleavedEpilogue {
public:
    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using OutputTileIterator = OutputTileIterator_;
    using OutputOp = OutputOp_;

    using AccumulatorTile =
            typename AccumulatorFragmentIterator::AccumulatorTile;

    using ElementAccumulator = typename AccumulatorTile::Element;

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

    using AccumulatorAccessType =
            Array<ElementAccumulator, OutputTileIterator::kElementsPerAccess>;

    using WarpCount = gemm::GemmShape<Shape::kM / WarpMmaOperator::Shape::kM,
                                      Shape::kN / WarpMmaOperator::Shape::kN,
                                      kPartitionsK>;

public:
    static_assert(OutputTileIterator::kElementsPerAccess,
                  "This must not be zero.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    OutputTileIterator::kElementsPerAccess),
                  "Divisibility");

    struct SharedStorage {};

public:
    CUTLASS_DEVICE
    InterleavedEpilogue(
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
            OutputTileIterator
                    source_iterator) {


        if (IsBetaZero && output_op.is_source_needed())
            assert(0);

        typename OutputTileIterator::Fragment source_fragment;

        if (!IsBetaZero) {
            if (!output_op.is_source_needed()) {
                source_iterator.clear_mask();
            }
        }

        source_fragment.clear();


        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);


        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            if (!IsBetaZero) {
                source_iterator.set_iteration_index(iter);
                source_iterator.load(source_fragment);
                ++source_iterator;
            }


            typename AccumulatorFragmentIterator::Fragment accum_fragment;

            accum_fragment_iterator.load(accum_fragment);
            ++accum_fragment_iterator;


            typename OutputTileIterator::Fragment output_fragment;
            apply_output_operator_(output_op, output_fragment, accum_fragment,
                                   source_fragment);


            destination_iterator.set_iteration_index(iter);
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }
    }

private:
    CUTLASS_DEVICE
    void apply_output_operator_(
            OutputOp const& output_op,
            typename OutputTileIterator::Fragment& output_fragment,
            typename AccumulatorFragmentIterator::Fragment const&
                    aligned_accum_fragment,
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
};


}
}
}

