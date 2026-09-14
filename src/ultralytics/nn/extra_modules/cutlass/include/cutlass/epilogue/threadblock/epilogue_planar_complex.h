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
#include "cutlass/array_planar_complex.h"
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
class EpiloguePlanarComplex {
public:
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

    using AccumulatorTile =
            ArrayPlanarComplex<typename WarpMmaOperator::FragmentC::Element,
                               WarpMmaOperator::FragmentC::kElements>;

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

    using WarpShape = typename WarpMmaOperator::Shape;

    using WarpCount = gemm::GemmShape<Shape::kM / WarpShape::kM,
                                      Shape::kN / WarpShape::kN, kPartitionsK>;

    struct SharedStorage {

        using Element = typename WarpTileIterator::Element;

        using TensorRef = typename WarpTileIterator::TensorRef;

        using Layout = typename WarpTileIterator::Layout;

        using Shape =
                MatrixShape<WarpCount::kM * WarpTileIterator::Shape::kRow *
                                    WarpCount::kK,
                            WarpCount::kN * WarpTileIterator::Shape::kColumn>;

        using StorageShape = MatrixShape<Shape::kRow + Padding::kRow,
                                         Shape::kColumn + Padding::kColumn>;

        static int const kImaginaryStride = StorageShape::kCount;


        AlignedBuffer<Element, kImaginaryStride * 2> storage;


        CUTLASS_DEVICE
        Element* data() { return storage.data(); }

        CUTLASS_DEVICE
        TensorRef reference() {
            return TensorRef(storage.data(),
                             Layout::packed({StorageShape::kRow,
                                             StorageShape::kColumn}));
        }
    };

private:

    SharedStorage& shared_storage_;

    SharedLoadIterator shared_load_iterator_;

    WarpTileIterator warp_tile_iterator_;

public:
    CUTLASS_DEVICE
    EpiloguePlanarComplex(
            SharedStorage& shared_storage,
            int thread_idx,
            int warp_idx,
            int lane_idx
            )
            : shared_storage_(shared_storage),
              shared_load_iterator_(shared_storage.reference(), thread_idx),
              warp_tile_iterator_(shared_storage.reference(), lane_idx) {

        int warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
        int warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
        int warp_m = warp_mn % WarpCount::kM;
        int warp_n = warp_mn / WarpCount::kM;

        MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m, warp_n};

        warp_tile_iterator_.add_tile_offset(warp_offset);
    }

    CUTLASS_DEVICE
    void operator()(
            OutputOp const& output_op,
            OutputTileIterator destination_iterator_real,
            OutputTileIterator destination_iterator_imag,
            AccumulatorTile const&
                    accumulators,
            OutputTileIterator
                    source_iterator_real,
            OutputTileIterator
                    source_iterator_imag) {

        typename OutputTileIterator::Fragment source_fragment_real;
        typename OutputTileIterator::Fragment source_fragment_imag;

        if (!output_op.is_source_needed()) {
            source_iterator_real.clear_mask();
            source_iterator_imag.clear_mask();
        }

        source_fragment_real.clear();
        source_fragment_imag.clear();


        AccumulatorFragmentIterator accum_fragment_iterator_real(
                accumulators.real);
        AccumulatorFragmentIterator accum_fragment_iterator_imag(
                accumulators.imag);


        CUTLASS_PRAGMA_UNROLL
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            source_iterator_real.load(source_fragment_real);
            source_iterator_imag.load(source_fragment_imag);

            ++source_iterator_real;
            ++source_iterator_imag;


            __syncthreads();

            typename AccumulatorFragmentIterator::Fragment accum_fragment_real;
            typename AccumulatorFragmentIterator::Fragment accum_fragment_imag;

            accum_fragment_iterator_real.load(accum_fragment_real);
            accum_fragment_iterator_imag.load(accum_fragment_imag);

            ++accum_fragment_iterator_real;
            ++accum_fragment_iterator_imag;

            this->warp_tile_iterator_.store(accum_fragment_real);
            this->warp_tile_iterator_.store_with_pointer_offset(
                    accum_fragment_imag, SharedStorage::kImaginaryStride);

            __syncthreads();


            typename SharedLoadIterator::Fragment
                    aligned_accum_fragment_real[kPartitionsK];
            typename SharedLoadIterator::Fragment
                    aligned_accum_fragment_imag[kPartitionsK];

            shared_load_iterator_.load(aligned_accum_fragment_real[0]);
            shared_load_iterator_.load_with_pointer_offset(
                    aligned_accum_fragment_imag[0],
                    SharedStorage::kImaginaryStride);

            static_assert(
                    kPartitionsK == 1,
                    "Sliced-K not supported for planar complex at this time");


            typename OutputTileIterator::Fragment output_fragment_real;
            typename OutputTileIterator::Fragment output_fragment_imag;

            apply_output_operator_(output_fragment_real, output_fragment_imag,
                                   output_op, aligned_accum_fragment_real[0],
                                   aligned_accum_fragment_imag[0],
                                   source_fragment_real, source_fragment_imag);


            destination_iterator_real.store(output_fragment_real);
            destination_iterator_imag.store(output_fragment_imag);

            ++destination_iterator_real;
            ++destination_iterator_imag;
        }
    }

private:
    CUTLASS_DEVICE
    void apply_output_operator_(
            typename OutputTileIterator::Fragment& output_fragment_real,
            typename OutputTileIterator::Fragment& output_fragment_imag,
            OutputOp const& output_op,
            typename SharedLoadIterator::Fragment const&
                    aligned_accum_fragment_real,
            typename SharedLoadIterator::Fragment const&
                    aligned_accum_fragment_imag,
            typename OutputTileIterator::Fragment const& source_fragment_real,
            typename OutputTileIterator::Fragment const& source_fragment_imag) {
        OutputAccessType* output_frag_real_ptr =
                reinterpret_cast<OutputAccessType*>(&output_fragment_real);

        OutputAccessType* output_frag_imag_ptr =
                reinterpret_cast<OutputAccessType*>(&output_fragment_imag);

        AccumulatorAccessType const* compute_frag_real_ptr =
                reinterpret_cast<AccumulatorAccessType const*>(
                        &aligned_accum_fragment_real);

        AccumulatorAccessType const* compute_frag_imag_ptr =
                reinterpret_cast<AccumulatorAccessType const*>(
                        &aligned_accum_fragment_imag);

        OutputAccessType const* source_frag_real_ptr =
                reinterpret_cast<OutputAccessType const*>(
                        &source_fragment_real);

        OutputAccessType const* source_frag_imag_ptr =
                reinterpret_cast<OutputAccessType const*>(
                        &source_fragment_imag);

        int const kOutputOpIterations =
                OutputTileIterator::Fragment::kElements /
                OutputTileIterator::kElementsPerAccess;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kOutputOpIterations; ++i) {
            auto result_fragment =
                    output_op(make_ArrayPlanarComplex(compute_frag_real_ptr[i],
                                                      compute_frag_imag_ptr[i]),
                              make_ArrayPlanarComplex(source_frag_real_ptr[i],
                                                      source_frag_imag_ptr[i]));

            output_frag_real_ptr[i] = result_fragment.real;
            output_frag_imag_ptr[i] = result_fragment.imag;
        }
    }
};


}
}
}

