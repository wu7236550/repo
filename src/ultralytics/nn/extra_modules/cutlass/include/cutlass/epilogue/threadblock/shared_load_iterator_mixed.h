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
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {


template <typename ThreadMap_,
          typename Element_,
          int ElementSizeBits_,
          int OutputSizeBits_,
          int ElementsPerAccess,
          int ContiguousLanes
          >
class SharedLoadIteratorMixed;


template <typename ThreadMap_,
          typename Element_
          >
class SharedLoadIteratorMixed<ThreadMap_, Element_, 32, 16, 8, 8> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = Element_;

    using Layout = layout::RowMajor;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = MatrixCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static int const kAlignment =
            ThreadMap::kElementsPerAccess * sizeof_bits<Element_>::value / 8;

    static int const kThreads = ThreadMap::kThreads;

    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    using AccessType =
            AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

    using LoadType = AlignedArray<Element,
                                  const_min(128 / sizeof_bits<Element>::value,
                                            ThreadMap::kElementsPerAccess),
                                  const_min(16, kAlignment)>;

    static int const kLoadsPerAccess =
            AccessType::kElements / LoadType::kElements;

private:

    LoadType const* pointers_[kLoadsPerAccess];

    int stride_;

public:

    CUTLASS_DEVICE
    SharedLoadIteratorMixed(TensorRef ref, int thread_idx)
            : stride_((ref.stride(0) / LoadType::kElements)) {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            pointers_[i] = reinterpret_cast<LoadType const*>(ref.data());

            int col_idx = (thread_offset.column() / kElementsPerAccess) *
                          kLoadsPerAccess;
            int bank_offset =
                    (col_idx * sizeof(LoadType) / 128) % kLoadsPerAccess;

            col_idx += (bank_offset + i) % kLoadsPerAccess;

            pointers_[i] += thread_offset.row() * stride_ + col_idx;
        }
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            pointers_ += pointer_offset / LoadType::kElements;
        }
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            pointers_[i] += offset.row() * stride_ +
                            offset.column() / LoadType::kElements;
        }
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int row_ptr_offset =
                            row * ThreadMap::Delta::kRow * stride_ +
                            group * ThreadMap::Delta::kGroup * stride_ +
                            cluster * ThreadMap::Delta::kCluster * stride_ +
                            pointer_offset / LoadType::kElements;

                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    LoadType* frag_ptr = reinterpret_cast<LoadType*>(&frag);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        int frag_idx =
                                frag_row_idx * ThreadMap::Iterations::kColumn +
                                column;

                        CUTLASS_PRAGMA_UNROLL
                        for (int v = 0; v < kLoadsPerAccess; ++v) {
                            int vector_idx =
                                    (column * ThreadMap::Delta::kColumn /
                                     kElementsPerAccess * kLoadsPerAccess);

                            LoadType const* memory_pointer =
                                    pointers_[v] + row_ptr_offset;

                            frag_ptr[frag_idx * kLoadsPerAccess + v] =
                                    memory_pointer[vector_idx];
                        }
                    }
                }
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }
};


template <typename ThreadMap_
          >
class SharedLoadIteratorMixed<ThreadMap_, int32_t, 32, 8, 16, 8> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = int32_t;

    using Layout = layout::RowMajor;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = MatrixCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static int const kAlignment = 16;

    static int const kThreads = ThreadMap::kThreads;

    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    using AccessType = AlignedArray<Element, 16, kAlignment>;

    using LoadType = AlignedArray<Element, 4, 16>;

    static int const kLoadsPerAccess = 4;

private:

    LoadType const* pointers_[kLoadsPerAccess];

    int stride_;

public:

    CUTLASS_DEVICE
    SharedLoadIteratorMixed(TensorRef ref, int thread_idx)
            : stride_((ref.stride(0) / LoadType::kElements)) {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

        LoadType const* base_ptr =
                reinterpret_cast<LoadType const*>(ref.data()) +
                thread_offset.row() * stride_;

        int lane_col_idx = thread_offset.column() / 16;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            int lane_offset = (lane_col_idx % 2) * 4 |
                              ((lane_col_idx / 2) * 8) |
                              ((lane_col_idx / 2) ^ i);

            pointers_[i] = base_ptr + lane_offset;
        }
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            pointers_[i] += pointer_offset / LoadType::kElements;
        }
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            pointers_[i] += offset.row() * stride_ +
                            offset.column() / LoadType::kElements;
        }
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int row_ptr_offset =
                            row * ThreadMap::Delta::kRow * stride_ +
                            group * ThreadMap::Delta::kGroup * stride_ +
                            cluster * ThreadMap::Delta::kCluster * stride_ +
                            pointer_offset / LoadType::kElements;

                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    LoadType* frag_ptr = reinterpret_cast<LoadType*>(&frag);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        int frag_idx =
                                frag_row_idx * ThreadMap::Iterations::kColumn +
                                column;

                        CUTLASS_PRAGMA_UNROLL
                        for (int v = 0; v < kLoadsPerAccess; ++v) {
                            LoadType const* memory_pointer = pointers_[v];

                            frag_ptr[frag_idx * kLoadsPerAccess + v] =
                                    memory_pointer[row_ptr_offset];
                        }
                    }
                }
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }
};


template <typename ThreadMap_
          >
class SharedLoadIteratorMixed<ThreadMap_, int32_t, 32, 8, 8, 8> {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;

    using Element = int32_t;

    using Layout = layout::RowMajor;
    using TensorRef = TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = MatrixCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static int const kAlignment = 8;

    static int const kThreads = ThreadMap::kThreads;

    using Fragment = Array<Element, ThreadMap::Iterations::kColumn *
                                            ThreadMap::Iterations::kRow *
                                            ThreadMap::Iterations::kGroup *
                                            ThreadMap::Iterations::kCluster *
                                            ThreadMap::kElementsPerAccess>;

    using AccessType = AlignedArray<Element, 8, kAlignment>;

    using LoadType = AlignedArray<Element, 4, 16>;

    static int const kLoadsPerAccess = 2;

private:

    LoadType const* pointers_[kLoadsPerAccess];

    int stride_;

public:

    CUTLASS_DEVICE
    SharedLoadIteratorMixed(TensorRef ref, int thread_idx)
            : stride_((ref.stride(0) / LoadType::kElements)) {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

        LoadType const* base_ptr =
                reinterpret_cast<LoadType const*>(ref.data()) +
                thread_offset.row() * stride_;

        int lane_col_idx = thread_offset.column() / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            int lane_offset = (lane_col_idx % 8) * 2 | ((lane_col_idx / 4) ^ i);

            pointers_[i] = base_ptr + lane_offset;
        }
    }

    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            pointers_[i] += pointer_offset / LoadType::kElements;
        }
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kLoadsPerAccess; ++i) {
            pointers_[i] += offset.row() * stride_ +
                            offset.column() / LoadType::kElements;
        }
    }

    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
             ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup;
                 ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int row_ptr_offset =
                            row * ThreadMap::Delta::kRow * stride_ +
                            group * ThreadMap::Delta::kGroup * stride_ +
                            cluster * ThreadMap::Delta::kCluster * stride_ +
                            pointer_offset / LoadType::kElements;

                    int frag_row_idx =
                            (row +
                             ThreadMap::Iterations::kRow *
                                     (group +
                                      ThreadMap::Iterations::kGroup * cluster));

                    LoadType* frag_ptr = reinterpret_cast<LoadType*>(&frag);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0;
                         column < ThreadMap::Iterations::kColumn; ++column) {
                        int frag_idx =
                                frag_row_idx * ThreadMap::Iterations::kColumn +
                                column;

                        CUTLASS_PRAGMA_UNROLL
                        for (int v = 0; v < kLoadsPerAccess; ++v) {
                            LoadType const* memory_pointer = pointers_[v];

                            frag_ptr[frag_idx * kLoadsPerAccess + v] =
                                    memory_pointer[row_ptr_offset];
                        }
                    }
                }
            }
        }
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) { load_with_pointer_offset(frag, 0); }
};


}
}
}

