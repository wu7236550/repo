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

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/epilogue/warp/tensor_op_policy.h"


namespace cutlass {
namespace epilogue {
namespace warp {


template <typename WarpShape_,
          typename OperatorShape_,
          typename Element_,
          int ElementSizeBits,
          int OutputSizeBits,
          int OutputElementCount,
          int ContiguousLanes
          >
class TileIteratorTensorOpMixed {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using Element = Element_;
    using Layout = layout::RowMajor;
    static int const kOutputElementCount = OutputElementCount;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord =
            MatrixCoord;
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Shape = MatrixShape<Policy::kRowsPerIteration, WarpShape::kN>;

    using Fragment = Array<Element, Policy::OperatorCount::kColumn *
                                            Policy::kElementsPerAccess>;


    static int const kIterations = Policy::kIterations;

    struct Detail {
        static int const kLanesInQuad = 4;

        static int const kPointerCount =
                (OutputElementCount * sizeof_bits<Element>::value) /
                (const_min(128,
                           OutputElementCount* sizeof_bits<Element>::value));

        static_assert(kPointerCount <= 4,
                      "Can only accommodate four pointers at present.");
        static_assert(sizeof(Element) == 4,
                      "This can only be used with 32b accumulator data types "
                      "(f32, s32).");
    };

    using Padding =
            MatrixShape<0, Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:
    using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;


    AccessType* pointers_[Detail::kPointerCount];

    int stride_;

    int warp_column_;

public:
    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed() {
        CUTLASS_PRAGMA_UNROLL
        for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
            pointers_[i] = nullptr;
        }
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed(TensorRef const& ref, unsigned lane_id)
            : stride_(ref.stride()[0] / Policy::kElementsPerAccess),
              warp_column_(0) {
        int quad_id = (lane_id / Detail::kLanesInQuad);
        int lane_in_quad = (lane_id % Detail::kLanesInQuad);

        CUTLASS_PRAGMA_UNROLL
        for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
            AccessType* ptr = reinterpret_cast<AccessType*>(ref.data()) +
                              quad_id * stride_;
            int column_idx =
                    (lane_in_quad % 2) +
                    (((lane_in_quad / 2) + i) % Detail::kPointerCount) * 2;

            ptr += column_idx;

            if (i == 0) {
                pointers_[0 % Detail::kPointerCount] = ptr;
            } else if (i == 1) {
                pointers_[1 % Detail::kPointerCount] = ptr;
            } else if (i == 2) {
                pointers_[2 % Detail::kPointerCount] = ptr;
            } else if (i == 3) {
                pointers_[3 % Detail::kPointerCount] = ptr;
            }
        }
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& add_pointer_offset(Index pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
            pointers_[i] += pointer_offset / Policy::kElementsPerAccess;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& add_tile_offset(TensorCoord const& tile_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
            pointers_[i] += tile_offset.row() * Shape::kRow * stride_ +
                            tile_offset.column() * Shape::kColumn /
                                    Policy::kElementsPerAccess;
        }

        warp_column_ += tile_offset.column() * Shape::kColumn;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& operator+=(TensorCoord const& tile_offset) {
        return add_tile_offset(tile_offset);
    }

    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int64_t n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            int column_idx = warp_column_ + n * Detail::kLanesInQuad *
                                                    Policy::kElementsPerAccess;
            int ptr_idx = ((column_idx * sizeof_bits<Element>::value) / 1024) %
                          Detail::kPointerCount;

            AccessType* ptr;
            if (ptr_idx == 0) {
                ptr = pointers_[0 % Detail::kPointerCount];
            } else if (ptr_idx == 1) {
                ptr = pointers_[1 % Detail::kPointerCount];
            } else if (ptr_idx == 2) {
                ptr = pointers_[2 % Detail::kPointerCount];
            } else if (ptr_idx == 3) {
                ptr = pointers_[3 % Detail::kPointerCount];
            }

            int offset = n * Detail::kLanesInQuad +
                         pointer_offset / Policy::kElementsPerAccess;
#if 0
      AccessType *smem_ptr = pointers_[ptr_idx];
      smem_ptr[offset] = frag_ptr[n];
#else
            uint32_t smem_addr = arch::cutlass_get_smem_pointer(ptr);
            uint32_t const* data =
                    reinterpret_cast<uint32_t const*>(frag_ptr + n);
            uint32_t offset_in_bytes = offset * sizeof(AccessType);

            asm volatile(
                    "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; "
                    "st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
                    :
                    : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]),
                      "r"(data[1]));
#endif
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }

    CUTLASS_HOST_DEVICE
    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int64_t n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            int column_idx = warp_column_ + n * Detail::kLanesInQuad *
                                                    Policy::kElementsPerAccess;
            int ptr_idx = ((column_idx * sizeof_bits<Element>::value) / 1024) %
                          Detail::kPointerCount;

            AccessType const* smem_ptr = pointers_[ptr_idx];
            frag_ptr[n] = smem_ptr[n * Detail::kLanesInQuad +
                                   pointer_offset / Policy::kElementsPerAccess];
        }
    }

    CUTLASS_HOST_DEVICE
    void load(Fragment& frag) const { load_with_pointer_offset(frag, 0); }
};


template <typename WarpShape_,
          typename OperatorShape_
          >
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, int32_t, 32, 8, 16,
                                8> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using Element = int32_t;
    using Layout = layout::RowMajor;
    static int const kOutputElementCount = 16;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord =
            MatrixCoord;
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Shape = MatrixShape<Policy::kRowsPerIteration, WarpShape::kN>;

    using Fragment = Array<Element, Policy::OperatorCount::kColumn *
                                            Policy::kElementsPerAccess>;


    static int const kIterations = Policy::kIterations;

    struct Detail {
        static int const kLanesInQuad = 4;

        static int const kPointerCount = 2;

        static int const kOffsetCount = 4;

        static_assert(sizeof(Element) == 4,
                      "This can only be used with 32b accumulator data types "
                      "(f32, s32).");
    };

    using Padding = MatrixShape<0, Detail::kLanesInQuad * 2>;

private:
    using AccessType = AlignedArray<Element, 2>;


    AccessType* pointers_[Detail::kPointerCount];

    int stride_;

    int uniform_offset_[Detail::kOffsetCount];

public:
    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed() {
        CUTLASS_PRAGMA_UNROLL
        for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
            pointers_[i] = nullptr;
        }
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed(TensorRef const& ref, unsigned lane_id)
            : stride_(ref.stride()[0] / AccessType::kElements) {
        int quad_id = (lane_id / Detail::kLanesInQuad);
        int lane_in_quad = (lane_id % Detail::kLanesInQuad);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Detail::kPointerCount; ++i) {
            AccessType* ptr = reinterpret_cast<AccessType*>(ref.data()) +
                              quad_id * stride_;
            int column_idx = lane_in_quad ^ (i * 2);

            ptr += column_idx;

            if (i == 0) {
                pointers_[0] = ptr;
            } else if (i == 1) {
                pointers_[1] = ptr;
            }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Detail::kOffsetCount; ++i) {
            uniform_offset_[i] = (i ^ 0) * 4 * sizeof(AccessType);
        }
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& add_pointer_offset(Index pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
            pointers_[i] += pointer_offset / AccessType::kElements;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& add_tile_offset(TensorCoord const& tile_offset) {
        int ptr_offset =
                tile_offset.row() * Shape::kRow * stride_ +
                tile_offset.column() * Shape::kColumn / AccessType::kElements;

        pointers_[0] += ptr_offset;
        pointers_[1] += ptr_offset;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Detail::kOffsetCount; ++i) {
            uniform_offset_[i] =
                    (i ^ tile_offset.column()) * 4 * sizeof(AccessType);
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& operator+=(TensorCoord const& tile_offset) {
        return add_tile_offset(tile_offset);
    }

    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            int ptr_idx = (n / 4);
            int offset_idx = (n % 4);

            AccessType* ptr;
            if (ptr_idx == 0) {
                ptr = pointers_[0];
            } else if (ptr_idx == 1) {
                ptr = pointers_[1];
            }

            int offset = (n / 4) * 16 + pointer_offset / AccessType::kElements;

#if 0
      AccessType *smem_ptr = pointers_[ptr_idx];
      smem_ptr[offset] = frag_ptr[n];
#else
            uint32_t smem_addr = arch::cutlass_get_smem_pointer(ptr);
            uint32_t const* data =
                    reinterpret_cast<uint32_t const*>(frag_ptr + n);
            uint32_t offset_in_bytes =
                    offset * sizeof(AccessType) + uniform_offset_[offset_idx];

            asm volatile(
                    "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; "
                    "st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
                    :
                    : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]),
                      "r"(data[1]));
#endif
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};


template <typename WarpShape_,
          typename OperatorShape_
          >
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, int32_t, 32, 8, 8,
                                8> {
public:
    using WarpShape = WarpShape_;
    using OperatorShape = OperatorShape_;
    using Element = int32_t;
    using Layout = layout::RowMajor;
    static int const kOutputElementCount = 8;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorCoord =
            MatrixCoord;
    using Index = typename TensorRef::Index;
    using LongIndex = typename TensorRef::LongIndex;

    using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

    using Shape = MatrixShape<Policy::kRowsPerIteration, WarpShape::kN>;

    using Fragment = Array<Element, Policy::OperatorCount::kColumn *
                                            Policy::kElementsPerAccess>;


    static int const kIterations = Policy::kIterations;

    struct Detail {
        static int const kLanesInQuad = 4;

        static int const kPointerCount = 2;

        static_assert(sizeof(Element) == 4,
                      "This can only be used with 32b accumulator data types "
                      "(f32, s32).");
    };

    using Padding = MatrixShape<0, Detail::kLanesInQuad * 2>;

private:
    using AccessType = AlignedArray<Element, 2>;


    AccessType* pointers_[Detail::kPointerCount];

    int stride_;

public:
    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed() {
        CUTLASS_PRAGMA_UNROLL
        for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
            pointers_[i] = nullptr;
        }
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed(TensorRef const& ref, unsigned lane_id)
            : stride_(ref.stride()[0] / AccessType::kElements) {
        int quad_id = (lane_id / Detail::kLanesInQuad);
        int lane_in_quad = (lane_id % Detail::kLanesInQuad);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Detail::kPointerCount; ++i) {
            AccessType* ptr = reinterpret_cast<AccessType*>(ref.data()) +
                              quad_id * stride_;
            int column_idx = lane_in_quad ^ (i * 2);

            ptr += column_idx;

            if (i == 0) {
                pointers_[0] = ptr;
            } else if (i == 1) {
                pointers_[1] = ptr;
            }
        }
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& add_pointer_offset(Index pointer_offset) {
        CUTLASS_PRAGMA_UNROLL
        for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
            pointers_[i] += pointer_offset / AccessType::kElements;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& add_tile_offset(TensorCoord const& tile_offset) {
        int ptr_offset =
                tile_offset.row() * Shape::kRow * stride_ +
                tile_offset.column() * Shape::kColumn / AccessType::kElements;

        pointers_[0] += ptr_offset;
        pointers_[1] += ptr_offset;

        if (tile_offset.column() % 2) {
            auto tmp = pointers_[0];
            pointers_[0] = pointers_[1];
            pointers_[1] = tmp;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    TileIteratorTensorOpMixed& operator+=(TensorCoord const& tile_offset) {
        return add_tile_offset(tile_offset);
    }

    CUTLASS_DEVICE
    void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
            int ptr_idx = (n / 4);

            AccessType* ptr;
            if (ptr_idx == 0) {
                ptr = pointers_[0];
            } else if (ptr_idx == 1) {
                ptr = pointers_[1];
            }

            int offset = (n / 4) * 16 + pointer_offset / AccessType::kElements +
                         (n % 4) * 4;

#if 0
      AccessType *smem_ptr = pointers_[ptr_idx];
      smem_ptr[offset] = frag_ptr[n];
#else
            uint32_t smem_addr = arch::cutlass_get_smem_pointer(ptr);
            uint32_t const* data =
                    reinterpret_cast<uint32_t const*>(frag_ptr + n);
            uint32_t offset_in_bytes = offset * sizeof(AccessType);

            asm volatile(
                    "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; "
                    "st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
                    :
                    : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]),
                      "r"(data[1]));
#endif
        }
    }

    CUTLASS_HOST_DEVICE
    void store(Fragment const& frag) { store_with_pointer_offset(frag, 0); }
};


}
}
}

