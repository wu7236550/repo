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

#include <stdexcept>
#include <list>
#include <vector>

#include "cutlass/library/library.h"
#include "cutlass/util/distribution.h"

#include "enumerated_types.h"


namespace cutlass {
namespace profiler {


class DeviceAllocation {
private:
    library::NumericTypeID type_;

    size_t batch_stride_;

    size_t capacity_;

    void* pointer_;

    library::LayoutTypeID layout_;

    std::vector<int> stride_;

    std::vector<int> extent_;

    int batch_count_;

    std::vector<uint8_t> tensor_ref_buffer_;

public:

    static size_t bytes(library::NumericTypeID type, size_t capacity);

    static std::vector<int> get_packed_layout(library::LayoutTypeID layout_id,
                                              std::vector<int> const& extent);

    static size_t construct_layout(void* bytes, library::LayoutTypeID layout_id,
                                   std::vector<int> const& extent,
                                   std::vector<int>& stride);

    static bool block_compare_equal(library::NumericTypeID numeric_type,
                                    void const* ptr_A, void const* ptr_B,
                                    size_t capacity);

    static bool block_compare_relatively_equal(
            library::NumericTypeID numeric_type, void const* ptr_A,
            void const* ptr_B, size_t capacity, double epsilon,
            double nonzero_floor);

public:

    DeviceAllocation();

    DeviceAllocation(library::NumericTypeID type, size_t capacity);

    DeviceAllocation(library::NumericTypeID type,
                     library::LayoutTypeID layout_id,
                     std::vector<int> const& extent,
                     std::vector<int> const& stride = std::vector<int>(),
                     int batch_count = 1);

    ~DeviceAllocation();

    DeviceAllocation& reset();

    DeviceAllocation& reset(library::NumericTypeID type, size_t capacity);

    DeviceAllocation& reset(library::NumericTypeID type,
                            library::LayoutTypeID layout_id,
                            std::vector<int> const& extent,
                            std::vector<int> const& stride = std::vector<int>(),
                            int batch_count = 1);

    std::vector<uint8_t>& tensor_ref() { return tensor_ref_buffer_; }

    bool good() const;

    library::NumericTypeID type() const;

    void* data() const;

    void* batch_data(int batch_idx) const;

    library::LayoutTypeID layout() const;

    std::vector<int> const& stride() const;

    std::vector<int> const& extent() const;

    int batch_count() const;

    int64_t batch_stride() const;

    int64_t batch_stride_bytes() const;

    size_t capacity() const;

    size_t bytes() const;

    void initialize_random_device(int seed, Distribution dist);

    void initialize_random_host(int seed, Distribution dist);

    void initialize_random_sparsemeta_device(int seed, int MetaSizeInBits);

    void initialize_random_sparsemeta_host(int seed, int MetaSizeInBits);

    void copy_from_device(void const* ptr);

    void copy_from_host(void const* ptr);

    void copy_to_host(void* ptr);

    void write_tensor_csv(std::ostream& out);
};

using DeviceAllocationList = std::list<DeviceAllocation>;


}
}

