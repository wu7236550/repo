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

#include <map>
#include <string>

#include "cutlass/library/library.h"
#include "cutlass/library/util.h"

#include "options.h"
#include "device_allocation.h"

namespace cutlass {
namespace profiler {


class DeviceContext {
public:
    using AllocationMap = std::map<std::string, DeviceAllocation*>;

private:

    DeviceAllocationList device_memory_;

    AllocationMap allocations_;

public:
    DeviceAllocation* allocate_block(std::string const& name,
                                     library::NumericTypeID type,
                                     size_t capacity);

    DeviceAllocation* allocate_tensor(
            std::string const& name, library::NumericTypeID type,
            library::LayoutTypeID layout_id, std::vector<int> const& extent,
            std::vector<int> const& stride = std::vector<int>(),
            int batch_count = 1);

    DeviceAllocation* allocate_tensor(
            Options const& options, std::string const& name,
            library::NumericTypeID type, library::LayoutTypeID layout_id,
            std::vector<int> const& extent,
            std::vector<int> const& stride = std::vector<int>(),
            int batch_count = 1);

    DeviceAllocation* allocate_sparsemeta_tensor(
            Options const& options, std::string const& name,
            library::NumericTypeID type, library::LayoutTypeID layout_id,
            library::NumericTypeID type_a, std::vector<int> const& extent,
            std::vector<int> const& stride = std::vector<int>(),
            int batch_count = 1);

    void clear();

    void free();

    DeviceAllocation& at(std::string const& name);

    size_t size() const;

    AllocationMap::iterator begin();
    AllocationMap::iterator end();
};


}
}
