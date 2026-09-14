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

#include <vector>

#include "cutlass/cutlass.h"

#include "enumerated_types.h"

#include "cutlass/library/library.h"

namespace cutlass {
namespace profiler {


struct PerformanceResult {
    size_t problem_index;

    library::Provider provider;

    library::OperationKind op_kind;

    Status status;

    Disposition disposition;

    DispositionMap verification_map;

    std::string operation_name;

    std::vector<std::pair<std::string, std::string> > arguments;

    int64_t bytes;

    int64_t flops;

    double runtime;


    PerformanceResult()
            : problem_index(0),
              op_kind(library::OperationKind::kInvalid),
              provider(library::Provider::kInvalid),
              disposition(Disposition::kNotRun),
              status(Status::kInvalid),
              bytes(0),
              flops(0),
              runtime(0) {}

    bool good() const { return runtime > 0; }

    double gflops_per_sec() const { return double(flops) / runtime / 1.0e6; }

    double gbytes_per_sec() const {
        return double(bytes) / double(1 << 30) / runtime * 1000.0;
    }
};

using PerformanceResultVector = std::vector<PerformanceResult>;


}
}
