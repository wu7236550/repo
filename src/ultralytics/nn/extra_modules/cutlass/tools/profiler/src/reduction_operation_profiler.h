/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
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
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>

#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/library/manifest.h"

#include "options.h"
#include "device_context.h"
#include "operation_profiler.h"
#include "performance_result.h"
#include "problem_space.h"
#if CUTLASS_ENABLE_CUDNN
#include "cudnn_helpers.h"
#endif
#include "debug.h"


namespace cutlass {
namespace profiler {


class ReductionOperationProfiler : public OperationProfiler {
public:
    struct ReductionWorkspace {
        DeviceAllocation* Workspace;
        DeviceAllocation* Source;
        DeviceAllocation* Destination;
        DeviceAllocation* Reference;

        library::ReductionConfiguration configuration;
        library::ReductionArguments arguments;

        std::vector<uint8_t> host_workspace;

        DeviceAllocation device_workspace;


        ReductionWorkspace()
                : Workspace(nullptr),
                  Source(nullptr),
                  Destination(nullptr),
                  Reference(nullptr) {}
    };

protected:

    MatrixCoord problem_;

    ReductionWorkspace conv_workspace_;

public:

    ReductionOperationProfiler(Options const& options);

    virtual ~ReductionOperationProfiler();

    virtual void print_usage(std::ostream& out) const;

    virtual void print_examples(std::ostream& out) const;

    virtual Status initialize_configuration(
            Options const& options, PerformanceReport& report,
            DeviceContext& device_context, library::Operation const* operation,
            ProblemSpace const& problem_space,
            ProblemSpace::Problem const& problem);

    virtual Status initialize_workspace(Options const& options,
                                        PerformanceReport& report,
                                        DeviceContext& device_context,
                                        library::Operation const* operation,
                                        ProblemSpace const& problem_space,
                                        ProblemSpace::Problem const& problem);

    virtual bool verify_cutlass(Options const& options,
                                PerformanceReport& report,
                                DeviceContext& device_context,
                                library::Operation const* operation,
                                ProblemSpace const& problem_space,
                                ProblemSpace::Problem const& problem);

    virtual bool profile(Options const& options, PerformanceReport& report,
                         DeviceContext& device_context,
                         library::Operation const* operation,
                         ProblemSpace const& problem_space,
                         ProblemSpace::Problem const& problem);
};


}
}

