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
#include <string>
#include <memory>
#include <unordered_map>

#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/library/manifest.h"

#include "options.h"
#include "device_context.h"
#include "performance_result.h"
#include "performance_report.h"
#include "problem_space.h"
#include "debug.h"


namespace cutlass {
namespace profiler {


class OperationProfiler {
public:
protected:

    library::OperationKind kind_;

    std::string description_;

    ArgumentDescriptionVector arguments_;

    ProviderVector verification_providers_;

    PerformanceResult model_result_;

    PerformanceResultVector results_;

public:

    OperationProfiler();

    OperationProfiler(
            Options const& options, library::OperationKind kind,
            ArgumentDescriptionVector const& arguments =
                    ArgumentDescriptionVector(),
            ProviderVector const& verification_providers = ProviderVector());

    virtual ~OperationProfiler();

    library::OperationKind kind() const { return kind_; }

    std::string const& description() const;

    ArgumentDescriptionVector const& arguments() const { return arguments_; }

public:

    virtual void print_usage(std::ostream& out) const;

    virtual void print_examples(std::ostream& out) const = 0;

    virtual int profile_all(Options const& options,
                            library::Manifest const& manifest,
                            DeviceContext& device_context);

public:

    virtual Status initialize_configuration(
            Options const& options, PerformanceReport& report,
            DeviceContext& device_context, library::Operation const* operation,
            ProblemSpace const& problem_space,
            ProblemSpace::Problem const& problem) = 0;

    virtual Status initialize_workspace(
            Options const& options, PerformanceReport& report,
            DeviceContext& device_context, library::Operation const* operation,
            ProblemSpace const& problem_space,
            ProblemSpace::Problem const& problem) = 0;

    virtual bool verify_cutlass(Options const& options,
                                PerformanceReport& report,
                                DeviceContext& device_context,
                                library::Operation const* operation,
                                ProblemSpace const& problem_space,
                                ProblemSpace::Problem const& problem) = 0;

    virtual bool profile(Options const& options, PerformanceReport& report,
                         DeviceContext& device_context,
                         library::Operation const* operation,
                         ProblemSpace const& problem_space,
                         ProblemSpace::Problem const& problem) = 0;

public:

    static void sleep(int sleep_duration);

    static bool satisfies(library::OperationDescription const& op_desc,
                          ProblemSpace const& problem_space,
                          ProblemSpace::Problem const& problem);

    static Disposition compare_tensors(Options const& options,
                                       DeviceAllocation& experimental,
                                       DeviceAllocation& reference,
                                       int64_t count = 0);

    static void save_workspace(DeviceContext& device_context,
                               Options const& options,
                               library::OperationDescription const& desc,
                               library::Provider provider,
                               library::Provider verification_provider =
                                       library::Provider::kInvalid);

    static void set_argument(PerformanceResult& result, char const* name,
                             ProblemSpace const& problem_space,
                             std::string const& value);

    static void set_argument(PerformanceResult& result, char const* name,
                             ProblemSpace const& problem_space, int64_t value);

protected:
    static void initialize_result_(
            PerformanceResult& result,
            library::OperationDescription const& operation_desc,
            ProblemSpace const& problem_space);

    virtual Status profile_cutlass_(double& runtime, Options const& options,
                                    library::Operation const* operation,
                                    void* arguments, void* host_workspace,
                                    void* device_workspace);

private:
    bool find_string_matches_(std::string const& filter_string,
                              std::string const& operation_name);
};


using OperationProfilerVector = std::vector<std::unique_ptr<OperationProfiler>>;


}
}

