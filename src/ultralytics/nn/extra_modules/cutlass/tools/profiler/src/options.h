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

#include <string>
#include <vector>
#include <map>

#include <cuda_runtime.h>

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/library/library.h"

#include "enumerated_types.h"

namespace cutlass {
namespace profiler {


class Options {
public:
    struct Library {

        AlgorithmMode algorithm_mode;

        std::vector<int> algorithms;


        Library(CommandLine const& cmdline);

        void print_usage(std::ostream& out) const;
        void print_options(std::ostream& out, int indent = 0) const;
    };

    struct Device {
        int device;

        cudaDeviceProp properties;

        size_t maximum_capacity;


        Device(CommandLine const& cmdline);

        void print_usage(std::ostream& out) const;
        void print_options(std::ostream& out, int indent = 0) const;
        void print_device_info(std::ostream& out) const;

        int compute_capability() const;
    };

    struct Initialization {
        bool enabled;

        bool fix_data_distribution;

        Distribution data_distribution;

        library::Provider provider;

        int seed;


        Initialization(CommandLine const& cmdline);

        void print_usage(std::ostream& out) const;
        void print_options(std::ostream& out, int indent = 0) const;

        static void get_distribution(cutlass::CommandLine const& args,
                                     std::string const& arg,
                                     cutlass::Distribution& dist);
    };

    struct Verification {

        bool enabled;

        double epsilon;

        double nonzero_floor;

        ProviderVector providers;

        SaveWorkspace save_workspace;


        Verification(CommandLine const& cmdline);

        void print_usage(std::ostream& out) const;
        void print_options(std::ostream& out, int indent = 0) const;

        bool provider_enabled(library::Provider provider) const;

        size_t index(library::Provider provider) const;
    };

    struct Profiling {
        int workspace_count;

        int warmup_iterations;

        int iterations;

        int sleep_duration;

        bool enabled;

        ProviderVector providers;


        Profiling(CommandLine const& cmdline);

        void print_usage(std::ostream& out) const;
        void print_options(std::ostream& out, int indent = 0) const;

        bool provider_enabled(library::Provider provider) const;

        size_t index(library::Provider provider) const;
    };

    struct Report {
        bool append;

        std::string output_path;

        std::string junit_output_path;

        std::vector<std::pair<std::string, std::string>> pivot_tags;

        bool report_not_run;

        bool verbose;


        Report(CommandLine const& cmdline);

        void print_usage(std::ostream& out) const;
        void print_options(std::ostream& out, int indent = 0) const;
    };

    struct About {
        bool help;

        bool version;

        bool device_info;


        About(CommandLine const& cmdline);

        void print_usage(std::ostream& out) const;
        void print_options(std::ostream& out, int indent = 0) const;

        static void print_version(std::ostream& out);
    };

public:

    ExecutionMode execution_mode;

    library::OperationKind operation_kind;

    std::vector<std::string> operation_names;

    std::vector<std::string> excluded_operation_names;


    CommandLine cmdline;
    Device device;
    Initialization initialization;
    Library library;
    Verification verification;
    Profiling profiling;
    Report report;
    About about;

public:
    Options(CommandLine const& cmdline);

    void print_usage(std::ostream& out) const;
    void print_options(std::ostream& out) const;

    static std::string indent_str(int indent);
};


}
}
