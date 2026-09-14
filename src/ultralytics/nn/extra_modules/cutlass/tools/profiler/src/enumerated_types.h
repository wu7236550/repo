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
#include <iostream>
#include "cutlass/library/library.h"

#define TRACE(x) \
    { std::cout << __FILE__ << ":" << __LINE__ << "  " << x << std::endl; }

namespace cutlass {
namespace profiler {


template <typename T>
T from_string(std::string const&);


enum class ExecutionMode {
    kProfile,
    kDryRun,
    kEnumerate,
    kTrace,
    kInvalid
};

char const* to_string(ExecutionMode mode, bool pretty = false);

template <>
ExecutionMode from_string<ExecutionMode>(std::string const& str);


enum class AlgorithmMode {
    kMatching,
    kBest,
    kDefault,
    kInvalid
};

char const* to_string(AlgorithmMode mode, bool pretty = false);

template <>
AlgorithmMode from_string<AlgorithmMode>(std::string const& str);


enum class Disposition {
    kPassed,
    kFailed,
    kNotRun,
    kIncorrect,
    kNotVerified,
    kInvalidProblem,
    kNotSupported,
    kInvalid
};

char const* to_string(Disposition disposition, bool pretty = false);

template <>
Disposition from_string<Disposition>(std::string const& str);


enum class SaveWorkspace { kNever, kIncorrect, kAlways, kInvalid };

char const* to_string(SaveWorkspace save_option, bool pretty = false);

template <>
SaveWorkspace from_string<SaveWorkspace>(std::string const& str);


enum class ArgumentTypeID {
    kScalar,
    kInteger,
    kTensor,
    kBatchedTensor,
    kStructure,
    kEnumerated,
    kInvalid
};

char const* to_string(ArgumentTypeID type, bool pretty = false);

template <>
ArgumentTypeID from_string<ArgumentTypeID>(std::string const& str);

using ProviderVector = std::vector<library::Provider>;
using DispositionMap = std::map<library::Provider, Disposition>;


template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    for (int i = 0; i < v.size(); ++i) {
        out << to_string(v[i], true) << (i + 1 != v.size() ? "," : "");
    }
    return out;
}

}
}
