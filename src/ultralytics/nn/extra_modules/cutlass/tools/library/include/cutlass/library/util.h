/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#include "cutlass/library/library.h"


namespace cutlass {
namespace library {


template <typename T>
T from_string(std::string const&);

char const* to_string(Provider provider, bool pretty = false);

template <>
Provider from_string<Provider>(std::string const& str);

char const* to_string(GemmKind type, bool pretty = false);

char const* to_string(OperationKind type, bool pretty = false);

template <>
OperationKind from_string<OperationKind>(std::string const& str);

char const* to_string(NumericTypeID type, bool pretty = false);

template <>
NumericTypeID from_string<NumericTypeID>(std::string const& str);

int sizeof_bits(NumericTypeID type);

bool is_complex_type(NumericTypeID type);

NumericTypeID get_real_type(NumericTypeID type);

bool is_integer_type(NumericTypeID type);

bool is_signed_type(NumericTypeID type);

bool is_signed_integer(NumericTypeID type);

bool is_unsigned_integer(NumericTypeID type);

bool is_float_type(NumericTypeID type);

char const* to_string(Status status, bool pretty = false);

char const* to_string(LayoutTypeID layout, bool pretty = false);

template <>
LayoutTypeID from_string<LayoutTypeID>(std::string const& str);

int get_layout_stride_rank(LayoutTypeID layout_id);

char const* to_string(OpcodeClassID type, bool pretty = false);

template <>
OpcodeClassID from_string<OpcodeClassID>(std::string const& str);

char const* to_string(ComplexTransform type, bool pretty = false);

template <>
ComplexTransform from_string<ComplexTransform>(std::string const& str);

char const* to_string(SplitKMode split_k_mode, bool pretty = false);

template <>
SplitKMode from_string<SplitKMode>(std::string const& str);

char const* to_string(ConvModeID type, bool pretty = false);

template <>
ConvModeID from_string<ConvModeID>(std::string const& str);

char const* to_string(IteratorAlgorithmID type, bool pretty = false);

template <>
IteratorAlgorithmID from_string<IteratorAlgorithmID>(std::string const& str);

char const* to_string(ConvKind type, bool pretty = false);

template <>
ConvKind from_string<ConvKind>(std::string const& str);

std::string lexical_cast(int64_t int_value);

bool lexical_cast(std::vector<uint8_t>& bytes, NumericTypeID type,
                  std::string const& str);

std::string lexical_cast(std::vector<uint8_t>& bytes, NumericTypeID type);

bool cast_from_int64(std::vector<uint8_t>& bytes, NumericTypeID type,
                     int64_t src);

bool cast_from_uint64(std::vector<uint8_t>& bytes, NumericTypeID type,
                      uint64_t src);

bool cast_from_double(std::vector<uint8_t>& bytes, NumericTypeID type,
                      double src);


}
}

