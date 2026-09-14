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

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif

#include "cutlass/platform/platform.h"

namespace cutlass {


template <int Bits, bool Signed = true>
struct integer_subbyte {
    static int const kBits = Bits;

    static bool const kSigned = Signed;

    using T = typename platform::conditional<kSigned, int, unsigned>::type;

    using Storage = uint8_t;

    static Storage const kMask = Storage((1 << kBits) - 1);


    Storage storage;


    CUTLASS_HOST_DEVICE
    integer_subbyte() {}

    CUTLASS_HOST_DEVICE
    integer_subbyte(int value)
            : storage(reinterpret_cast<Storage const&>(value) & kMask) {}

    CUTLASS_HOST_DEVICE
    integer_subbyte(unsigned value)
            : storage(reinterpret_cast<Storage const&>(value) & kMask) {}

    CUTLASS_HOST_DEVICE
    integer_subbyte(double value) {
        T tmp = static_cast<T>(value);
        storage = Storage(reinterpret_cast<unsigned const&>(tmp) & kMask);
    }

    CUTLASS_HOST_DEVICE
    operator T() const {
        if (kSigned) {
            if (storage & Storage(1 << (kBits - 1))) {
                return T(storage) | ~T(kMask);
            }
        }
        return T(storage);
    }

    CUTLASS_HOST_DEVICE
    bool operator==(integer_subbyte const& rhs) const {
        return storage == rhs.storage;
    }

    CUTLASS_HOST_DEVICE
    bool operator!=(integer_subbyte const& rhs) const {
        return storage != rhs.storage;
    }

    CUTLASS_HOST_DEVICE
    bool operator<=(integer_subbyte const& rhs) const {
        if (kSigned) {
            if (storage & (1 << (kBits - 1))) {
                return !(rhs.storage < storage);
            }
        }
        return storage <= rhs.storage;
    }

    CUTLASS_HOST_DEVICE
    bool operator<(integer_subbyte const& rhs) const {
        if (kSigned) {
            if (storage & (1 << (kBits - 1))) {
                return !(rhs.storage <= storage);
            }
        }
        return storage < rhs.storage;
    }

    CUTLASS_HOST_DEVICE
    bool operator>=(integer_subbyte const& rhs) const { return !(*this < rhs); }

    CUTLASS_HOST_DEVICE
    bool operator>(integer_subbyte const& rhs) const { return !(rhs < *this); }
};


using uint1b_t = integer_subbyte<1, false>;

using int2b_t = integer_subbyte<2, true>;

using uint2b_t = integer_subbyte<2, false>;

using int4b_t = integer_subbyte<4, true>;

using uint4b_t = integer_subbyte<4, false>;


template <>
struct sizeof_bits<uint1b_t> {
    static int const value = 1;
};

template <>
struct sizeof_bits<int2b_t> {
    static int const value = 2;
};

template <>
struct sizeof_bits<uint2b_t> {
    static int const value = 2;
};

template <>
struct sizeof_bits<int4b_t> {
    static int const value = 4;
};

template <>
struct sizeof_bits<uint4b_t> {
    static int const value = 4;
};


namespace platform {

template <>
struct numeric_limits<cutlass::int4b_t> {
    CUTLASS_HOST_DEVICE
    static cutlass::int4b_t const lowest() noexcept { return -8; }
    CUTLASS_HOST_DEVICE
    static cutlass::int4b_t const max() noexcept { return 7; }
};

template <>
struct numeric_limits<cutlass::uint4b_t> {
    CUTLASS_HOST_DEVICE
    static cutlass::uint4b_t const lowest() noexcept { return 0; }
    CUTLASS_HOST_DEVICE
    static cutlass::uint4b_t const max() noexcept { return 15; }
};


}
}
