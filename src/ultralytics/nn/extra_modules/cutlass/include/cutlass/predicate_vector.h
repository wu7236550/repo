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
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#else
#include <assert.h>
#include <stdint.h>
#endif

#include "cutlass/cutlass.h"

#include "cutlass/platform/platform.h"

namespace cutlass {








template <
        int kPredicates_,
        int kPredicatesPerByte_ = 4,
        int kPredicateStart_ = 0>
struct PredicateVector {
    static int const kPredicates = kPredicates_;

    static int const kPredicatesPerByte = kPredicatesPerByte_;

    static int const kPredicateStart = kPredicateStart_;

    static_assert(kPredicatesPerByte <= 8,
                  "kPredicatesPerByte must fit within an actual byte");
    static_assert(kPredicateStart + kPredicatesPerByte <= 8,
                  "The offsetted predicates must fit within an actual byte.");

    typedef uint32_t Storage;

    static int const kBytes =
            (kPredicates + kPredicatesPerByte - 1) / kPredicatesPerByte;

    static int const kWordCount =
            (kBytes + sizeof(Storage) - 1) / sizeof(Storage);

private:

    Storage storageData[kWordCount];


    CUTLASS_HOST_DEVICE void computeStorageOffset(int& word, int& bit,
                                                  int idx) const {
        CUTLASS_ASSERT(idx < kPredicates);

        int byte = (idx / kPredicatesPerByte);
        int bit_offset = (idx % kPredicatesPerByte);

        word = byte / sizeof(Storage);
        int byte_offset = (byte % sizeof(Storage));

        bit = byte_offset * 8 + bit_offset + kPredicateStart;
    }

    CUTLASS_HOST_DEVICE Storage& storage(int word) {
        CUTLASS_ASSERT(word < kWordCount);
        return storageData[word];
    }

    CUTLASS_HOST_DEVICE Storage const& storage(int word) const {
        CUTLASS_ASSERT(word < kWordCount);
        return storageData[word];
    }

public:

    class Iterator {
        PredicateVector& vec_;

        int bit_;

    public:
        CUTLASS_HOST_DEVICE
        Iterator(Iterator const& it) : vec_(it.vec_), bit_(it.bit_) {}

        CUTLASS_HOST_DEVICE
        Iterator(PredicateVector& vec, int _start = 0)
                : vec_(vec), bit_(_start) {}

        CUTLASS_HOST_DEVICE
        Iterator& operator++() {
            ++bit_;
            return *this;
        }

        CUTLASS_HOST_DEVICE
        Iterator& operator+=(int offset) {
            bit_ += offset;
            return *this;
        }

        CUTLASS_HOST_DEVICE
        Iterator& operator--() {
            --bit_;
            return *this;
        }

        CUTLASS_HOST_DEVICE
        Iterator& operator-=(int offset) {
            bit_ -= offset;
            return *this;
        }

        CUTLASS_HOST_DEVICE
        Iterator operator++(int) {
            Iterator ret(*this);
            ret.bit_++;
            return ret;
        }

        CUTLASS_HOST_DEVICE
        Iterator operator--(int) {
            Iterator ret(*this);
            ret.bit_--;
            return ret;
        }

        CUTLASS_HOST_DEVICE
        Iterator operator+(int offset) {
            Iterator ret(*this);
            ret.bit_ += offset;
            return ret;
        }

        CUTLASS_HOST_DEVICE
        Iterator operator-(int offset) {
            ConstIterator ret(*this);
            ret.bit_ -= offset;
            return ret;
        }

        CUTLASS_HOST_DEVICE
        bool operator==(Iterator const& it) const { return bit_ == it.bit_; }

        CUTLASS_HOST_DEVICE
        bool operator!=(Iterator const& it) const { return bit_ != it.bit_; }

        CUTLASS_HOST_DEVICE
        bool get() { return vec_.at(bit_); }

        CUTLASS_HOST_DEVICE
        bool at() const { return vec_.at(bit_); }

        CUTLASS_HOST_DEVICE
        bool operator*() const { return at(); }

        CUTLASS_HOST_DEVICE
        void set(bool value = true) { vec_.set(bit_, value); }
    };

    class ConstIterator {
        PredicateVector const& vec_;

        int bit_;

    public:
        CUTLASS_HOST_DEVICE
        ConstIterator(ConstIterator const& it) : vec_(it.vec_), bit_(it.bit_) {}

        CUTLASS_HOST_DEVICE
        ConstIterator(PredicateVector const& vec, int _start = 0)
                : vec_(vec), bit_(_start) {}

        CUTLASS_HOST_DEVICE
        ConstIterator& operator++() {
            ++bit_;
            return *this;
        }

        CUTLASS_HOST_DEVICE
        ConstIterator& operator+=(int offset) {
            bit_ += offset;
            return *this;
        }

        CUTLASS_HOST_DEVICE
        ConstIterator& operator--() {
            --bit_;
            return *this;
        }

        CUTLASS_HOST_DEVICE
        ConstIterator& operator-=(int offset) {
            bit_ -= offset;
            return *this;
        }

        CUTLASS_HOST_DEVICE
        ConstIterator operator++(int) {
            ConstIterator ret(*this);
            ret.bit_++;
            return ret;
        }

        CUTLASS_HOST_DEVICE
        ConstIterator operator--(int) {
            ConstIterator ret(*this);
            ret.bit_--;
            return ret;
        }

        CUTLASS_HOST_DEVICE
        ConstIterator operator+(int offset) {
            ConstIterator ret(*this);
            ret.bit_ += offset;
            return ret;
        }

        CUTLASS_HOST_DEVICE
        ConstIterator operator-(int offset) {
            ConstIterator ret(*this);
            ret.bit_ -= offset;
            return ret;
        }

        CUTLASS_HOST_DEVICE
        bool operator==(ConstIterator const& it) const {
            return bit_ == it.bit_;
        }

        CUTLASS_HOST_DEVICE
        bool operator!=(ConstIterator const& it) const {
            return bit_ != it.bit_;
        }

        CUTLASS_HOST_DEVICE
        bool get() { return vec_.at(bit_); }

        CUTLASS_HOST_DEVICE
        bool at() const { return vec_.at(bit_); }

        CUTLASS_HOST_DEVICE
        bool operator*() const { return at(); }
    };

    struct TrivialIterator {
        CUTLASS_HOST_DEVICE
        TrivialIterator() {}

        CUTLASS_HOST_DEVICE
        TrivialIterator(Iterator const& it) {}

        CUTLASS_HOST_DEVICE
        TrivialIterator(PredicateVector const& _vec) {}

        CUTLASS_HOST_DEVICE
        TrivialIterator& operator++() { return *this; }

        CUTLASS_HOST_DEVICE
        TrivialIterator operator++(int) { return *this; }

        CUTLASS_HOST_DEVICE
        bool operator*() const { return true; }
    };

public:

    CUTLASS_HOST_DEVICE PredicateVector(bool value = true) { fill(value); }

    CUTLASS_HOST_DEVICE void fill(bool value = true) {
        Storage item = (value ? ~Storage(0) : Storage(0));

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kWordCount; ++i) {
            storage(i) = item;
        }
    }

    CUTLASS_HOST_DEVICE void clear() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kWordCount; ++i) {
            storage(i) = 0;
        }
    }

    CUTLASS_HOST_DEVICE void enable() {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kWordCount; ++i) {
            storage(i) = ~Storage(0);
        }
    }

    CUTLASS_HOST_DEVICE bool operator[](int idx) const { return at(idx); }

    CUTLASS_HOST_DEVICE bool at(int idx) const {
        int bit, word;
        computeStorageOffset(word, bit, idx);

        return ((storage(word) >> bit) & 1);
    }

    CUTLASS_HOST_DEVICE void set(int idx, bool value = true) {
        int bit, word;
        computeStorageOffset(word, bit, idx);

        Storage disable_mask = (~(Storage(1) << bit));
        Storage enable_mask = (Storage(value) << bit);

        storage(word) = ((storage(word) & disable_mask) | enable_mask);
    }

    CUTLASS_HOST_DEVICE PredicateVector& operator&=(
            PredicateVector const& predicates) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kWordCount; ++i) {
            storage(i) = (storage(i) & predicates.storage(i));
        }
        return *this;
    }

    CUTLASS_HOST_DEVICE PredicateVector& operator|=(
            PredicateVector const& predicates) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kWordCount; ++i) {
            storage(i) = (storage(i) | predicates.storage(i));
        }
        return *this;
    }

    CUTLASS_HOST_DEVICE bool is_zero() const {
        Storage mask(0);
        for (int byte = 0; byte < sizeof(Storage); ++byte) {
            Storage byte_mask =
                    (((1 << kPredicatesPerByte) - 1) << kPredicateStart);
            mask |= (byte_mask << (byte * 8));
        }
        uint32_t result = 0;
        for (int word = 0; word < kWordCount; ++word) {
            result |= storage(word);
        }
        return result == 0;
    }

    CUTLASS_DEVICE
    Iterator begin() { return Iterator(*this); }

    CUTLASS_DEVICE
    Iterator end() { return Iterator(*this, kPredicates); }

    CUTLASS_DEVICE
    ConstIterator const_begin() const { return ConstIterator(*this); }

    CUTLASS_DEVICE
    ConstIterator const_end() const {
        return ConstIterator(*this, kPredicates);
    }
};


}
