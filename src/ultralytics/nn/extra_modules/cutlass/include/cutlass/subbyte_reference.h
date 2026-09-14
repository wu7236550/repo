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

#include "cutlass/numeric_types.h"

namespace cutlass {


template <typename Element_,
          typename Storage_ = uint8_t
          >
class ConstSubbyteReference {
public:
    using Element = Element_;
    using Storage = Storage_;
    using StoragePointer = Storage const*;

    static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value,
                  "Size of Element must not be greater than Storage.");

    static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value),
                  "Storage must be divisible by Element");

private:
    int const kElementsPerVector =
            sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

    Storage const kMask =
            ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value)
                     ? (Storage(1) << sizeof_bits<Element>::value) - Storage(1)
                     : ~Storage(0));

private:
    StoragePointer ptr_;

    int offset_;

public:
    CUTLASS_HOST_DEVICE
    ConstSubbyteReference() : ptr_(nullptr), offset_(0) {}

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference(Element const* ptr,
                          int64_t offset
                          )
            : ptr_(reinterpret_cast<StoragePointer>(ptr)), offset_(0) {
        int64_t offset_in_vectors = offset / kElementsPerVector;
        int64_t offset_in_elements = offset % kElementsPerVector;

        ptr_ += offset_in_vectors;
        offset_ = int(offset_in_elements);
    }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference(Element* ptr = nullptr)
            : ConstSubbyteReference(ptr, 0) {}

    CUTLASS_HOST_DEVICE
    StoragePointer storage_pointer() const { return ptr_; }

    CUTLASS_HOST_DEVICE
    int element_offset() const { return offset_; }

    CUTLASS_HOST_DEVICE
    Element get() const {
        Storage item = Storage(
                (*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
        return reinterpret_cast<Element const&>(item);
    }

    CUTLASS_HOST_DEVICE
    operator Element() const { return get(); }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference& operator+=(int offset) {
        offset += offset_;

        int offset_in_vectors = offset / kElementsPerVector;
        int offset_in_elements = offset % kElementsPerVector;

        ptr_ += offset_in_vectors;
        offset_ = offset_in_elements;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference& operator+=(long long offset) {
        offset += offset_;

        long long offset_in_vectors = offset / kElementsPerVector;
        int offset_in_elements = int(offset % kElementsPerVector);

        ptr_ += offset_in_vectors;
        offset_ = offset_in_elements;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference& operator-=(int offset) {
        int offset_in_vectors = offset / kElementsPerVector;
        int offset_in_elements = offset % kElementsPerVector;

        ptr_ -= offset_in_vectors;
        offset_ -= offset_in_elements;

        if (offset_ < 0) {
            offset_ += kElementsPerVector;
            --ptr_;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference& operator-=(long long offset) {
        long long offset_in_vectors = offset / kElementsPerVector;
        int offset_in_elements = int(offset % kElementsPerVector);

        ptr_ -= offset_in_vectors;
        offset_ -= offset_in_elements;

        if (offset_ < 0) {
            offset_ += kElementsPerVector;
            --ptr_;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference operator+(int offset) const {
        ConstSubbyteReference ref(ptr_, offset_);
        ref += offset;

        return ref;
    }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference operator+(long long offset) const {
        ConstSubbyteReference ref(ptr_, offset_);
        ref += offset;

        return ref;
    }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference operator-(int offset) const {
        ConstSubbyteReference ref(ptr_, offset_);
        ref -= offset;

        return ref;
    }

    CUTLASS_HOST_DEVICE
    ConstSubbyteReference operator-=(long long offset) const {
        ConstSubbyteReference ref(ptr_, offset_);
        ref -= offset;

        return ref;
    }

    CUTLASS_HOST_DEVICE
    ptrdiff_t operator-(ConstSubbyteReference ref) const {
        return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
    }

    CUTLASS_HOST_DEVICE
    explicit operator int() const { return int(get()); }

    CUTLASS_HOST_DEVICE
    explicit operator int64_t() const { return int64_t(get()); }

    CUTLASS_HOST_DEVICE
    explicit operator uint64_t() const { return uint64_t(get()); }

    CUTLASS_HOST_DEVICE
    explicit operator float() const { return float(get()); }

    CUTLASS_HOST_DEVICE
    explicit operator double() const { return double(get()); }
};

template <typename Element_,
          typename Storage_ = uint8_t
          >
class SubbyteReference {
public:
    using Element = Element_;
    using Storage = Storage_;
    using StoragePointer = Storage*;

    static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value,
                  "Size of Element must not be greater than Storage.");

    static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value),
                  "Storage must be divisible by Element");

private:
    int const kElementsPerVector =
            sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

    Storage const kMask =
            ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value)
                     ? (Storage(1) << sizeof_bits<Element>::value) - Storage(1)
                     : ~Storage(0));

private:
    StoragePointer ptr_;

    int offset_;

public:
    CUTLASS_HOST_DEVICE
    SubbyteReference() : ptr_(nullptr), offset_(0) {}

    CUTLASS_HOST_DEVICE
    SubbyteReference(Element* ptr,
                     int64_t offset
                     )
            : ptr_(reinterpret_cast<StoragePointer>(ptr)), offset_(0) {
        int64_t offset_in_vectors = offset / kElementsPerVector;
        int64_t offset_in_elements = offset % kElementsPerVector;

        ptr_ += offset_in_vectors;
        offset_ = int(offset_in_elements);
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference(Element* ptr = nullptr) : SubbyteReference(ptr, 0) {}

    CUTLASS_HOST_DEVICE
    StoragePointer storage_pointer() const { return ptr_; }

    CUTLASS_HOST_DEVICE
    int element_offset() const { return offset_; }

    CUTLASS_HOST_DEVICE
    Element get() const {
        Storage item = Storage(
                (*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
        return reinterpret_cast<Element const&>(item);
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference& set(Element const& x) {
        Storage item = (reinterpret_cast<Storage const&>(x) & kMask);

        Storage kUpdateMask =
                Storage(~(kMask << (offset_ * sizeof_bits<Element>::value)));
        *ptr_ = Storage(
                (*ptr_ & kUpdateMask) |
                Storage(item << (offset_ * sizeof_bits<Element>::value)));

        return *this;
    }

    CUTLASS_HOST_DEVICE
    operator Element() const { return get(); }

    CUTLASS_HOST_DEVICE
    SubbyteReference& operator=(Element const& x) { return set(x); }

    CUTLASS_HOST_DEVICE
    SubbyteReference& operator=(SubbyteReference const& x) {
        return set(x.get());
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference& operator=(
            ConstSubbyteReference<Element, Storage> const& x) {
        return set(x.get());
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference& operator+=(int offset) {
        offset += offset_;

        int offset_in_vectors = offset / kElementsPerVector;
        int offset_in_elements = offset % kElementsPerVector;

        ptr_ += offset_in_vectors;
        offset_ = offset_in_elements;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference& operator+=(long long offset) {
        offset += offset_;

        long long offset_in_vectors = offset / kElementsPerVector;
        int offset_in_elements = int(offset % kElementsPerVector);

        ptr_ += offset_in_vectors;
        offset_ = offset_in_elements;

        return *this;
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference& operator-=(int offset) {
        int offset_in_vectors = offset / kElementsPerVector;
        int offset_in_elements = offset % kElementsPerVector;

        ptr_ -= offset_in_vectors;
        offset_ -= offset_in_elements;

        if (offset_ < 0) {
            offset_ += kElementsPerVector;
            --ptr_;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference& operator-=(long long offset) {
        long long offset_in_vectors = offset / kElementsPerVector;
        int offset_in_elements = int(offset % kElementsPerVector);

        ptr_ -= offset_in_vectors;
        offset_ -= offset_in_elements;

        if (offset_ < 0) {
            offset_ += kElementsPerVector;
            --ptr_;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference operator+(int offset) const {
        SubbyteReference ref(ptr_, offset_);
        ref += offset;

        return ref;
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference operator+(long long offset) const {
        SubbyteReference ref(ptr_, offset_);
        ref += offset;

        return ref;
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference operator-(int offset) const {
        SubbyteReference ref(ptr_, offset_);
        ref -= offset;

        return ref;
    }

    CUTLASS_HOST_DEVICE
    SubbyteReference operator-=(long long offset) const {
        SubbyteReference ref(ptr_, offset_);
        ref -= offset;

        return ref;
    }

    CUTLASS_HOST_DEVICE
    ptrdiff_t operator-(SubbyteReference ref) const {
        return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
    }

    CUTLASS_HOST_DEVICE
    explicit operator int() const { return int(get()); }

    CUTLASS_HOST_DEVICE
    explicit operator int64_t() const { return int64_t(get()); }

    CUTLASS_HOST_DEVICE
    explicit operator uint64_t() const { return uint64_t(get()); }

    CUTLASS_HOST_DEVICE
    explicit operator float() const { return float(get()); }

    CUTLASS_HOST_DEVICE
    explicit operator double() const { return double(get()); }
};


template <typename Element, bool subbyte = (sizeof_bits<Element>::value < 8)>
struct ReferenceFactory;

template <typename Element>
struct ReferenceFactory<Element, false> {
    CUTLASS_HOST_DEVICE
    static Element& get(Element* ptr, int64_t offset) { return ptr[offset]; }

    CUTLASS_HOST_DEVICE
    static Element const& get(Element const* ptr, int64_t offset) {
        return ptr[offset];
    }
};

template <typename Element>
struct ReferenceFactory<Element, true> {
    CUTLASS_HOST_DEVICE
    static SubbyteReference<Element> get(Element* ptr, int64_t offset) {
        return SubbyteReference<Element>(ptr, offset);
    }

    CUTLASS_HOST_DEVICE
    static ConstSubbyteReference<Element> get(Element const* ptr,
                                              int64_t offset) {
        return ConstSubbyteReference<Element>(ptr, offset);
    }
};


}
