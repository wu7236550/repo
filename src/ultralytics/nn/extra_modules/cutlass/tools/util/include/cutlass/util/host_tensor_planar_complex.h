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

#include "cutlass/tensor_ref_planar_complex.h"
#include "cutlass/tensor_view_planar_complex.h"

#include "device_memory.h"

namespace cutlass {


template <
        typename Element_,
        typename Layout_>
class HostTensorPlanarComplex {
public:
    using Element = Element_;

    using Layout = Layout_;

    static int const kRank = Layout::kRank;

    using Index = typename Layout::Index;

    using LongIndex = typename Layout::LongIndex;

    using TensorCoord = typename Layout::TensorCoord;

    using Stride = typename Layout::Stride;

    using TensorRef = TensorRefPlanarComplex<Element, Layout>;

    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using TensorView = TensorViewPlanarComplex<Element, Layout>;

    using ConstTensorView = typename TensorView::ConstTensorView;

    using Reference = typename TensorRef::Reference;

    using ConstReference = typename ConstTensorRef::Reference;

private:

    TensorCoord extent_;

    Layout layout_;

    std::vector<Element> host_;

    device_memory::allocation<Element> device_;

public:

    HostTensorPlanarComplex() {}

    HostTensorPlanarComplex(TensorCoord const& extent,
                            bool device_backed = true) {
        this->reset(extent, Layout::packed(extent), device_backed);
    }

    HostTensorPlanarComplex(TensorCoord const& extent, Layout const& layout,
                            bool device_backed = true) {
        this->reset(extent, layout, device_backed);
    }

    ~HostTensorPlanarComplex() {}

    void reset() {
        extent_ = TensorCoord();
        layout_ = Layout::packed(extent_);

        host_.clear();
        device_.reset();
    }

    void reserve(size_t count,
                 bool device_backed_ =
                         true) {

        device_.reset();
        host_.clear();

        host_.resize(count * 2);

        Element* device_memory = nullptr;
        if (device_backed_) {
            device_memory = device_memory::allocate<Element>(count * 2);
        }
        device_.reset(device_memory, device_backed_ ? count * 2 : 0);
    }

    void reset(TensorCoord const& extent,
               Layout const& layout,
               bool device_backed_ =
                       true) {

        extent_ = extent;
        layout_ = layout;

        reserve(size_t(layout_.capacity(extent_)), device_backed_);
    }

    void reset(TensorCoord const& extent,
               bool device_backed_ =
                       true) {

        reset(extent, Layout::packed(extent), device_backed_);
    }

    void resize(TensorCoord const& extent,
                Layout const& layout,
                bool device_backed_ =
                        true) {

        extent_ = extent;
        layout_ = layout;

        LongIndex new_size = size_t(layout_.capacity(extent_));

        if (static_cast<decltype(host_.size())>(new_size * 2) > host_.size()) {
            reserve(new_size);
        }
    }

    void resize(TensorCoord const& extent,
                bool device_backed_ =
                        true) {

        resize(extent, Layout::packed(extent), device_backed_);
    }

    size_t size() const { return host_.size() / 2; }

    LongIndex capacity() const { return layout_.capacity(extent_); }

    LongIndex imaginary_stride() const { return host_.size() / 2; }

    Element* host_data() { return host_.data(); }

    Element* host_data_imag() { return host_.data() + imaginary_stride(); }

    Element* host_data_ptr_offset(LongIndex ptr_element_offset) {
        return host_data() + ptr_element_offset;
    }

    Element* host_data_imag_ptr_offset(LongIndex ptr_element_offset) {
        return host_data_imag() + ptr_element_offset;
    }

    Reference host_data(LongIndex idx) {
        return PlanarComplexReference<Element>(host_data() + idx,
                                               host_data_imag() + idx);
    }

    Element const* host_data() const { return host_.data(); }

    Element const* host_data_imag() const {
        return host_.data() + imaginary_stride();
    }

    ConstReference host_data(LongIndex idx) const {
        return PlanarComplexReference<Element const>(host_data() + idx,
                                                     host_data_imag() + idx);
    }

    Element* device_data() { return device_.get(); }

    Element* device_data_ptr_offset(LongIndex ptr_element_offset) {
        return device_.get() + ptr_element_offset;
    }

    Element const* device_data() const { return device_.get(); }

    Element const* device_data_ptr_offset(LongIndex ptr_element_offset) const {
        return device_.get() + ptr_element_offset;
    }

    Element* device_data_imag() { return device_.get() + imaginary_stride(); }

    TensorRef host_ref(LongIndex ptr_element_offset = 0) {
        return TensorRef(host_data_ptr_offset(ptr_element_offset), layout_,
                         imaginary_stride());
    }

    cutlass::TensorRef<Element, Layout> host_ref_real() {
        return cutlass::TensorRef<Element, Layout>(host_data(), layout_);
    }

    cutlass::TensorRef<Element, Layout> host_ref_imag() {
        return cutlass::TensorRef<Element, Layout>(
                host_data_ptr_offset(imaginary_stride()), layout_);
    }

    ConstTensorRef host_ref(LongIndex ptr_element_offset = 0) const {
        return ConstTensorRef(host_data_ptr_offset(ptr_element_offset), layout_,
                              imaginary_stride());
    }

    TensorRef device_ref(LongIndex ptr_element_offset = 0) {
        return TensorRef(device_data_ptr_offset(ptr_element_offset), layout_,
                         imaginary_stride());
    }

    ConstTensorRef device_ref(LongIndex ptr_element_offset = 0) const {
        return TensorRef(device_data_ptr_offset(ptr_element_offset), layout_,
                         imaginary_stride());
    }

    cutlass::TensorRef<Element, Layout> device_ref_real() {
        return cutlass::TensorRef<Element, Layout>(device_data(), layout_);
    }

    cutlass::TensorRef<Element, Layout> device_ref_imag() {
        return cutlass::TensorRef<Element, Layout>(
                device_data_ptr_offset(imaginary_stride()), layout_);
    }

    TensorView host_view(LongIndex ptr_element_offset = 0) {
        return TensorView(host_data_ptr_offset(ptr_element_offset), layout_,
                          imaginary_stride(), extent_);
    }

    ConstTensorView host_view(LongIndex ptr_element_offset = 0) const {
        return ConstTensorView(host_data_ptr_offset(ptr_element_offset),
                               layout_, imaginary_stride(), extent_);
    }

    cutlass::TensorView<Element, Layout> host_view_real() {
        return cutlass::TensorView<Element, Layout>(host_data(), layout_,
                                                    extent_);
    }

    cutlass::TensorView<Element, Layout> host_view_imag() {
        return cutlass::TensorView<Element, Layout>(
                host_data_ptr_offset(imaginary_stride()), layout_, extent_);
    }

    TensorView device_view(LongIndex ptr_element_offset = 0) {
        return TensorView(device_data_ptr_offset(ptr_element_offset), layout_,
                          imaginary_stride(), extent_);
    }

    ConstTensorView device_view(LongIndex ptr_element_offset = 0) const {
        return ConstTensorView(device_data_ptr_offset(ptr_element_offset),
                               layout_, imaginary_stride(), extent_);
    }

    cutlass::TensorView<Element, Layout> device_view_real() {
        return cutlass::TensorView<Element, Layout>(device_data(), layout_,
                                                    extent_);
    }

    cutlass::TensorView<Element, Layout> device_view_imag() {
        return cutlass::TensorView<Element, Layout>(
                device_data_ptr_offset(imaginary_stride()), layout_, extent_);
    }

    bool device_backed() const {
        return (device_.get() == nullptr) ? false : true;
    }

    Layout layout() const { return layout_; }

    Stride stride() const { return layout_.stride(); }

    Index stride(int dim) const { return layout_.stride().at(dim); }

    LongIndex offset(TensorCoord const& coord) const { return layout_(coord); }

    Reference at(TensorCoord const& coord) { return host_data(offset(coord)); }

    ConstReference at(TensorCoord const& coord) const {
        return host_data(offset(coord));
    }

    TensorCoord extent() const { return extent_; }

    TensorCoord& extent() { return extent_; }

    void sync_host() {
        if (device_backed()) {
            device_memory::copy_to_host(host_data(), device_data(),
                                        imaginary_stride() * 2);
        }
    }

    void sync_device() {
        if (device_backed()) {
            device_memory::copy_to_device(device_data(), host_data(),
                                          imaginary_stride() * 2);
        }
    }

    void copy_in_device_to_host(
            Element const* ptr_device_real,
            Element const* ptr_device_imag,
            LongIndex count =
                    -1) {

        if (count < 0) {
            count = capacity();
        } else {
            count = __NV_STD_MIN(capacity(), count);
        }

        device_memory::copy_to_host(host_data(), ptr_device_real, count);

        device_memory::copy_to_host(host_data_imag(), ptr_device_imag, count);
    }

    void copy_in_device_to_device(
            Element const* ptr_device_real,
            Element const* ptr_device_imag,
            LongIndex count =
                    -1) {

        if (count < 0) {
            count = capacity();
        } else {
            count = __NV_STD_MIN(capacity(), count);
        }

        device_memory::copy_device_to_device(device_data(), ptr_device_real,
                                             count);

        device_memory::copy_device_to_device(device_data_imag(),
                                             ptr_device_imag, count);
    }

    void copy_in_host_to_device(
            Element const* ptr_host_real,
            Element const* ptr_host_imag,
            LongIndex count =
                    -1) {

        if (count < 0) {
            count = capacity();
        } else {
            count = __NV_STD_MIN(capacity(), count);
        }

        device_memory::copy_to_device(device_data(), ptr_host_real, count);

        device_memory::copy_to_device(device_data_imag(), ptr_host_imag, count);
    }

    void copy_in_host_to_host(
            Element const* ptr_host_real,
            Element const* ptr_host_imag,
            LongIndex count =
                    -1) {

        if (count < 0) {
            count = capacity();
        } else {
            count = __NV_STD_MIN(capacity(), count);
        }

        device_memory::copy_host_to_host(host_data(), ptr_host_real, count);

        device_memory::copy_host_to_host(host_data_imag(), ptr_host_imag,
                                         count);
    }

    void copy_out_device_to_host(
            Element* ptr_host_real,
            Element* ptr_host_imag,
            LongIndex count =
                    -1) const {

        if (count < 0) {
            count = capacity();
        } else {
            count = __NV_STD_MIN(capacity(), count);
        }

        device_memory::copy_to_host(ptr_host_real, device_data(), count);

        device_memory::copy_to_host(ptr_host_imag, device_data_imag(), count);
    }

    void copy_out_device_to_device(
            Element* ptr_device_real,
            Element* ptr_device_imag,
            LongIndex count =
                    -1) const {

        if (count < 0) {
            count = capacity();
        } else {
            count = __NV_STD_MIN(capacity(), count);
        }

        device_memory::copy_device_to_device(ptr_device_real, device_data(),
                                             count);

        device_memory::copy_device_to_device(ptr_device_imag,
                                             device_data_imag(), count);
    }

    void copy_out_host_to_device(
            Element* ptr_device_real,
            Element* ptr_device_imag,
            LongIndex count =
                    -1) const {

        if (count < 0) {
            count = capacity();
        } else {
            count = __NV_STD_MIN(capacity(), count);
        }

        device_memory::copy_to_device(ptr_device_real, host_data(), count);

        device_memory::copy_to_device(ptr_device_imag, host_data_imag(), count);
    }

    void copy_out_host_to_host(Element* ptr_host_real,
                               Element* ptr_host_imag,
                               LongIndex count = -1)
            const {

        if (count < 0) {
            count = capacity();
        } else {
            count = __NV_STD_MIN(capacity(), count);
        }

        device_memory::copy_host_to_host(ptr_host_real, host_data(), count);

        device_memory::copy_host_to_host(ptr_host_imag, host_data_imag(),
                                         count);
    }
};


}
