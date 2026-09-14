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

#include <iostream>
#include <stdexcept>
#include <cstdint>

#include "cutlass/library/handle.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/util.h"

namespace cutlass {
namespace library {


Handle::Handle(cudaStream_t stream, size_t workspace_size)
        : provider_(Provider::kCUTLASS),
          stream_(stream),
          workspace_(nullptr),
          workspace_size_(0),
          scalar_pointer_mode_(ScalarPointerMode::kHost),
          last_operation_(nullptr) {
    int device_idx = -1;

    cudaError_t error = cudaGetDevice(&device_idx);
    if (error != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice() failed");
    }

    error = cudaGetDeviceProperties(&device_, device_idx);
    if (error != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    set_workspace_size(workspace_size);

    Singleton::get();
}

Handle::~Handle() {
    if (workspace_) {
        if (workspace_) {
            cudaFree(workspace_);
        }

        workspace_ = nullptr;
        workspace_size_ = 0;
    }
}

Handle::Handle(Handle&& handle) {
    device_ = handle.device_;
    workspace_size_ = handle.workspace_size_;
    workspace_ = handle.workspace_;
    stream_ = handle.stream_;
    scalar_pointer_mode_ = handle.scalar_pointer_mode_;

    handle.workspace_ = nullptr;
    handle.workspace_size_ = 0;
}

Handle& Handle::operator=(Handle&& handle) {
    provider_ = handle.provider_;
    device_ = handle.device_;
    workspace_size_ = handle.workspace_size_;
    workspace_ = handle.workspace_;
    stream_ = handle.stream_;
    scalar_pointer_mode_ = handle.scalar_pointer_mode_;

    handle.workspace_ = nullptr;
    handle.workspace_size_ = 0;

    return *this;
}

int Handle::compute_capability() const {
    return device_.major * 10 + device_.minor;
}

void Handle::set_stream(cudaStream_t stream) {
    stream_ = stream;
}

cudaStream_t Handle::get_stream() const {
    return stream_;
}

Provider Handle::get_provider() const {
    return provider_;
}

void Handle::set_provider(Provider provider) {
    provider_ = provider;
}

size_t Handle::get_workspace_size() const {
    return workspace_size_;
}

void* Handle::get_workspace() const {
    return workspace_;
}

void Handle::set_workspace_size(size_t bytes) {
    if (bytes != workspace_size_) {
        if (workspace_) {
            cudaFree(workspace_);
        }

        workspace_ = nullptr;
        workspace_size_ = bytes;

        if (workspace_size_) {
            cudaError_t error =
                    cudaMalloc((void**)&workspace_, workspace_size_);

            if (error != cudaSuccess) {
                throw std::runtime_error("Failed to allocate workspace");
            }
        }
    }

    if (workspace_) {
        cudaError_t error = cudaMemset(workspace_, 0, workspace_size_);

        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to clear workspace");
        }
    }
}

ScalarPointerMode Handle::get_scalar_pointer_mode() const {
    return scalar_pointer_mode_;
}

void Handle::set_scalar_pointer_mode(ScalarPointerMode mode) {
    scalar_pointer_mode_ = mode;
}

Operation const* Handle::get_last_operation() const {
    return last_operation_;
}


static int maximum_alignment_requirement(GemmDescription const& desc) {
    return std::max(std::max(desc.A.alignment, desc.B.alignment),
                    desc.C.alignment);
}

static int gemm_problem_alignment(
        int M, int N, int K, NumericTypeID element_A, void const* ptr_A,
        int lda, int64_t batch_stride_A, NumericTypeID element_B,
        void const* ptr_B, int ldb, int64_t batch_stride_B,
        NumericTypeID element_C, void const* ptr_C, int ldc,
        int64_t batch_stride_C, void const* ptr_D, int ldd,
        int64_t batch_stride_D, int max_alignment_in_bytes = 16) {
    void const* pointers[] = {ptr_A, ptr_B, ptr_C, ptr_D};

    int64_t extents[] = {M,
                         N,
                         K,
                         lda,
                         ldb,
                         ldc,
                         ldd,
                         batch_stride_A,
                         batch_stride_B,
                         batch_stride_C,
                         batch_stride_D};

    NumericTypeID elements[] = {element_A, element_B, element_C};

    for (; max_alignment_in_bytes > 0; max_alignment_in_bytes /= 2) {
        bool satisfied = true;

        for (void const* ptr : pointers) {
            std::uintptr_t int_ptr = reinterpret_cast<std::uintptr_t>(ptr);

            if (int_ptr % max_alignment_in_bytes) {
                satisfied = false;
                break;
            }
        }

        if (!satisfied) {
            continue;
        }

        int max_element_alignment = 0;

        for (NumericTypeID type_id : elements) {
            int element_alignment =
                    max_alignment_in_bytes * 8 / library::sizeof_bits(type_id);
            max_element_alignment =
                    std::max(max_element_alignment, element_alignment);
        }

        for (int64_t extent : extents) {
            if (extent % max_element_alignment) {
                satisfied = false;
                break;
            }
        }

        if (!satisfied) {
            continue;
        }

        return max_element_alignment;
    }

    return 0;
}

static Operation const* find_gemm_operation(
        GemmOperationFunctionalMap::const_iterator operators_it,
        GemmPreferenceKey const preference_key) {
    auto cc_it = operators_it->second.upper_bound(preference_key);

    if (cc_it == operators_it->second.begin()) {
        return nullptr;
    }

    Operation const* operation = nullptr;

    do {
        --cc_it;

        for (auto const* op : cc_it->second) {
            GemmDescription const& desc =
                    static_cast<GemmDescription const&>(op->description());

            int min_cc = desc.tile_description.minimum_compute_capability;
            int max_cc = desc.tile_description.maximum_compute_capability;

            int op_alignment = maximum_alignment_requirement(desc);

            if ((min_cc <= preference_key.compute_capability) &&
                (preference_key.compute_capability <= max_cc) &&
                (op_alignment <= preference_key.alignment)) {
                operation = op;
                break;
            }
        }
    } while (!operation && cc_it != operators_it->second.begin());

    return operation;
}


Status Handle::gemm(

        int M,
        int N,
        int K,

        NumericTypeID element_compute,

        NumericTypeID element_scalar,

        void const* alpha,

        NumericTypeID element_A,
        LayoutTypeID layout_A,
        ComplexTransform
                transform_A,

        void const* ptr_A,
        int lda,

        NumericTypeID element_B,
        LayoutTypeID layout_B,
        ComplexTransform
                transform_B,

        void const* ptr_B,
        int ldb,

        void const* beta,

        NumericTypeID element_C,

        void const* ptr_C,
        int ldc,

        void* ptr_D,
        int ldd
) {

    GemmFunctionalKey key(provider_, GemmKind::kGemm, element_compute,
                          element_scalar, element_A, layout_A, transform_A,
                          element_B, layout_B, transform_B, element_C);

    auto operators_it =
            Singleton::get().operation_table.gemm_operations.find(key);

    if (operators_it ==
        Singleton::get().operation_table.gemm_operations.end()) {
        return cutlass::Status::kErrorNotSupported;
    }

    if (operators_it->second.empty()) {
        return cutlass::Status::kErrorNotSupported;
    }


    int const kMaximumAlignmentSize = 16;

    int alignment = gemm_problem_alignment(
            M, N, K, element_A, ptr_A, lda, 0, element_B, ptr_B, ldb, 0,
            element_C, ptr_C, ldc, 0, ptr_D, ldd, 0, kMaximumAlignmentSize);


    GemmPreferenceKey preference_key(compute_capability(), alignment);

    Operation const* operation =
            find_gemm_operation(operators_it, preference_key);

    if (!operation) {
        return cutlass::Status::kErrorNotSupported;
    }

    last_operation_ = operation;


    GemmConfiguration configuration{{M, N, K}, lda, ldb, ldc, ldd, 1};

    uint64_t host_workspace_size_needed =
            operation->get_host_workspace_size(&configuration);

    if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
        return cutlass::Status::kErrorNotSupported;
    }

    char host_workspace[kHostWorkspaceSize];

    uint64_t device_workspace_size_needed =
            operation->get_device_workspace_size(&configuration);

    if (uint64_t(workspace_size_) < device_workspace_size_needed) {
        return cutlass::Status::kErrorNotSupported;
    }

    Status status = operation->initialize(&configuration, host_workspace,
                                          workspace_, stream_);

    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    GemmArguments arguments{
            ptr_A, ptr_B, ptr_C, ptr_D, alpha, beta, scalar_pointer_mode_};

    return operation->run(&arguments, host_workspace, workspace_, stream_);
}


Status Handle::gemm_universal(

        GemmUniversalMode mode,

        int M,
        int N,
        int K,

        NumericTypeID element_compute,

        NumericTypeID element_scalar,

        void const* alpha,

        NumericTypeID element_A,
        LayoutTypeID layout_A,
        ComplexTransform
                transform_A,

        void const* ptr_A,
        int lda,

        NumericTypeID element_B,
        LayoutTypeID layout_B,
        ComplexTransform
                transform_B,

        void const* ptr_B,
        int ldb,

        void const* beta,

        NumericTypeID element_C,

        void const* ptr_C,
        int ldc,

        void* ptr_D,
        int ldd,

        int batch_count,

        int64_t batch_stride_A,
        int64_t batch_stride_B,
        int64_t batch_stride_C,
        int64_t batch_stride_D
) {

    GemmFunctionalKey key(provider_, GemmKind::kUniversal, element_compute,
                          element_scalar, element_A, layout_A, transform_A,
                          element_B, layout_B, transform_B, element_C);

    auto operators_it =
            Singleton::get().operation_table.gemm_operations.find(key);

    if (operators_it ==
        Singleton::get().operation_table.gemm_operations.end()) {
        return cutlass::Status::kErrorNotSupported;
    }

    if (operators_it->second.empty()) {
        return cutlass::Status::kErrorNotSupported;
    }


    int const kMaximumAlignmentSize = 16;

    void const* ptr_A_check = ptr_A;
    void const* ptr_B_check = ptr_B;
    void const* ptr_C_check = ptr_C;
    void* ptr_D_check = ptr_D;

    if (mode == GemmUniversalMode::kArray) {
        ptr_A_check = nullptr;
        ptr_B_check = nullptr;
        ptr_C_check = nullptr;
        ptr_D_check = nullptr;
    }

    int alignment = gemm_problem_alignment(
            M, N, K, element_A, ptr_A_check, lda, 0, element_B, ptr_B_check,
            ldb, 0, element_C, ptr_C_check, ldc, 0, ptr_D_check, ldd, 0,
            kMaximumAlignmentSize);


    GemmPreferenceKey preference_key(compute_capability(), alignment);

    Operation const* operation =
            find_gemm_operation(operators_it, preference_key);

    if (!operation) {
        return cutlass::Status::kErrorNotSupported;
    }

    last_operation_ = operation;


    GemmUniversalConfiguration configuration{mode, {M, N, K}, batch_count, lda,
                                             ldb,  ldc,       ldd};

    uint64_t host_workspace_size_needed =
            operation->get_host_workspace_size(&configuration);

    if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
        return cutlass::Status::kErrorNotSupported;
    }

    char host_workspace[kHostWorkspaceSize];

    uint64_t device_workspace_size_needed =
            operation->get_device_workspace_size(&configuration);

    if (uint64_t(workspace_size_) < device_workspace_size_needed) {
        return cutlass::Status::kErrorNotSupported;
    }

    Status status = operation->initialize(&configuration, host_workspace,
                                          workspace_, stream_);

    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    GemmUniversalArguments arguments{ptr_A,
                                     ptr_B,
                                     ptr_C,
                                     ptr_D,
                                     alpha,
                                     beta,
                                     scalar_pointer_mode_,
                                     batch_stride_A,
                                     batch_stride_B,
                                     batch_stride_C,
                                     batch_stride_D};

    return operation->run(&arguments, host_workspace, workspace_, stream_);
}


Status Handle::gemm_planar_complex(

        int M,
        int N,
        int K,

        NumericTypeID element_compute,

        NumericTypeID element_scalar,

        void const* alpha,

        NumericTypeID element_A,
        LayoutTypeID layout_A,
        ComplexTransform
                transform_A,

        void const* ptr_A_real,
        void const* ptr_A_imag,
        int lda_real,
        int lda_imag,

        NumericTypeID element_B,
        LayoutTypeID layout_B,
        ComplexTransform
                transform_B,

        void const* ptr_B_real,
        void const* ptr_B_imag,
        int ldb_real,
        int ldb_imag,

        void const* beta,

        NumericTypeID element_C,

        void const* ptr_C_real,
        void const* ptr_C_imag,
        int ldc_real,
        int ldc_imag,

        void* ptr_D_real,
        void* ptr_D_imag,
        int ldd_real,
        int ldd_imag,

        int batch_count,

        int64_t batch_stride_A_real, int64_t batch_stride_A_imag,

        int64_t batch_stride_B_real, int64_t batch_stride_B_imag,

        int64_t batch_stride_C_real, int64_t batch_stride_C_imag,

        int64_t batch_stride_D_real, int64_t batch_stride_D_imag) {

    GemmFunctionalKey key(provider_, GemmKind::kPlanarComplex, element_compute,
                          element_scalar, element_A, layout_A, transform_A,
                          element_B, layout_B, transform_B, element_C);

    auto operators_it =
            Singleton::get().operation_table.gemm_operations.find(key);

    if (operators_it ==
        Singleton::get().operation_table.gemm_operations.end()) {
        return cutlass::Status::kErrorNotSupported;
    }

    if (operators_it->second.empty()) {
        return cutlass::Status::kErrorNotSupported;
    }


    int const kMaximumAlignmentSize = 16;

    int alignment = std::max(
            gemm_problem_alignment(M, N, K, element_A, ptr_A_real, lda_real,
                                   batch_stride_A_real, element_B, ptr_B_real,
                                   ldb_real, batch_stride_B_real, element_C,
                                   ptr_C_real, ldc_real, batch_stride_C_real,
                                   ptr_D_real, ldd_real, batch_stride_D_real,
                                   kMaximumAlignmentSize),
            gemm_problem_alignment(M, N, K, element_A, ptr_A_imag, lda_imag,
                                   batch_stride_A_imag, element_B, ptr_B_imag,
                                   ldb_imag, batch_stride_B_imag, element_C,
                                   ptr_C_imag, ldc_imag, batch_stride_C_imag,
                                   ptr_D_imag, ldd_imag, batch_stride_D_imag,
                                   kMaximumAlignmentSize));


    GemmPreferenceKey preference_key(compute_capability(), alignment);

    Operation const* operation =
            find_gemm_operation(operators_it, preference_key);

    if (!operation) {
        return cutlass::Status::kErrorNotSupported;
    }

    last_operation_ = operation;


    GemmPlanarComplexConfiguration configuration{GemmUniversalMode::kBatched,
                                                 {M, N, K},
                                                 batch_count,
                                                 lda_real,
                                                 lda_imag,
                                                 ldb_real,
                                                 ldb_imag,
                                                 ldc_real,
                                                 ldc_imag,
                                                 ldd_real,
                                                 ldd_imag};

    uint64_t host_workspace_size_needed =
            operation->get_host_workspace_size(&configuration);

    if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
        return cutlass::Status::kErrorNotSupported;
    }

    char host_workspace[kHostWorkspaceSize];

    uint64_t device_workspace_size_needed =
            operation->get_device_workspace_size(&configuration);

    if (uint64_t(workspace_size_) < device_workspace_size_needed) {
        return cutlass::Status::kErrorNotSupported;
    }

    Status status = operation->initialize(&configuration, host_workspace,
                                          workspace_, stream_);

    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    GemmPlanarComplexArguments arguments{ptr_A_real,
                                         ptr_A_imag,
                                         ptr_B_real,
                                         ptr_B_imag,
                                         ptr_C_real,
                                         ptr_C_imag,
                                         ptr_D_real,
                                         ptr_D_imag,
                                         alpha,
                                         beta,
                                         scalar_pointer_mode_,
                                         batch_stride_A_real,
                                         batch_stride_A_imag,
                                         batch_stride_B_real,
                                         batch_stride_B_imag,
                                         batch_stride_C_real,
                                         batch_stride_C_imag,
                                         batch_stride_D_real,
                                         batch_stride_D_imag};

    return operation->run(&arguments, host_workspace, workspace_, stream_);
}


Status Handle::gemm_planar_complex_array(

        int expected_M,
        int expected_N,
        int expected_K,
        int batch_count,

        int const* M,
        int const* N,
        int const* K,

        NumericTypeID element_compute,

        NumericTypeID element_scalar,

        void const* alpha,

        NumericTypeID element_A,
        LayoutTypeID layout_A,
        ComplexTransform
                transform_A,

        void const* const* ptr_A_real,
        void const* const* ptr_A_imag,

        int lda_real,
        int lda_imag,

        NumericTypeID element_B,
        LayoutTypeID layout_B,
        ComplexTransform
                transform_B,

        void const* const* ptr_B_real,
        void const* const* ptr_B_imag,

        int ldb_real,
        int ldb_imag,

        void const* beta,

        NumericTypeID element_C,

        void const* const* ptr_C_real,
        void const* const* ptr_C_imag,

        int ldc_real,
        int ldc_imag,

        void* const* ptr_D_real,
        void* const* ptr_D_imag,

        int ldd_real,
        int ldd_imag
) {

    GemmFunctionalKey key(provider_, GemmKind::kPlanarComplexArray,
                          element_compute, element_scalar, element_A, layout_A,
                          transform_A, element_B, layout_B, transform_B,
                          element_C);

    auto operators_it =
            Singleton::get().operation_table.gemm_operations.find(key);

    if (operators_it ==
        Singleton::get().operation_table.gemm_operations.end()) {
        return cutlass::Status::kErrorNotSupported;
    }

    if (operators_it->second.empty()) {
        return cutlass::Status::kErrorNotSupported;
    }


    int const kMaximumAlignmentSize = 16;

    int alignment = std::max(
            gemm_problem_alignment(expected_M, expected_N, expected_K,
                                   element_A, nullptr, lda_real, 0, element_B,
                                   nullptr, ldb_real, 0, element_C, nullptr,
                                   ldc_real, 0, nullptr, ldd_real, 0,
                                   kMaximumAlignmentSize),
            gemm_problem_alignment(expected_M, expected_N, expected_K,
                                   element_A, nullptr, lda_imag, 0, element_B,
                                   nullptr, ldb_imag, 0, element_C, nullptr,
                                   ldc_imag, 0, nullptr, ldd_imag, 0,
                                   kMaximumAlignmentSize));


    GemmPreferenceKey preference_key(compute_capability(), alignment);

    Operation const* operation =
            find_gemm_operation(operators_it, preference_key);

    if (!operation) {
        return cutlass::Status::kErrorNotSupported;
    }

    last_operation_ = operation;


    GemmPlanarComplexArrayConfiguration configuration{
            {expected_M, expected_N, expected_K},
            batch_count,
            lda_real,
            lda_imag,
            ldb_real,
            ldb_imag,
            ldc_real,
            ldc_imag,
            ldd_real,
            ldd_imag};

    uint64_t host_workspace_size_needed =
            operation->get_host_workspace_size(&configuration);

    if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
        return cutlass::Status::kErrorNotSupported;
    }

    char host_workspace[kHostWorkspaceSize];

    uint64_t device_workspace_size_needed =
            operation->get_device_workspace_size(&configuration);

    if (uint64_t(workspace_size_) < device_workspace_size_needed) {
        return cutlass::Status::kErrorNotSupported;
    }

    Status status = operation->initialize(&configuration, host_workspace,
                                          workspace_, stream_);

    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    GemmPlanarComplexArrayArguments arguments{
            M,          N,          K,          ptr_A_real,          ptr_A_imag,
            ptr_B_real, ptr_B_imag, ptr_C_real, ptr_C_imag,          ptr_D_real,
            ptr_D_imag, alpha,      beta,       scalar_pointer_mode_};

    return operation->run(&arguments, host_workspace, workspace_, stream_);
}


Operation const* find_conv_operation_for_parallel_reduction(
        Operation const* operation) {
    ConvDescription const& conv_desc =
            static_cast<ConvDescription const&>(operation->description());

    if (conv_desc.tile_description.math_instruction.element_accumulator ==
        conv_desc.C.element) {
        return operation;
    }

    ConvFunctionalKey key(
            library::Provider::kCUTLASS, conv_desc.conv_kind,
            conv_desc.A.element, conv_desc.A.layout, conv_desc.B.element,
            conv_desc.B.layout,
            conv_desc.tile_description.math_instruction.element_accumulator,
            conv_desc.C.layout,
            conv_desc.tile_description.math_instruction.element_accumulator,
            conv_desc.element_epilogue);

    auto conv_operations =
            (conv_desc.kind == OperationKind::kConv2d)
                    ? Singleton::get().operation_table.conv2d_operations
                    : Singleton::get().operation_table.conv3d_operations;

    auto operators_it = conv_operations.find(key);

    if (operators_it == conv_operations.end()) {
        return nullptr;
    }

    if (operators_it->second.empty()) {
        return nullptr;
    }

    ConvPreferenceKey preference_key(
            conv_desc.tile_description.minimum_compute_capability,
            conv_desc.iterator_algorithm);

    auto it = operators_it->second.find(preference_key);

    if (it == operators_it->second.end()) {
        return nullptr;
    }

    for (auto op : it->second) {
        if (op->description().tile_description ==
            operation->description().tile_description) {
            return op;
        }
    }

    return nullptr;
}

}
}

