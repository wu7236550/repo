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


#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"


namespace cutlass {
namespace library {


enum class LayoutTypeID {
    kUnknown,
    kColumnMajor,
    kRowMajor,
    kColumnMajorInterleavedK2,
    kRowMajorInterleavedK2,
    kColumnMajorInterleavedK4,
    kRowMajorInterleavedK4,
    kColumnMajorInterleavedK16,
    kRowMajorInterleavedK16,
    kColumnMajorInterleavedK32,
    kRowMajorInterleavedK32,
    kColumnMajorInterleavedK64,
    kRowMajorInterleavedK64,
    kTensorNCHW,
    kTensorNCDHW,
    kTensorNHWC,
    kTensorNDHWC,
    kTensorNC32HW32,
    kTensorC32RSK32,
    kTensorNC64HW64,
    kTensorC64RSK64,
    kInvalid
};

enum class NumericTypeID {
    kUnknown,
    kVoid,
    kB1,
    kU2,
    kU4,
    kU8,
    kU16,
    kU32,
    kU64,
    kS2,
    kS4,
    kS8,
    kS16,
    kS32,
    kS64,
    kF16,
    kBF16,
    kTF32,
    kF32,
    kF64,
    kCF16,
    kCBF16,
    kCF32,
    kCTF32,
    kCF64,
    kCS2,
    kCS4,
    kCS8,
    kCS16,
    kCS32,
    kCS64,
    kCU2,
    kCU4,
    kCU8,
    kCU16,
    kCU32,
    kCU64,
    kInvalid
};

enum class ComplexTransform { kNone, kConjugate, kInvalid };

enum class Provider {
    kNone,
    kCUTLASS,
    kReferenceHost,
    kReferenceDevice,
    kCUBLAS,
    kCUDNN,
    kInvalid
};


enum class OperationKind {
    kGemm,
    kConv2d,
    kConv3d,
    kEqGemm,
    kSparseGemm,
    kReduction,
    kInvalid
};

enum class ScalarPointerMode { kHost, kDevice, kInvalid };

enum class SplitKMode { kNone, kSerial, kParallel, kParallelSerial, kInvalid };

enum class OpcodeClassID {
    kSimt,
    kTensorOp,
    kWmmaTensorOp,
    kSparseTensorOp,
    kInvalid
};

enum class MathOperationID {
    kAdd,
    kMultiplyAdd,
    kMultiplyAddSaturate,
    kMultiplyAddFastBF16,
    kMultiplyAddFastF16,
    kMultiplyAddComplex,
    kMultiplyAddGaussianComplex,
    kXorPopc,
    kInvalid
};


enum class GemmKind {
    kGemm,
    kSparse,
    kUniversal,
    kPlanarComplex,
    kPlanarComplexArray,
    kInvalid
};

using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

enum class ConvKind { kUnknown, kFprop, kDgrad, kWgrad, kInvalid };

enum class ConvModeID { kCrossCorrelation, kConvolution, kInvalid };

enum class IteratorAlgorithmID { kNone, kAnalytic, kOptimized, kInvalid };

enum class EpilogueKind {
    kUnknown,
    kConversion,
    kLinearCombination,
    kLinearCombinationClamp,
    kLinearCombinationPlanarComplex,
    kLinearCombinationRelu,
    kLinearCombinationSigmoid,
    kInvalid
};


struct MathInstructionDescription {
    cutlass::gemm::GemmCoord instruction_shape;

    NumericTypeID element_accumulator;

    OpcodeClassID opcode_class;

    MathOperationID math_operation;


    MathInstructionDescription(
            cutlass::gemm::GemmCoord instruction_shape =
                    cutlass::gemm::GemmCoord(),
            NumericTypeID element_accumulator = NumericTypeID::kInvalid,
            OpcodeClassID opcode_class = OpcodeClassID::kInvalid,
            MathOperationID math_operation = MathOperationID::kMultiplyAdd)
            : instruction_shape(instruction_shape),
              element_accumulator(element_accumulator),
              opcode_class(opcode_class),
              math_operation(math_operation) {}

    inline bool operator==(MathInstructionDescription const& rhs) const {
        return ((instruction_shape == rhs.instruction_shape) &&
                (element_accumulator == rhs.element_accumulator) &&
                (opcode_class == rhs.opcode_class) &&
                (math_operation == rhs.math_operation));
    }

    inline bool operator!=(MathInstructionDescription const& rhs) const {
        return !(*this == rhs);
    }
};

struct TileDescription {
    cutlass::gemm::GemmCoord threadblock_shape;

    int threadblock_stages;

    cutlass::gemm::GemmCoord warp_count;

    MathInstructionDescription math_instruction;

    int minimum_compute_capability;

    int maximum_compute_capability;


    TileDescription(
            cutlass::gemm::GemmCoord threadblock_shape =
                    cutlass::gemm::GemmCoord(),
            int threadblock_stages = 0,
            cutlass::gemm::GemmCoord warp_count = cutlass::gemm::GemmCoord(),
            MathInstructionDescription math_instruction =
                    MathInstructionDescription(),
            int minimum_compute_capability = 0,
            int maximum_compute_capability = 0)
            : threadblock_shape(threadblock_shape),
              threadblock_stages(threadblock_stages),
              warp_count(warp_count),
              math_instruction(math_instruction),
              minimum_compute_capability(minimum_compute_capability),
              maximum_compute_capability(maximum_compute_capability) {}

    inline bool operator==(TileDescription const& rhs) const {
        return ((threadblock_shape == rhs.threadblock_shape) &&
                (threadblock_stages == rhs.threadblock_stages) &&
                (warp_count == rhs.warp_count) &&
                (math_instruction == rhs.math_instruction) &&
                (minimum_compute_capability ==
                 rhs.minimum_compute_capability) &&
                (maximum_compute_capability == rhs.maximum_compute_capability));
    }

    inline bool operator!=(TileDescription const& rhs) const {
        return !(*this == rhs);
    }
};

struct OperationDescription {
    char const* name;

    Provider provider;

    OperationKind kind;

    TileDescription tile_description;

    OperationDescription(
            char const* name = "unknown",
            Provider Provider = Provider::kInvalid,
            OperationKind kind = OperationKind::kInvalid,
            TileDescription const& tile_description = TileDescription())
            : name(name), kind(kind), tile_description(tile_description) {}
};

struct TensorDescription {
    NumericTypeID element;

    LayoutTypeID layout;

    int alignment;

    int log_extent_range;

    int log_stride_range;


    TensorDescription(NumericTypeID element = NumericTypeID::kInvalid,
                      LayoutTypeID layout = LayoutTypeID::kInvalid,
                      int alignment = 1, int log_extent_range = 24,
                      int log_stride_range = 24)
            : element(element),
              layout(layout),
              alignment(alignment),
              log_extent_range(log_extent_range),
              log_stride_range(log_stride_range) {}
};


struct GemmDescription : public OperationDescription {
    GemmKind gemm_kind;

    TensorDescription A;

    TensorDescription B;

    TensorDescription C;

    TensorDescription E;

    NumericTypeID element_epilogue;

    SplitKMode split_k_mode;

    ComplexTransform transform_A;

    ComplexTransform transform_B;


    GemmDescription(GemmKind gemm_kind = GemmKind::kGemm,
                    TensorDescription const& A = TensorDescription(),
                    TensorDescription const& B = TensorDescription(),
                    TensorDescription const& C = TensorDescription(),
                    NumericTypeID element_epilogue = NumericTypeID::kInvalid,
                    SplitKMode split_k_mode = SplitKMode::kNone,
                    ComplexTransform transform_A = ComplexTransform::kNone,
                    ComplexTransform transform_B = ComplexTransform::kNone)
            : gemm_kind(gemm_kind),
              A(A),
              B(B),
              C(C),
              element_epilogue(element_epilogue),
              split_k_mode(split_k_mode),
              transform_A(transform_A),
              transform_B(transform_B) {}
};


struct SparseGemmDescription : public GemmDescription {
    SparseGemmDescription(
            GemmKind gemm_kind = GemmKind::kGemm,
            TensorDescription const& A = TensorDescription(),
            TensorDescription const& B = TensorDescription(),
            TensorDescription const& C = TensorDescription(),
            TensorDescription const& E = TensorDescription(),
            NumericTypeID element_epilogue = NumericTypeID::kInvalid,
            SplitKMode split_k_mode = SplitKMode::kNone,
            ComplexTransform transform_A = ComplexTransform::kNone,
            ComplexTransform transform_B = ComplexTransform::kNone)
            : GemmDescription(gemm_kind, A, B, C, element_epilogue,
                              split_k_mode, transform_A, transform_B) {
        this->E = E;
    }
};

struct ReductionDescription : public OperationDescription {
    NumericTypeID element_workspace;

    NumericTypeID element_output;

    NumericTypeID element_epilogue;
};


struct ConvDescription : public OperationDescription {
    int conv_dim;

    ConvKind conv_kind;

    IteratorAlgorithmID iterator_algorithm;

    TensorDescription A;

    TensorDescription B;

    TensorDescription C;

    NumericTypeID element_epilogue;

    TensorDescription activation() const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return A;
            case library::ConvKind::kDgrad:
                return C;
            case library::ConvKind::kWgrad:
                return B;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    TensorDescription filter() const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return B;
            case library::ConvKind::kDgrad:
                return B;
            case library::ConvKind::kWgrad:
                return C;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    TensorDescription output() const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return C;
            case library::ConvKind::kDgrad:
                return A;
            case library::ConvKind::kWgrad:
                return A;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }
};


class Operation {
public:
    virtual ~Operation() {}

    virtual OperationDescription const& description() const = 0;

    virtual Status can_implement(void const* configuration,
                                 void const* arguments) const = 0;

    virtual uint64_t get_host_workspace_size(
            void const* configuration) const = 0;

    virtual uint64_t get_device_workspace_size(
            void const* configuration) const = 0;

    virtual Status initialize(void const* configuration, void* host_workspace,
                              void* device_workspace = nullptr,
                              cudaStream_t stream = nullptr) const = 0;

    virtual Status run(void const* arguments, void* host_workspace,
                       void* device_workspace = nullptr,
                       cudaStream_t stream = nullptr) const = 0;
};


struct GemmConfiguration {
    gemm::GemmCoord problem_size;

    int64_t lda;

    int64_t ldb;

    int64_t ldc;

    int64_t ldd;

    int split_k_slices;
};

struct GemmArguments {
    void const* A;

    void const* B;

    void const* C;

    void* D;

    void const* alpha;

    void const* beta;

    ScalarPointerMode pointer_mode;
};



struct GemmBatchedConfiguration {
    gemm::GemmCoord problem_size;

    int64_t lda;

    int64_t ldb;

    int64_t ldc;

    int64_t ldd;

    int64_t batch_stride_A;

    int64_t batch_stride_B;

    int64_t batch_stride_C;

    int64_t batch_stride_D;

    int batch_count;
};

using GemmBatchedArguments = GemmArguments;



struct GemmArrayConfiguration {
    gemm::GemmCoord problem_size;

    int64_t lda;

    int64_t ldb;

    int64_t ldc;

    int64_t ldd;

    int batch_count;
};

struct GemmArrayArguments {
    void const* const* A;
    void const* const* B;
    void const* const* C;
    void* const* D;
    void const* alpha;
    void const* beta;
    ScalarPointerMode pointer_mode;
};



struct GemmUniversalConfiguration {
    GemmUniversalMode mode;
    gemm::GemmCoord problem_size;
    int batch_count;

    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;
};

struct GemmUniversalArguments {
    void const* A;
    void const* B;
    void const* C;
    void* D;

    void const* alpha;
    void const* beta;
    ScalarPointerMode pointer_mode;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;
};



struct GemmPlanarComplexConfiguration {
    GemmUniversalMode mode;
    gemm::GemmCoord problem_size;
    int batch_count;

    int64_t lda_real;
    int64_t lda_imag;

    int64_t ldb_real;
    int64_t ldb_imag;

    int64_t ldc_real;
    int64_t ldc_imag;

    int64_t ldd_real;
    int64_t ldd_imag;
};

struct GemmPlanarComplexArguments {
    void const* A_real;
    void const* A_imag;

    void const* B_real;
    void const* B_imag;

    void const* C_real;
    void const* C_imag;

    void* D_real;
    void* D_imag;

    void const* alpha;
    void const* beta;
    ScalarPointerMode pointer_mode;

    int64_t batch_stride_A_real;
    int64_t batch_stride_A_imag;

    int64_t batch_stride_B_real;
    int64_t batch_stride_B_imag;

    int64_t batch_stride_C_real;
    int64_t batch_stride_C_imag;

    int64_t batch_stride_D_real;
    int64_t batch_stride_D_imag;
};


struct GemmPlanarComplexArrayConfiguration {
    gemm::GemmCoord problem_size;
    int batch_count;

    int64_t lda_real;
    int64_t lda_imag;

    int64_t ldb_real;
    int64_t ldb_imag;

    int64_t ldc_real;
    int64_t ldc_imag;

    int64_t ldd_real;
    int64_t ldd_imag;
};

struct GemmPlanarComplexArrayArguments {
    int const* M;
    int const* N;
    int const* K;

    void const* const* A_real;
    void const* const* A_imag;
    void const* const* B_real;
    void const* const* B_imag;
    void const* const* C_real;
    void const* const* C_imag;
    void* const* D_real;
    void* const* D_imag;

    void const* alpha;
    void const* beta;
    ScalarPointerMode pointer_mode;
};


struct SparseGemmConfiguration {
    GemmUniversalMode mode;
    gemm::GemmCoord problem_size;
    int batch_count;

    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;
    int64_t lde;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;
    int64_t batch_stride_E;
};

struct SparseGemmArguments {
    void const* A;
    void const* B;
    void const* C;
    void* D;
    void const* E;

    void const* alpha;
    void const* beta;
    ScalarPointerMode pointer_mode;
};



struct Conv2dConfiguration {
    conv::SplitKMode split_k_mode;

    conv::Conv2dProblemSize problem_size;

    layout::TensorNHWC layout_activations;

    layout::TensorNHWC layout_filters;

    layout::TensorNHWC layout_source;

    layout::TensorNHWC layout_output;


    layout::TensorNHWC layout_a(library::ConvKind const& conv_kind) const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return layout_activations;
            case library::ConvKind::kDgrad:
                return layout_output;
            case library::ConvKind::kWgrad:
                return layout_output;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    layout::TensorNHWC layout_b(library::ConvKind const& conv_kind) const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return layout_filters;
            case library::ConvKind::kDgrad:
                return layout_filters;
            case library::ConvKind::kWgrad:
                return layout_activations;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    layout::TensorNHWC layout_c(library::ConvKind const& conv_kind) const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return layout_output;
            case library::ConvKind::kDgrad:
                return layout_activations;
            case library::ConvKind::kWgrad:
                return layout_filters;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }
};

struct Conv3dConfiguration {
    conv::SplitKMode split_k_mode;

    conv::Conv3dProblemSize problem_size;

    layout::TensorNDHWC layout_activations;

    layout::TensorNDHWC layout_filters;

    layout::TensorNDHWC layout_source;

    layout::TensorNDHWC layout_output;


    layout::TensorNDHWC layout_a(library::ConvKind const& conv_kind) const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return layout_activations;
            case library::ConvKind::kDgrad:
                return layout_output;
            case library::ConvKind::kWgrad:
                return layout_output;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    layout::TensorNDHWC layout_b(library::ConvKind const& conv_kind) const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return layout_filters;
            case library::ConvKind::kDgrad:
                return layout_filters;
            case library::ConvKind::kWgrad:
                return layout_activations;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }

    layout::TensorNDHWC layout_c(library::ConvKind const& conv_kind) const {
        switch (conv_kind) {
            case library::ConvKind::kFprop:
                return layout_output;
            case library::ConvKind::kDgrad:
                return layout_activations;
            case library::ConvKind::kWgrad:
                return layout_filters;
            default:
                throw std::runtime_error(
                        "Invalid Conv Operator (fprop, dgrad, wgrad)");
        }
    }
};

struct ConvArguments {
    void const* A;

    void const* B;

    void const* C;

    void* D;

    void const* alpha;

    void const* beta;

    ScalarPointerMode pointer_mode;
};


struct ReductionConfiguration {
    MatrixCoord problem_size;

    int partitions;

    int64_t partition_stride;

    int64_t ldw;

    int64_t lds;

    int64_t ldd;
};

struct ReductionArguments {
    void const* workspace;

    void const* source;

    void* destination;

    void* reference;

    void const* alpha;

    void const* beta;

    ScalarPointerMode pointer_mode;
};

}
}

