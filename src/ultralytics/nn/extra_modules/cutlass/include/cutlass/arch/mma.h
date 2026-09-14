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

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"


namespace cutlass {
namespace arch {


struct OpMultiplyAdd;


struct OpMultiplyAddSaturate;


struct OpMultiplyAddFastBF16;


struct OpMultiplyAddFastF16;


struct OpMultiplyAddComplex;


struct OpMultiplyAddGaussianComplex;


struct OpXorPopc;


struct OpClassSimt;


struct OpClassTensorOp;

struct OpClassWmmaTensorOp;


template <
        typename Shape_,
        int kThreads_,
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename LayoutC,
        typename Operator>
struct Mma;


template <
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename LayoutC,
        typename Operator>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, ElementA, LayoutA, ElementB, LayoutB,
           ElementC, LayoutC, Operator> {
    using Shape = gemm::GemmShape<1, 1, 1>;

    CUTLASS_HOST_DEVICE
    void operator()(Array<ElementC, 1>& d, Array<ElementA, 1> const& a,
                    Array<ElementB, 1> const& b, Array<ElementC, 1> const& c) {
        d[0] = a[0] * b[0] + c[0];
    }
};



struct SPFormatType {
    enum Kind { Thread };
};


template <
        typename Shape_,
        int kThreads_,
        typename ElementA,
        typename LayoutA,
        typename ElementB,
        typename LayoutB,
        typename ElementC,
        typename LayoutC,
        typename Operator,
        SPFormatType::Kind SPFormat = SPFormatType::Thread>
struct SparseMma;

}
}



#include "cutlass/arch/mma_sm50.h"
#include "cutlass/arch/mma_sm60.h"
#include "cutlass/arch/mma_sm61.h"
#include "cutlass/arch/mma_sm70.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/arch/mma_sparse_sm80.h"
