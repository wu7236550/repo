/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/


#include "selective_scan_bwd_kernel.cuh"

template void selective_scan_bwd_cuda<at::Half, complex_t>(SSMParamsBwd &params, cudaStream_t stream);