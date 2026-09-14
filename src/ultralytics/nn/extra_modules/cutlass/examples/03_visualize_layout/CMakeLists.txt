# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted
#     * Redistributions of source code must retain the above copyright notice, this list of
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       to endorse or promote products derived from this software without specific prior written
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR

set(TEST_COMMAND_00 RowMajor --extent=16,16)
set(TEST_COMMAND_01 "ColumnMajorInterleaved<4>" --extent=32,8 --output-shape=16 --vectorize=4)

cutlass_example_add_executable(
  03_visualize_layout
  visualize_layout.cpp
  register_layout.cu
  TEST_COMMAND_OPTIONS
  TEST_COMMAND_00
  TEST_COMMAND_01
  )

