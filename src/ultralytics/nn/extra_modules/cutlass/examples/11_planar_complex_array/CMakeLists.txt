# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted
#     * Redistributions of source code must retain the above copyright notice, this list of
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       to endorse or promote products derived from this software without specific prior written
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR


cutlass_example_add_executable(
  11_planar_complex_array
  planar_complex_array.cu
)



target_link_libraries(
  11_planar_complex_array
  PRIVATE
  cutlass_lib
  cutlass_tools_util_includes
)

