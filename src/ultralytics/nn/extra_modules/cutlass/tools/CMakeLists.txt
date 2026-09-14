# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted
#     * Redistributions of source code must retain the above copyright notice, this list of
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       to endorse or promote products derived from this software without specific prior written
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR

add_subdirectory(util)

if (CUTLASS_ENABLE_LIBRARY)
  add_subdirectory(library)
endif()
if (CUTLASS_ENABLE_PROFILER)
  if (NOT CUTLASS_ENABLE_LIBRARY)
    message(SEND_ERROR "Build conflict: The CUTLASS profiler requires the CUTLASS library.")
    message(SEND_ERROR "  CUTLASS_ENABLE_PROFILER = ${CUTLASS_ENABLE_PROFILER}")
    message(SEND_ERROR "  CUTLASS_ENABLE_LIBRARY = ${CUTLASS_ENABLE_LIBRARY}")
  else()
    add_subdirectory(profiler)
  endif()
endif()

