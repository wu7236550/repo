# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted
#     * Redistributions of source code must retain the above copyright notice, this list of
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       to endorse or promote products derived from this software without specific prior written
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR

add_library(cutlass_tools_util_includes INTERFACE)
add_library(nvidia::cutlass::tools::util ALIAS cutlass_tools_util_includes)
set_target_properties(cutlass_tools_util_includes PROPERTIES EXPORT_NAME tools::util)

target_include_directories(
  cutlass_tools_util_includes
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}>
  )

target_link_libraries(
  cutlass_tools_util_includes
  INTERFACE
 	$<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:cublas>
  )

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  )

install(
  TARGETS cutlass_tools_util_includes
  EXPORT NvidiaCutlass
  )

