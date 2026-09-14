# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted
#     * Redistributions of source code must retain the above copyright notice, this list of
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       to endorse or promote products derived from this software without specific prior written
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR

set(CUTLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(cutlass_examples)
add_custom_target(test_examples)

function(cutlass_example_add_executable NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS})

  add_dependencies(cutlass_examples ${NAME})

  target_link_libraries(
    ${NAME}
    PRIVATE
    CUTLASS
    cutlass_tools_util_includes
    )

  target_include_directories(
    ${NAME}
    PRIVATE
    ${CUTLASS_EXAMPLES_COMMON_SOURCE_DIR}
    )

  install(
    TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

  cutlass_add_executable_tests(
    test_examples_${NAME} ${NAME}
    DEPENDS ${__DEPENDS}
    DEPENDEES test_examples ${__DEPENDEES}
    TEST_COMMAND_OPTIONS ${__TEST_COMMAND_OPTIONS}
    DISABLE_EXECUTABLE_INSTALL_RULE
    )

endfunction()

foreach(EXAMPLE
  00_basic_gemm
  01_cutlass_utilities
  02_dump_reg_shmem
  03_visualize_layout
  04_tile_iterator
  05_batched_gemm
  06_splitK_gemm
  07_volta_tensorop_gemm
  08_turing_tensorop_gemm
  09_turing_tensorop_conv2dfprop
  10_planar_complex
  11_planar_complex_array
  12_gemm_bias_relu
  13_fused_two_gemms
  14_ampere_tf32_tensorop_gemm
  15_ampere_sparse_tensorop_gemm
  16_large_depthwise_conv2dfprop
  17_large_depthwise_conv2ddgrad
  18_large_depthwise_conv2dwgrad
  22_ampere_tensorop_conv2dfprop
  )

  add_subdirectory(${EXAMPLE})

endforeach()
