// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                           \
    [&] {                                                                            \
        if (COND) {                                                                  \
            constexpr bool CONST_NAME = true;                                        \
            return __VA_ARGS__();                                                    \
        } else {                                                                     \
            constexpr bool CONST_NAME = false;                                       \
            return __VA_ARGS__();                                                    \
        }                                                                            \
    }()
