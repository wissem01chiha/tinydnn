/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cstddef>
#include <cstdint>

/**
 * define if you want to use intel TBB library
 */
#define USE_TBB

/**
 * define to enable avx vectorization
 */
#define USE_AVX

/**
 * define to enable sse2 vectorization
 */
#define USE_SSE

/**
 * define to enable OMP parallelization
 */
#define USE_OMP

/**
 * define to enable Grand Central Dispatch parallelization
 */
// #define CNN_USE_GCD

/**
 * define to use exceptions
 */
#define USE_EXCEPTIONS

/**
 * comment out if you want tiny-dnn to be quiet
 */
#define USE_STDOUT

#define SINGLE_THREAD

/**
 * disable serialization/deserialization function
 * You can uncomment this to speedup compilation & linking time,
 * if you don't use network::save / network::load functions.
 **/
#define NO_SERIALIZATION

/**
 * Enable Gemmlowp support.
 **/
#ifdef USE_GEMMLOWP
#if !defined(_MSC_VER) && !defined(_WIN32) && !defined(WIN32)
#define USE_GEMMLOWP  // gemmlowp doesn't support MSVC/mingw
#endif
#endif  // USE_GEMMLOWP

/**
 * number of task in batch-gradient-descent.
 * @todo automatic optimization
 */
#ifdef USE_OMP
    #define TASK_SIZE 100
#else
    #define TASK_SIZE 8
#endif


