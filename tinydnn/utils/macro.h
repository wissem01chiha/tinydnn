/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#define UNREFERENCED_PARAMETER(x) (void)(x)

#if defined _WIN32 && !defined(__MINGW32__)
#define WINDOWS
#endif

#if defined(_MSC_VER)
#define MUST_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__) || defined(__ICC)
#define MUST_INLINE __attribute__((always_inline)) inline
#else
#define MUST_INLINE inline
#endif

#define LOG_VECTOR(vec, name)
/*
void LOG_VECTOR(const vec_t& vec, const std::string& name) {
    std::cout << name << ",";

    if (vec.empty()) {
        std::cout << "(empty)" << std::endl;
    }
    else {
        for (size_t i = 0; i < vec.size(); i++) {
            std::cout << vec[i] << ",";
        }
    }

    std::cout << std::endl;
}
*/