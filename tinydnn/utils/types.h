/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

#pragma once

namespace tinydnn {

    #ifdef USE_DOUBLE
        typedef double float_t;
    #else
        typedef float float_t;
    #endif

}  // namespace tinydnn