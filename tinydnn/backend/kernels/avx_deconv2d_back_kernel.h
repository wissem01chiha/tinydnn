/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tinydnn/core/deconv_params.h"
#include "tinydnn/utils/utils.h"
#include "tinydnn/backend/kernels/tiny_deconv2d_back_kernel.h"

namespace tinydnn {
namespace core {
namespace kernels {

inline void avx_deconv2d_back_kernel(const deconv_params &params,
                                     const tensor_t &prev_out,
                                     const vec_t &W,
                                     tensor_t &dW,
                                     tensor_t &db,
                                     tensor_t &curr_delta,
                                     tensor_t *prev_delta) {
  // fallback to non-avx version
  tiny_deconv2d_back_kernel(params, prev_out, W, dW, db, curr_delta,
                            prev_delta);
}

}  // namespace kernels
}  // namespace core
}  // namespace tinydnn
