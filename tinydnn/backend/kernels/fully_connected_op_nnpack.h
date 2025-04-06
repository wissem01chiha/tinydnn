/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tinydnn/backend/backend.h"
#include "tinydnn/core/fully_params.h"

namespace tinydnn {
namespace kernels {

inline void fully_connected_op_nnpack(const tensor_t &in_data,
                                      const vec_t &W,
                                      const vec_t &bias,
                                      tensor_t &out_data,
                                      const core::fully_params &params,
                                      const bool layer_parallelize) {
#ifdef USE_NNPACK
  // call singleton to initialize NNPACK
  core::NNPackInitializer::getInstance().initialize();

  const float *kernel_ptr = W.data();
  const float *input_ptr  = in_data[0].data();
  float *output_ptr       = out_data[0].data();

  // TODO(edgarriba): embed it into a class
  const size_t num_mkl_threads = 1;
  pthreadpool_t threadpool     = pthreadpool_create(num_mkl_threads);

  const auto status =
    nnp_fully_connected_inference(params.in_size_, params.out_size_, input_ptr,
                                  kernel_ptr, output_ptr, threadpool);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_max_pooling_output");
  }

  // TODO(edgarriba): embed it into a class
  pthreadpool_destroy(threadpool);

  if (params.has_bias_) {
    for_i(layer_parallelize, params.out_size_,
          [&](size_t i) { output_ptr[i] += bias[i]; });
  }
#else
  UNREFERENCED_PARAMETER(in_data);
  UNREFERENCED_PARAMETER(W);
  UNREFERENCED_PARAMETER(bias);
  UNREFERENCED_PARAMETER(out_data);
  UNREFERENCED_PARAMETER(params);
  UNREFERENCED_PARAMETER(layer_parallelize);
  throw nn_error("TinyDNN has not been compiled with NNPACK support.");
#endif
}

}  // namespace kernels
}  // namespace tinydnn
