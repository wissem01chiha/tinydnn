/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tinydnn/backend/backend.h"
#include "tinydnn/core/maxpool_params.h"

namespace tinydnn {
namespace kernels {

inline void maxpool_op_nnpack(const tensor_t &in_data,
                              tensor_t &out_data,
                              const core::maxpool_params &params) {
#ifdef USE_NNPACK
  // call singleton to initialize NNPACK
  core::NNPackInitializer::getInstance().initialize();

  const size_t input_channels = params.in.depth_;

  const nnp_size input_size = {params.in.width_, params.in.height_};

  const nnp_padding input_padding = {0, 0, 0, 0};

  const nnp_size pooling_size = {params.pool_size_x, params.pool_size_y};

  const nnp_size pooling_stride = {params.stride_x, params.stride_y};

  const float *input_ptr = in_data[0].data();
  float *output_ptr      = out_data[0].data();

  // TODO: embed it into a class
  const size_t num_mkl_threads = 1;
  pthreadpool_t threadpool     = pthreadpool_create(num_mkl_threads);

  const size_t batch_size = 1;

  const auto status = nnp_max_pooling_output(
    batch_size, input_channels, input_size, input_padding, pooling_size,
    pooling_stride, input_ptr, output_ptr, threadpool);

  if (status != nnp_status_success) {
    throw nn_error("Could not succeed with nnp_max_pooling_output");
  }

  // TODO: embed it into a class
  pthreadpool_destroy(threadpool);
#else
  UNREFERENCED_PARAMETER(in_data);
  UNREFERENCED_PARAMETER(out_data);
  UNREFERENCED_PARAMETER(params);
  throw nn_error("TinyDNN has not been compiled with NNPACK support.");
#endif
}

}  // namespace kernels
}  // namespace tinydnn
