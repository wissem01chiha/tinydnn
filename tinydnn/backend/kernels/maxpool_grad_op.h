/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tinydnn/core/op_kernel.h"
#include "tinydnn/backend/kernels/avx_maxpool_op.h"
#include "tinydnn/backend/kernels/maxpool_op_internal.h"
#include "tinydnn/backend/backend.h"

namespace tinydnn {

class MaxPoolGradOp : public core::OpKernel {
 public:
  explicit MaxPoolGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->maxpool();

    // incoming/outcoming data
    tensor_t &prev_delta = context.input_grad(0);
    tensor_t &curr_delta = context.output_grad(0);

    // initialize outputs
    fill_tensor(prev_delta, float_t{0});

    // call the algorithm depending on the selected engine type

    const backend_t engine = context.engine();

    if (engine == backend_t::internal) {
      kernels::maxpool_grad_op_internal(prev_delta, curr_delta,
                                        params.out2inmax, params.in2out,
                                        context.parallelize());
    } else if (engine == backend_t::avx) {
      kernels::maxpool_grad_op_avx(prev_delta, curr_delta, params.out2inmax,
                                   params.in2out, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tinydnn
