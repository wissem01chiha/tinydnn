/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>
#include "tinydnn/backend/backend.h"
#include "tinydnn/core/framework/device.fwd.h"

namespace tinydnn {

/**
 * Abstract class for recurrent cells.
 **/
class cell : public layer {
 public:
  cell() : layer({}, {}) {}

  virtual std::vector<vector_type> input_order() = 0;

  virtual std::vector<vector_type> output_order() = 0;

  virtual void forward_propagation(const std::vector<tensor_t *> &in_data,
                                   std::vector<tensor_t *> &out_data) = 0;

  virtual void back_propagation(const std::vector<tensor_t *> &in_data,
                                const std::vector<tensor_t *> &out_data,
                                std::vector<tensor_t *> &out_grad,
                                std::vector<tensor_t *> &in_grad) = 0;

  virtual core::backend_t backend_type() const { return wrapper_->engine(); }

  virtual void init_backend(const layer *wrapper) = 0;

 protected:
  inline void set_wrapper(const layer *wrapper) { wrapper_ = wrapper; }

  const layer *wrapper_;  // every forward iteration, we must get the engine,
                          // backend, etc from the wrapper
};

}  // namespace tinydnn
