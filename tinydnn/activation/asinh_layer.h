/*
  Copyright (c) 2017, Taiga Nomi and the respective contributors
  All rights reserved.

  Use of this source code is governed by a BSD-style license that can be found
  in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>
#include "tinydnn/activation/activation_layer.h"
#include "tinydnn/layers/layer.h"

namespace tinydnn {

class asinh_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "asinh-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = std::asinh(x[j]);
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      dx[j] = dy[j] / std::cosh(y[j]);
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(-0.8), float_t(0.8));
  }

  friend struct serialization_buddy;
};

}  // namespace tinydnn
