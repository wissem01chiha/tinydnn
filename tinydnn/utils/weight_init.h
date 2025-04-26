/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tinydnn/utils/types.h"
#include "tinydnn/utils/random.h"
#include "tinydnn/utils/product.h"

namespace tinydnn {
namespace weight_init {

// -------------------------------- Declarations ----------------------------------

class function;
class scalable;
class xavier;
class lecun;
class gaussian;
class constant;
class he;


// -------------------------------- Implementations --------------------------------


class function {
 public:
  virtual void fill(vec_t *weight, size_t fan_in, size_t fan_out) = 0;
};

class scalable : public function {
 public:
  explicit scalable(float_t value) : scale_(value) {}

  void scale(float_t value) { scale_ = value; }

 protected:
  float_t scale_;
};


class xavier : public scalable {
 public:
  xavier() : scalable(float_t(6)) {}
  explicit xavier(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    const float_t weight_base = std::sqrt(scale_ / (fan_in + fan_out));

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
  }
};

/**
 * Use fan-in(number of input weight for each neuron) for scaling
 *
 * Y LeCun, L Bottou, G B Orr, and K Muller,
 * Efficient backprop
 * Neural Networks, Tricks of the Trade, Springer, 1998
 **/
class lecun : public scalable {
 public:
  lecun() : scalable(float_t{1}) {}
  explicit lecun(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    UNREFERENCED_PARAMETER(fan_out);

    const float_t weight_base = scale_ / std::sqrt(float_t(fan_in));

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
  }
};

class gaussian : public scalable {
 public:
  gaussian() : scalable(float_t{1}) {}
  explicit gaussian(float_t sigma) : scalable(sigma) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    UNREFERENCED_PARAMETER(fan_in);
    UNREFERENCED_PARAMETER(fan_out);

    gaussian_rand(weight->begin(), weight->end(), float_t{0}, scale_);
  }
};

class constant : public scalable {
 public:
  constant() : scalable(float_t{0}) {}
  explicit constant(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    UNREFERENCED_PARAMETER(fan_in);
    UNREFERENCED_PARAMETER(fan_out);

    vectorize::fill(&(*weight)[0], weight->size(), scale_);
  }
};

class he : public scalable {
 public:
  he() : scalable(float_t{2}) {}
  explicit he(float_t value) : scalable(value) {}

  void fill(vec_t *weight, size_t fan_in, size_t fan_out) override {
    UNREFERENCED_PARAMETER(fan_out);

    const float_t sigma = std::sqrt(scale_ / fan_in);

    gaussian_rand(weight->begin(), weight->end(), float_t{0}, sigma);
  }
};

}  // namespace weight_init
}  // namespace tinydnn
