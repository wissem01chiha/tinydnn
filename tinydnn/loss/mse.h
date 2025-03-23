/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

#include "tinydnn/utils/utils.h"

namespace tinydnn {

// mean-squared-error loss function for regression
class mse {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0.0};

    for (size_t i = 0; i < y.size(); ++i) d += (y[i] - t[i]) * (y[i] - t[i]);

    return d / static_cast<float_t>(y.size());
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());
    float_t factor = float_t(2) / static_cast<float_t>(t.size());

    for (size_t i = 0; i < y.size(); ++i) d[i] = factor * (y[i] - t[i]);

    return d;
  }
};
} // namsepace tinydnn 