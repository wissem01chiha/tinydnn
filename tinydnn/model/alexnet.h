/*
  Copyright (c) 2013, Taiga Nomi and the respective contributors
  All rights reserved.

  Use of this source code is governed by a BSD-style license that can be found
  in the LICENSE file.
*/

#pragma once

#include <string>
#include "tinydnn/activation/activation.h"
#include "tinydnn/layers/layers.h"
#include "tinydnn/network.h"

namespace tinydnn {

class alexnet : public tinydnn::network<tinydnn::sequential> {
 public:
  explicit alexnet(const std::string &name = "")
    : tinydnn::network<tinydnn::sequential>(name) {
    using relu     = tinydnn::relu_layer;
    using conv     = tinydnn::convolutional_layer;
    using max_pool = tinydnn::max_pool;
    *this << conv(224, 224, 11, 11, 3, 64, padding::valid, true, 4, 4);
    *this << relu(54, 54, 64);
    *this << max_pool(54, 54, 64, 2);
    *this << conv(27, 27, 5, 5, 64, 192, padding::valid, true, 1, 1);
    *this << relu(23, 23, 192);
    *this << max_pool(23, 23, 192, 1);
    *this << conv(23, 23, 3, 3, 192, 384, padding::valid, true, 1, 1);
    *this << relu(21, 21, 384);
    *this << conv(21, 21, 3, 3, 384, 256, padding::valid, true, 1, 1);
    *this << relu(19, 19, 256);
    *this << conv(19, 19, 3, 3, 256, 256, padding::valid, true, 1, 1);
    *this << relu(17, 17, 256);
    *this << max_pool(17, 17, 256, 1);
  }
};

}  // namespace tinydnn
