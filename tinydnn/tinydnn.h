/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tinydnn/config.h"
#include "tinydnn/network.h"
#include "tinydnn/nodes.h"
#include "tinydnn/core/core.h"
#include "tinydnn/activations/activations.h"
#include "tinydnn/layers.h"
#include "tinydnn/loss/loss.h"
#include "tinydnn/optimizers/optimizers.h"
#include "tinydnn/utils/utils.h"
#include "tinydnn/io/io.h"

#ifdef DNN_USE_IMAGE_API
    #include "tinydnn/image/image.h"
#endif  // DNN_USE_IMAGE_API

#ifndef CNN_NO_SERIALIZATION
    #include "tinydnn/utils/deserialization_helper.h"
    #include "tinydnn/utils/serialization_helper.h"
    // to allow upcasting
    CEREAL_REGISTER_TYPE(tinydnn::elu_layer)
    CEREAL_REGISTER_TYPE(tinydnn::leaky_relu_layer)
    CEREAL_REGISTER_TYPE(tinydnn::relu_layer)
    CEREAL_REGISTER_TYPE(tinydnn::sigmoid_layer)
    CEREAL_REGISTER_TYPE(tinydnn::softmax_layer)
    CEREAL_REGISTER_TYPE(tinydnn::softplus_layer)
    CEREAL_REGISTER_TYPE(tinydnn::softsign_layer)
    CEREAL_REGISTER_TYPE(tinydnn::tanh_layer)
    CEREAL_REGISTER_TYPE(tinydnn::tanh_p1m2_layer)
#endif  // CNN_NO_SERIALIZATION

// shortcut version of layer names
namespace tinydnn {
namespace layers {

using conv = tinydnn::convolutional_layer;

using q_conv = tinydnn::quantized_convolutional_layer;

using max_pool = tinydnn::max_pooling_layer;

using ave_pool = tinydnn::average_pooling_layer;

using fc = tinydnn::fully_connected_layer;

using dense = tinydnn::fully_connected_layer;

using zero_pad = tinydnn::zero_pad_layer;

// using rnn_cell = tinydnn::rnn_cell_layer;

#ifdef CNN_USE_GEMMLOWP
using q_fc = tinydnn::quantized_fully_connected_layer;
#endif

using add = tinydnn::elementwise_add_layer;

using dropout = tinydnn::dropout_layer;

using input = tinydnn::input_layer;

using linear = linear_layer;

using lrn = tinydnn::lrn_layer;

using concat = tinydnn::concat_layer;

using deconv = tinydnn::deconvolutional_layer;

using max_unpool = tinydnn::max_unpooling_layer;

using ave_unpool = tinydnn::average_unpooling_layer;

}  // namespace layers

namespace activation {

using sigmoid = tinydnn::sigmoid_layer;

using asinh = tinydnn::asinh_layer;

using tanh = tinydnn::tanh_layer;

using relu = tinydnn::relu_layer;

using rectified_linear = tinydnn::relu_layer;

using softmax = tinydnn::softmax_layer;

using leaky_relu = tinydnn::leaky_relu_layer;

using elu = tinydnn::elu_layer;

using selu = tinydnn::selu_layer;

using tanh_p1m2 = tinydnn::tanh_p1m2_layer;

using softplus = tinydnn::softplus_layer;

using softsign = tinydnn::softsign_layer;

}  // namespace activation

#include "tinydnn/models/alexnet.h"

using batch_norm = tinydnn::batch_normalization_layer;

using l2_norm = tinydnn::l2_normalization_layer;

using slice = tinydnn::slice_layer;

using power = tinydnn::power_layer;

}  // namespace tinydnn

#ifdef CNN_USE_CAFFE_CONVERTER
// experimental / require google protobuf
    #include "tinydnn/io/caffe/layer_factory.h"
#endif
