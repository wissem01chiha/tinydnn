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
#include "tinydnn/activation/activations.h"
#include "tinydnn/layers/layers.h"
#include "tinydnn/loss/loss.h"
#include "tinydnn/optimizer/optimizer.h"
#include "tinydnn/utils/utils.h"
#include "tinydnn/io/io.h"
#include "tinydnn/image/image.h"
#include "tinydd/audio/audio.h"
#include "tinydnn/model/model.h"







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
