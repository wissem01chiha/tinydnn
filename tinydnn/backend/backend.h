/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>
#include "tinydnn/core/conv_params.h"
#include "tinydnn/core/deconv_params.h"
#include "tinydnn/core/fully_params.h"
#include "tinydnn/core/global_avepool_params.h"
#include "tinydnn/core/maxpool_params.h"
#include "tinydnn/core/node.h"

#ifdef USE_NNPACK
#include <nnpack.h>
#endif

namespace tinydnn {

class context;

enum class backend_t { internal, nnpack, libdnn, avx, opencl, cblas, intel_mkl };

inline std::ostream &operator<<(std::ostream &os, backend_t type) {
  switch (type) {
    case backend_t::internal: os  << "Internal"; break;
    case backend_t::nnpack: os    << "NNPACK"; break;
    case backend_t::libdnn: os    << "LibDNN"; break;
    case backend_t::avx: os       << "AVX"; break;
    case backend_t::opencl: os    << "OpenCL"; break;
    case backend_t::cblas: os     << "CBLAS"; break;
    case backend_t::intel_mkl: os << "Intel MKL"; break;
    default: throw nn_error("Not supported ostream enum."); break;
  }
  return os;
}

inline backend_t default_engine() {
#ifdef USE_AVX
#if defined(__AVX__) || defined(__AVX2__)
  return backend_t::avx;
#else
#error "your compiler does not support AVX"
#endif
#else
  return backend_t::internal;
#endif
}

#ifdef USE_NNPACK
// Singleton to keep a global state whether NNPACK is initialized.
// Before using the API an initialization is required. For this reason
// we need to get an instance of the object in order to avoid a throw error.
//
// Usage:
//     NNPackInitializer::getInstance().initialize();
//
class NNPackInitializer {
 public:
  // We create a static instance of the object in case
  // that it wasn't created before and we return it.
  static NNPackInitializer &getInstance() {
    static NNPackInitializer instance;
    return instance;
  }

  // Tries to initialize NNPACK.
  // Calls an internal method to initialize in case that it's not,
  // otherwise it returns a void.
  // Throws an error if we do not succed with initialization.
  void initialize() {
    if (initialized_) return;  // alredy initialized, do nothig.

    // calls internal method to initialize
    nnp_status init_status = nnp_initialize();
    if (init_status != nnp_status_success) {
      throw nn_error("Cannot initialize NNPACK.");
    }

    // succeded with initialization. We set the global
    // state to avoid exception errors in addition to
    // reuse code.
    initialized_ = true;
  }

 private:
  /** Flag to store whether NNPACK is initialized */
  bool initialized_ = false;
};

// TODO(you): create an interface to let users choose the algorithm
inline nnp_convolution_algorithm nnp_algorithm() {
  return nnp_convolution_algorithm_auto;
}

// TODO(you): create an interface to let users choose the transform strategy
inline nnp_convolution_transform_strategy nnp_kts() {
  // some algorithm accept tuple based only
  return nnp_convolution_transform_strategy_tuple_based;
}
#endif

class backend {
 public:
  // context holds solution-dependent parameters
  // context should be able to hold any types of structures (like boost::any)
  explicit backend(context *ctx_ = nullptr) {
    UNREFERENCED_PARAMETER(ctx_);
  }

  virtual void conv2d_q(const std::vector<tensor_t *> &in_data,
                        std::vector<tensor_t *> &out_data) = 0;

  virtual void conv2d_eq(const std::vector<tensor_t *> &in_data,
                         std::vector<tensor_t *> &out_data) = 0;

  virtual void conv2d_q(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) = 0;

  virtual void deconv2d(const std::vector<tensor_t *> &in_data,
                        std::vector<tensor_t *> &out_data) = 0;

  virtual void deconv2d_q(const std::vector<tensor_t *> &in_data,
                          std::vector<tensor_t *> &out_data) = 0;

  virtual void deconv2d_eq(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) = 0;

  virtual void deconv2d(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) = 0;

  virtual void deconv2d_q(const std::vector<tensor_t *> &in_data,
                          const std::vector<tensor_t *> &out_data,
                          std::vector<tensor_t *> &out_grad,
                          std::vector<tensor_t *> &in_grad) = 0;

  virtual void fully_q(const std::vector<tensor_t *> &in_data,
                       std::vector<tensor_t *> &out_data) = 0;

  virtual void fully_eq(const std::vector<tensor_t *> &in_data,
                        std::vector<tensor_t *> &out_data) = 0;

  virtual void fully_q(const std::vector<tensor_t *> &in_data,
                       const std::vector<tensor_t *> &out_data,
                       std::vector<tensor_t *> &out_grad,
                       std::vector<tensor_t *> &in_grad) = 0;

  context *get_context() const { return ctx_; }

  void set_layer(layer *layer) { layer_ = layer; }

  virtual backend_t type() const = 0;

 protected:
  context *ctx_;
  layer *layer_;
};

}  // namespace tinydnn
