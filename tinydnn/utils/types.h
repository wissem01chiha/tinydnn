/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

#pragma once

#include <vector>
#include "tinydnn/utils/aligned_allocator.h"
#include "tinydnn/utils/index3d.h"
#include <thirdparty/xtensor/xassign.hpp>

namespace tinydnn {

// ---------- Forward Declarations --------------------

#ifdef USE_DOUBLE
    typedef double float_t;
#else
    typedef float float_t;
#endif

typedef size_t label_t;
typedef size_t layer_size_t;  
typedef std::vector<float_t, aligned_allocator<float_t, 64>> vec_t;
typedef std::vector<vec_t> tensor_t;
typedef std::uint32_t serial_size_t;
typedef index3d<size_t> shape3d;

template <typename T>
using xtensor_t = xt::xexpression<T>;
template <class ValType, class T>
using value_type_is = std::enable_if_t<std::is_same<T, typename ValType::value_type>::value>;

template <class ValType>
using value_is_float = value_type_is<ValType, float>;

template <class ValType>
using value_is_double = value_type_is<ValType, double>;

enum class net_phase;
enum class padding;
enum class vector_type : int32_t;

template <typename>
struct is_xexpression : std::false_type {};

template <typename T>
struct is_xexpression<xt::xexpression<T>> : std::true_type {};

template <template <typename> class checker, typename... Ts>
struct are_all : std::true_type {};

template <template <typename> class checker, typename T0, typename... Ts>
struct are_all<checker, T0, Ts...>
  : std::integral_constant<bool, checker<T0>::value && are_all<checker, Ts...>::value> {};

template <typename... Ts>
using are_all_xexpr = are_all<is_xexpression, Ts...>;


// ---------- Implementations --------------------

enum class net_phase { train, test };

enum class padding { valid, same };

enum class vector_type : int32_t {
    data = 0x0001000,  
    weight = 0x0002000,
    bias   = 0x0002001,
    label = 0x0004000,
    aux   = 0x0010000  
};

}  // namespace tinydnn