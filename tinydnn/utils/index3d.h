/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tinydnn {

template <typename T>
struct index3d {
  index3d(T width, T height, T depth) { reshape(width, height, depth); }

  index3d() : width_(0), height_(0), depth_(0) {}

  void reshape(T width, T height, T depth) {
    width_  = width;
    height_ = height;
    depth_  = depth;

    if ((int64_t)width * height * depth > std::numeric_limits<T>::max())
      throw nn_error(format_str(
        "error while constructing layer: layer size too large for "
        "tiny-dnn\nWidthxHeightxChannels=%dx%dx%d >= max size of "
        "[%s](=%d)",
        width, height, depth, typeid(T).name(), std::numeric_limits<T>::max()));
  }

  T get_index(T x, T y, T channel) const {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    assert(channel >= 0 && channel < depth_);
    return (height_ * channel + y) * width_ + x;
  }

  T area() const { return width_ * height_; }

  T size() const { return width_ * height_ * depth_; }

  T width_;
  T height_;
  T depth_;
};

} // namespace tinydnn