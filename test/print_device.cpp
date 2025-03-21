/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

#include "tinydnn/tinydnn.h"

using namespace tinydnn;

int main(int argc, char *argv[]) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
  if (argc < 3) {
    nn_warn("Need two parameters: platform_id and device_id.");
    return 0;
  }

  const int platform_id = atoi(argv[1]);
  const int device_id   = atoi(argv[2]);

  printAvailableDevice(platform_id, device_id);
#else
  CNN_UNREFERENCED_PARAMETER(argc);
  CNN_UNREFERENCED_PARAMETER(argv);
  nn_warn("TinyDNN was not compiled with OpenCL or CUDA support.");
#endif
  return 0;
}
