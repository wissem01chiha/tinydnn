/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <cassert>
#include <stdarg.h>
#include "tinydnn/utils/macro.h"

#ifdef WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif  
#include <Windows.h>
#endif

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "thirdparty/CLCudaAPI/clpp11.h"
#else
#include "thirdparty/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tinydnn {

enum class Color { RED, GREEN, BLUE, YELLOW };

#ifdef WINDOWS
inline WORD getColorAttr(Color c) {
  switch (c) {
    case Color::RED: return FOREGROUND_RED;
    case Color::GREEN: return FOREGROUND_GREEN;
    case Color::BLUE: return FOREGROUND_BLUE;
    case Color::YELLOW: return FOREGROUND_GREEN | FOREGROUND_RED;
    default: assert(0); return 0;
  }
}
#else
inline const char *getColorEscape(Color c) {
  switch (c) {
    case Color::RED: return "\033[31m";
    case Color::GREEN: return "\033[32m";
    case Color::BLUE: return "\033[34m";
    case Color::YELLOW: return "\033[33m";
    default: assert(0); return "";
  }
}
#endif

inline void coloredPrint(Color c, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);

#ifdef WINDOWS
  const HANDLE std_handle = ::GetStdHandle(STD_OUTPUT_HANDLE);

  CONSOLE_SCREEN_BUFFER_INFO buffer_info;
  ::GetConsoleScreenBufferInfo(std_handle, &buffer_info);
  const WORD old_color = buffer_info.wAttributes;
  const WORD new_color = getColorAttr(c) | FOREGROUND_INTENSITY;

  fflush(stdout);
  ::SetConsoleTextAttribute(std_handle, new_color);

  vprintf(fmt, args);

  fflush(stdout);
  ::SetConsoleTextAttribute(std_handle, old_color);
#else
  printf("%s", getColorEscape(c));
  vprintf(fmt, args);
  printf("\033[m");
#endif
  va_end(args);
}

inline void coloredPrint(Color c, const std::string &msg) {
  coloredPrint(c, msg.c_str());
}


/**
 * error exception class for tinydnn
 **/
class nn_error : public std::exception {
 public:
  explicit nn_error(const std::string &msg) : msg_(msg) {}
  const char *what() const throw() override { return msg_.c_str(); }

 private:
  std::string msg_;
};

/**
 * warning class for tiny-dnn (for debug)
 **/
class nn_warn {
 public:
  explicit nn_warn(const std::string &msg) : msg_(msg) {
#ifdef USE_STDOUT
    coloredPrint(Color::YELLOW, msg_h_ + msg_);
#endif
  }

 private:
  std::string msg_;
  std::string msg_h_ = std::string("[WARNING] ");
};

/**
 * info class for tiny-dnn (for debug)
 **/
class nn_info {
 public:
  explicit nn_info(const std::string &msg) : msg_(msg) {
#ifdef USE_STDOUT
    std::cout << msg_h + msg_ << std::endl;
#endif
  }

 private:
  std::string msg_;
  std::string msg_h = std::string("[INFO] ");
};

class nn_not_implemented_error : public nn_error {
 public:
  explicit nn_not_implemented_error(const std::string &msg = "not implemented")
    : nn_error(msg) {}
};


// get all platforms (drivers), e.g. NVIDIA
// https://github.com/CNugteren/CLCudaAPI/blob/master/samples/device_info.cc

inline void printAvailableDevice(const size_t platform_id,
    const size_t device_id) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
// Initializes the CLCudaAPI platform and device. This initializes the
// OpenCL/CUDA back-end and
// selects a specific device on the platform.
auto platform = CLCudaAPI::Platform(platform_id);
auto device   = CLCudaAPI::Device(platform, device_id);

// Prints information about the chosen device. Most of these results should
// stay the same when
// switching between the CUDA and OpenCL back-ends.
printf("\n## Printing device information...\n");
printf(" > Platform ID                  %zu\n", platform_id);
printf(" > Device ID                    %zu\n", device_id);
printf(" > Framework version            %s\n", device.Version().c_str());
printf(" > Vendor                       %s\n", device.Vendor().c_str());
printf(" > Device name                  %s\n", device.Name().c_str());
printf(" > Device type                  %s\n", device.Type().c_str());
printf(" > Max work-group size          %zu\n", device.MaxWorkGroupSize());
printf(" > Max thread dimensions        %zu\n",
device.MaxWorkItemDimensions());
printf(" > Max work-group sizes:\n");
for (auto i = size_t{0}; i < device.MaxWorkItemDimensions(); ++i) {
printf("   - in the %zu-dimension         %zu\n", i,
device.MaxWorkItemSizes()[i]);
}
printf(" > Local memory per work-group  %zu bytes\n", device.LocalMemSize());
printf(" > Device capabilities          %s\n", device.Capabilities().c_str());
printf(" > Core clock rate              %zu MHz\n", device.CoreClock());
printf(" > Number of compute units      %zu\n", device.ComputeUnits());
printf(" > Total memory size            %zu bytes\n", device.MemorySize());
printf(" > Maximum allocatable memory   %zu bytes\n", device.MaxAllocSize());
printf(" > Memory clock rate            %zu MHz\n", device.MemoryClock());
printf(" > Memory bus width             %zu bits\n", device.MemoryBusWidth());
#else
UNREFERENCED_PARAMETER(platform_id);
UNREFERENCED_PARAMETER(device_id);
nn_warn("TinyDNN was not build with OpenCL or CUDA support.");
#endif
}

}  // namespace tinydnn