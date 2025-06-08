<!-- omit in toc -->
<p align="center">
<img src="./docs/logo/TinyDNN-logo-letters-alpha-version.png" />
</p>


[![ubuntu](https://github.com/wissem01chiha/tiny-dnn/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/wissem01chiha/tinydnn/actions/workflows/ubuntu.yml)
[![Windows](https://github.com/wissem01chiha/tiny-dnn/actions/workflows/windows.yml/badge.svg)](https://github.com/wissem01chiha/tinydnn/actions/workflows/windows.yml)
[![MacOS](https://github.com/wissem01chiha/tiny-dnn/actions/workflows/macos.yml/badge.svg)](https://github.com/wissem01chiha/tinydnn/actions/workflows/macos.yml)
[![documentation](https://github.com/wissem01chiha/tiny-dnn/actions/workflows/documentation.yml/badge.svg)](https://github.com/wissem01chiha/tinydnn/actions/workflows/documentation.yml)
[![Style](https://github.com/wissem01chiha/tiny-dnn/actions/workflows/style.yml/badge.svg)](https://github.com/wissem01chiha/tinydnn/actions/workflows/style.yml)

tinydnn is a C++14 implementation of deep learning. It is suitable for deep learning on limited computational resource, embedded systems and IoT devices.

## Table of contents

- [Table of contents](#table-of-contents)
- [Features](#features)
- [Dependencies](#dependencies)
- [Build](#build)
- [Examples](#examples)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Features
- Reasonably fast, without GPU:
    - With TBB threading and SSE/AVX vectorization.
    - 98.8% accuracy on MNIST in 13 minutes training (@Core i7-3520M).
- Portable & header-only:
    - Runs anywhere as long as you have a compiler which supports C++14.
    - Just include tiny_dnn.h and write your model in C++. There is nothing to install.
- Easy to integrate with real applications:
    - No output to stdout/stderr.
    - A constant throughput (simple parallelization model, no garbage collection).
    - Works without throwing an exception.
    - [Can import caffe's model](https://github.com/tiny-dnn/tiny-dnn/tree/master/examples/caffe_converter).
- Simply implemented:
    - A good library for learning neural networks.

## Dependencies
Nothing. All you need is a C++14 compiler (gcc 4.9+, clang 3.6+ or VS 2015+).

## Build
tinydnn is header-only, so *there's nothing to build*. If you want to execute sample program or unit tests, you need to install [cmake](https://cmake.org/) and type the following commands:

```
cmake . -DBUILD_EXAMPLES=ON
make
```

Then change to `examples` directory and run executable files.

If you would like to use IDE like Visual Studio or Xcode, you can also use cmake to generate corresponding files:

```
cmake . -G "Xcode"            # for Xcode users
cmake . -G "NMake Makefiles"  # for Windows Visual Studio users
```

Then open .sln file in visual studio and build(on windows/msvc), or type ```make``` command(on linux/mac/windows-mingw).

Some cmake options are available:

|options|description|default|additional requirements to use|
|-----|-----|----|----|
|USE_TBB|Use [Intel TBB](https://www.threadingbuildingblocks.org/) for parallelization|OFF<sup>1</sup>|[Intel TBB](https://www.threadingbuildingblocks.org/)|
|USE_OMP|Use OpenMP for parallelization|OFF<sup>1</sup>|[OpenMP Compiler](http://openmp.org/wp/openmp-compilers/)|
|USE_SSE|Use Intel SSE instruction set|ON|Intel CPU which supports SSE|
|USE_AVX|Use Intel AVX instruction set|ON|Intel CPU which supports AVX|
|USE_AVX2|Build tiny-dnn with AVX2 library support|OFF|Intel CPU which supports AVX2|
|USE_NNPACK|Use NNPACK for convolution operation|OFF|[Acceleration package for neural networks on multi-core CPUs](https://github.com/Maratyszcza/NNPACK)|
|USE_OPENCL|Enable/Disable OpenCL support (experimental)|OFF|[The open standard for parallel programming of heterogeneous systems](https://www.khronos.org/opencl/)|
|USE_LIBDNN|Use Greentea LibDNN for convolution operation with GPU via OpenCL (experimental)|OFF|[An universal convolution implementation supporting CUDA and OpenCL](https://github.com/naibaf7/libdnn)|
|USE_SERIALIZER|Enable model serialization|ON<sup>2</sup>|-|
|USE_DOUBLE|Use double precision computations instead of single precision|OFF|-|
|USE_ASAN|Use Address Sanitizer|OFF|clang or gcc compiler|
|USE_IMAGE_API|Enable Image API support|ON|-|
|USE_GEMMLOWP|Enable gemmlowp support|OFF|-|
|BUILD_TESTS|Build unit tests|OFF<sup>3</sup>|-|
|BUILD_EXAMPLES|Build example projects|OFF|-|
|BUILD_DOCS|Build documentation|OFF|[Doxygen](http://www.doxygen.org/)|
|PROFILE|Build unit tests|OFF|gprof|

<sup>1</sup> tiny-dnn use C++14 standard library for parallelization by default.
<sup>2</sup> If you don't use serialization, you can switch off to speedup compilation time.
<sup>3</sup> tiny-dnn uses [Google Test](https://github.com/google/googletest) as default framework to run unit tests. No pre-installation required, it's  automatically downloaded during CMake configuration.

For example, type the following commands if you want to use Intel TBB and build tests:
```bash
cmake -DUSE_TBB=ON -DBUILD_TESTS=ON .
```

## Examples
Construct convolutional neural networks

```cpp
#include "tinydnn/tinydnn.h"
using namespace tinydnn;
using namespace tinydnn::activation;
using namespace tinydnn::layers;

void construct_cnn() {
    using namespace tinydnn;

    network<sequential> net;

    // add layers
    net << conv(32, 32, 5, 1, 6) << tanh()  // in:32x32x1, 5x5conv, 6fmaps
        << ave_pool(28, 28, 6, 2) << tanh() // in:28x28x6, 2x2pooling
        << fc(14 * 14 * 6, 120) << tanh()   // in:14x14x6, out:120
        << fc(120, 10);                     // in:120,     out:10

    assert(net.in_data_size() == 32 * 32);
    assert(net.out_data_size() == 10);

    // load MNIST dataset
    std::vector<label_t> train_labels;
    std::vector<vec_t> train_images;

    parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);

    // declare optimization algorithm
    adagrad optimizer;

    // train (50-epoch, 30-minibatch)
    net.train<mse, adagrad>(optimizer, train_images, train_labels, 30, 50);

    // save
    net.save("net");

    // load
    // network<sequential> net2;
    // net2.load("net");
}
```
Construct multi-layer perceptron (mlp)

```cpp
#include "tinydnn/tinydnn.h"
using namespace tinydnn;
using namespace tinydnn::activation;
using namespace tinydnn::layers;

void construct_mlp() {
    network<sequential> net;

    net << fc(32 * 32, 300) << sigmoid() << fc(300, 10);

    assert(net.in_data_size() == 32 * 32);
    assert(net.out_data_size() == 10);
}
```

Another way to construct mlp

```cpp
#include "tinydnn/tinydnn.h"
using namespace tinydnn;
using namespace tinydnn::activation;

void construct_mlp() {
    auto mynet = make_mlp<tanh>({ 32 * 32, 300, 10 });

    assert(mynet.in_data_size() == 32 * 32);
    assert(mynet.out_data_size() == 10);
}
```

## Contributing
For a quick guide to contributing, take a look at the [Contribution Documents](CONTRIBUTING.md).

## References

## License
The BSD 3-Clause License
