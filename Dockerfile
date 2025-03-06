FROM ubuntu:22.04 AS build 

MAINTAINER Wissem Chiha <chihawissem08@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential       \
    cmake                 \
    python-pip            \
    python-setuptools     \
    git                   \
    vim

RUN pip install --upgrade pip


RUN apt-get install -y    \
    libpthread-stubs0-dev \
    libtbb-dev

RUN apt-get install -y    \
    clang-format-4.0

RUN pip install cpplint

# configure and build NNPACK

RUN apt-get install ninja-build

RUN pip install --upgrade setuptools && \
    pip install wheel && \
    pip install ninja-syntax

RUN pip install --upgrade git+https://github.com/tiny-dnn/PeachPy
RUN pip install --upgrade git+https://github.com/tiny-dnn/confu

WORKDIR /opt
RUN git clone https://github.com/tiny-dnn/NNPACK.git && \
    cd NNPACK && \
    confu setup && \
    python ./configure.py && \
    ninja

# install opencl and viennacl

RUN apt-get install -y \
    ocl-icd-opencl-dev \ 
    libviennacl-dev

# build and configure libdnn

WORKDIR /opt
RUN git clone https://github.com/naibaf7/libdnn.git && \
    cd libdnn && mkdir build && cd build && \
    cmake .. && make -j2
