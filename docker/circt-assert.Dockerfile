FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    aptitude \
    build-essential \
    clang \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src

RUN git clone https://github.com/llvm/circt.git
WORKDIR /opt/src/circt
RUN git submodule update --init

WORKDIR /opt/src/circt/llvm/build
RUN cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
RUN ninja

WORKDIR /opt/src/circt/build
RUN cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=/opt/src/circt/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=/opt/src/circt/llvm/build/lib/cmake/llvm \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
RUN ninja

RUN ln -s /opt/src/circt/build/bin/circt-opt /usr/local/bin/circt-opt && \
    ln -s /opt/src/circt/build/bin/firtool /usr/local/bin/firtool

WORKDIR /