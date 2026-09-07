FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    aptitude \
    build-essential \
    ccache \
    wget \
    tar \
    gzip \
    ca-certificates \
    libstdc++6 \
    clang \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    libtbb-dev \
    libomp-dev \
    libopencv-dev \
    libboost-dev \
    libboost-all-dev \
    libz3-dev \
    flex \
    bison \
    autoconf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src

RUN echo "Building modern Verilator from source..." && \
    git clone https://github.com/verilator/verilator.git /tmp/verilator && \
    cd /tmp/verilator && \
    git checkout v5.022 && \
    autoconf && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/verilator

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

RUN echo "Cloning and building custom Polygeist from source (this will take 1-2 hours)..." && \
    git clone -b fix-brace-init-undef --recursive https://github.com/Jimmy-Hu/Polygeist.git /tmp/polygeist && \
    mkdir -p /tmp/polygeist/llvm-project/build && cd /tmp/polygeist/llvm-project/build && \
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON && \
    ninja && \
    mkdir -p /tmp/polygeist/build && cd /tmp/polygeist/build && \
    cmake -G Ninja .. \
        -DMLIR_DIR=/tmp/polygeist/llvm-project/build/lib/cmake/mlir \
        -DClang_DIR=/tmp/polygeist/llvm-project/build/lib/cmake/clang \
        -DCMAKE_BUILD_TYPE=Release && \
    ninja && \
    mkdir -p /opt/polygeist/bin /opt/polygeist/lib && \
    cp bin/cgeist /opt/polygeist/bin/ && \
    cp -r /tmp/polygeist/llvm-project/build/lib/clang /opt/polygeist/lib/ && \
    rm -rf /tmp/polygeist

# Inject the binaries into the system PATH
ENV PATH="/opt/circt/bin:/opt/polygeist/bin:${PATH}"

# Verify installations during the image build
RUN circt-opt --version || true

WORKDIR /