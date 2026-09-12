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
    help2man \
    libfl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src

# Build modern Verilator from source
RUN echo "Building modern Verilator from source..." && \
    git clone https://github.com/verilator/verilator.git /tmp/verilator && \
    cd /tmp/verilator && \
    git checkout v5.022 && \
    autoconf && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/verilator

# Clone and build CIRCT with Assertions
RUN git clone -b fix-invoke-verify https://github.com/Jimmy-Hu/circt.git
WORKDIR /opt/src/circt
RUN git submodule update --init

WORKDIR /opt/src/circt/llvm/build
RUN --mount=type=cache,target=/root/.ccache \
    cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
RUN ninja

WORKDIR /opt/src/circt/build
RUN --mount=type=cache,target=/root/.ccache \
    cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=/opt/src/circt/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=/opt/src/circt/llvm/build/lib/cmake/llvm \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
RUN ninja

# Clone and build custom Polygeist from source
RUN echo "Cloning and building custom Polygeist from source (this will take 1-2 hours)..." && \
    git clone -b fix-brace-init-undef --recursive https://github.com/Jimmy-Hu/Polygeist.git /tmp/polygeist && \
    mkdir -p /tmp/polygeist/llvm-project/build && cd /tmp/polygeist/llvm-project/build && \
    # Disable LLVM and MLIR tests to speed up CI build
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DLLVM_BUILD_TESTS=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DMLIR_INCLUDE_TESTS=OFF && \
    ninja && \
    mkdir -p /tmp/polygeist/build && cd /tmp/polygeist/build && \
    cmake -G Ninja .. \
        -DMLIR_DIR=/tmp/polygeist/llvm-project/build/lib/cmake/mlir \
        -DClang_DIR=/tmp/polygeist/llvm-project/build/lib/cmake/clang \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache && \
    ninja && \
    mkdir -p /opt/polygeist/bin /opt/polygeist/lib && \
    cp bin/cgeist /opt/polygeist/bin/ && \
    cp -r /tmp/polygeist/llvm-project/build/lib/clang /opt/polygeist/lib/ && \
    rm -rf /tmp/polygeist

# Create symlinks for all essential CIRCT binaries
RUN ln -s /opt/src/circt/build/bin/circt-opt /usr/local/bin/circt-opt && \
    ln -s /opt/src/circt/build/bin/circt-translate /usr/local/bin/circt-translate && \
    ln -s /opt/src/circt/build/bin/firtool /usr/local/bin/firtool

# Inject the binaries into the system PATH
ENV PATH="/opt/src/circt/build/bin:/opt/polygeist/bin:${PATH}"

# Verify installations during the image build
RUN circt-opt --version && \
    circt-translate --version && \
    firtool --version && \
    cgeist --version || true

# ==============================================================================
# Functional Sanity Checks
# ==============================================================================
WORKDIR /tmp/sanity_check

RUN echo "Running functional sanity checks for the hardware toolchain..." && \
    # 1. Test Polygeist Frontend (C++ to MLIR)
    echo 'int hw_kernel(int a) { return a + 1; }' > test.cpp && \
    cgeist -S -O3 --std=c++20 test.cpp -o test.mlir && \
    sed -i "s/module attributes {.*} {/module {/" test.mlir && \
    grep -q "func.func" test.mlir && \
    echo "[OK] Polygeist successfully compiled C++ to MLIR." && \
    \
    # 2. Test CIRCT Middle-end (MLIR Parsing & Optimization)
    circt-opt --pass-pipeline="builtin.module(canonicalize)" test.mlir -o test_opt.mlir && \
    echo "[OK] CIRCT-opt successfully parsed and optimized MLIR." && \
    \
    # 3. Test CIRCT Backend (FIRRTL to SystemVerilog via firtool)
    printf "FIRRTL version 4.0.0\ncircuit Dummy :\n  public module Dummy :\n    input in: UInt<32>\n    output out: UInt<32>\n    connect out, in\n" > test.fir && \
    firtool test.fir -o test.sv && \
    grep -q "module Dummy" test.sv && \
    echo "[OK] Firtool successfully lowered FIRRTL to SystemVerilog." && \
    \
    # 4. Test Verilator (SystemVerilog Syntax Linting)
    verilator --lint-only test.sv && \
    echo "[OK] Verilator successfully validated the generated SystemVerilog." && \
    \
    # Cleanup
    rm -rf /tmp/sanity_check

# Reset workspace to root
WORKDIR /