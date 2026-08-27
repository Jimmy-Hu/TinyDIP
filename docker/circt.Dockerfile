# Start from a clean, modern base
FROM ubuntu:24.04

# Avoid tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies and download tools (wget, tar)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    gzip \
    ca-certificates \
    libstdc++6 \
    cmake \
    git \
    ninja-build \
    python3 \
    libtbb-dev \
    libomp-dev \
    libopencv-dev \
    libboost-dev \
    libboost-all-dev \
    verilator \
    libz3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Pre-built CIRCT binaries directly
RUN echo "Downloading pre-built CIRCT binaries..." && \
    mkdir -p /opt/circt && \
    wget -qO- https://github.com/llvm/circt/releases/download/firtool-1.156.0/circt-full-shared-linux-x64.tar.gz | \
    tar -xz -C /opt/circt --strip-components=1

# Build Polygeist from source (Official repository does not provide pre-built binaries)
# This requires compiling the exact matching LLVM submodule first.
RUN echo "Cloning and building Polygeist from source (this will take 1-2 hours)..." && \
    git clone --recursive https://github.com/llvm/Polygeist.git /tmp/polygeist && \
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

WORKDIR /workspace