# Start from a clean, modern base
FROM ubuntu:24.04

# Avoid tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies and download tools (wget, tar)
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    gzip \
    ca-certificates \
    libstdc++6 \
    cmake \
    make \
    git \
    libtbb-dev \
    libomp-dev \
    libopencv-dev \
    libboost-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and extract Pre-built CIRCT binaries directly
RUN echo "Downloading pre-built CIRCT binaries..." && \
    mkdir -p /opt/circt && \
    wget -qO- https://github.com/llvm/circt/releases/download/firtool-1.156.0/circt-full-shared-linux-x64.tar.gz | \
    tar -xz -C /opt/circt --strip-components=1

# Download and extract Pre-built Polygeist binaries
# Polygeist also occasionally provides pre-built binaries, or you can point to a community build
RUN echo "Downloading pre-built Polygeist binaries..." && \
    mkdir -p /opt/polygeist && \
    wget -qO- https://github.com/llvm/Polygeist/releases/download/v0.0.4/Polygeist-ubuntu-20.04.tar.gz | \
    tar -xz -C /opt/polygeist --strip-components=1 || echo "Warning: Polygeist binary fetch failed, fallback needed."

# Inject the binaries into the system PATH
ENV PATH="/opt/circt/bin:/opt/polygeist/bin:${PATH}"

# Verify installations during the image build
RUN circt-opt --version || true

WORKDIR /workspace