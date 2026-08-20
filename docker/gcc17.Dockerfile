# Start from a modern and stable base image
FROM ubuntu:24.04

# Avoid tzdata interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and GCC prerequisites
RUN apt-get update && apt-get install -y \
    build-essential flex bison git \
    libgmp-dev libmpfr-dev libmpc-dev \
    cmake ninja-build clang-format \
    && rm -rf /var/lib/apt/lists/*

# Clone the GCC master branch (currently GCC 17 trunk) and compile
# Using depth=1 to fetch only the latest daily commit to save bandwidth
RUN echo "Cloning and building GCC 17 from source..." && \
    git clone --depth 1 https://gcc.gnu.org/git/gcc.git /tmp/gcc-source && \
    mkdir /tmp/gcc-build && cd /tmp/gcc-build && \
    /tmp/gcc-source/configure \
        --prefix=/opt/gcc-17 \
        --disable-multilib \
        --enable-languages=c,c++ && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/gcc-source /tmp/gcc-build

# Set environment variables to force the system and CMake to use GCC 17
ENV PATH="/opt/gcc-17/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/gcc-17/lib64:${LD_LIBRARY_PATH}"
ENV CC="/opt/gcc-17/bin/gcc"
ENV CXX="/opt/gcc-17/bin/g++"

# Set default working directory
WORKDIR /workspace