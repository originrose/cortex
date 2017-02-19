FROM ubuntu:16.10

RUN apt-get update && apt-get install -y openjdk-8-jdk libhdf5-dev curl


RUN apt-get install -y wget zip g++
RUN wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein && \
    cp lein /usr/local/bin && chmod a+rx /usr/local/bin/lein && lein upgrade

# This is required for the keyboard-configuration install
ENV TERM xterm
ENV DEBIAN_FRONTEND noninteractive

# Install CUDA 8.0 for cortex gpu support
RUN apt-get install -y nvidia-cuda-toolkit nvidia-367 libcuda1-367

# Install CUDNN v5
ENV CUDNN_VERSION 5.1
LABEL com.nvidia.cudnn.version="5"

RUN CUDNN_DOWNLOAD_SUM=c10719b36f2dd6e9ddc63e3189affaa1a94d7d027e63b71c3f64d449ab0645ce && \
    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz -O && \
    echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict - && \
    tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local --wildcards 'cuda/lib64/libcudnn.so.*' && \
    rm cudnn-8.0-linux-x64-v5.1.tgz && \
    ldconfig
