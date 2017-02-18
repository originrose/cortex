FROM ubuntu:16.10

RUN apt-get update && apt-get install -y openjdk-8-jdk libhdf5-dev


RUN apt-get install -y wget zip g++
RUN wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein && \
    cp lein /usr/local/bin && chmod a+rx /usr/local/bin/lein && lein upgrade

# This is required for the keyboard-configuration install
ENV TERM xterm
ENV DEBIAN_FRONTEND noninteractive

# Install CUDA for cortex gpu support
RUN apt-get install -y nvidia-cuda-toolkit nvidia-367 libcuda1-367
