FROM ubuntu:16.10

# This is required for the keyboard-configuration install
ENV TERM xterm
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends openjdk-8-jdk libhdf5-dev curl wget zip g++

RUN wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein && \
    cp lein /usr/local/bin && chmod a+rx /usr/local/bin/lein && lein upgrade
