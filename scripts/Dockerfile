FROM ubuntu:16.04

RUN apt-get update && apt-get install -y openjdk-8-jdk libhdf5-dev


RUN apt-get install -y wget zip g++
RUN wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein && \
    cp lein /usr/local/bin && chmod a+rx /usr/local/bin/lein && lein upgrade
