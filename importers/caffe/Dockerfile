FROM ubuntu:16.04

RUN apt-get update && apt-get install -y openjdk-8-jdk libhdf5-dev


RUN apt-get install -y wget zip g++ && \
    wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein && \
    wget https://releases.hashicorp.com/vault/0.3.1/vault_0.3.1_linux_amd64.zip && \
    unzip vault_0.3.1_linux_amd64.zip && \
    chmod a+rx lein && \
    cp vault lein /usr/local/bin && \
    mkdir /root/.lein

ADD profiles.clj /root/.lein/profiles.clj
