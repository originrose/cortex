FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER ThinkTopic

# For some reason this is required for the keyboard-configuration install (?)
ENV TERM xterm
ENV DEBIAN_FRONTEND noninteractive
ENV NATIVE_CUDA cudnn-8.0-linux-x64-v5.1.tgz
ENV VAULT_ADDR https://thinktopic.com:8200

# System requirements
RUN apt-get update                                && \
    apt-get install -y -q                            \
    python-software-properties software-properties-common \
    nmap wget curl htop unzip

# Install CUDA for cortex gpu support
# 16.10 install
#RUN apt-get install -y nvidia-cuda-toolkit nvidia-367 libcuda1-367

# 16.04 install (locally downloaded)
RUN apt-get install -y nvidia-367 cuda-cublas-8-0

# CUDNN from local
#ADD native/$NATIVE_CUDA native
#RUN mv native/cuda/lib64/libcudnn* /usr/lib

# CUDNN from nvidia -- this is provided if we start from the nvidia
# docker container, included here in case one wants to start from a
# different base image
#
# Install CUDNN v5
# From Here: https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/devel/cudnn5/Dockerfile
#ENV CUDNN_VERSION 5.1
#LABEL com.nvidia.cudnn.version="5"

#RUN CUDNN_DOWNLOAD_SUM=c10719b36f2dd6e9ddc63e3189affaa1a94d7d027e63b71c3f64d449ab0645ce && \
#    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz -O && \
#    echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict - && \
#    tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local --wildcards 'cuda/lib64/libcudnn.so.*' && \
#    rm cudnn-8.0-linux-x64-v5.1.tgz && \

# Move the libs to a standard location.
RUN mv /usr/local/cuda/lib64 /usr/lib/ && ldconfig

# java 8
RUN add-apt-repository ppa:webupd8team/java -y
RUN apt-get update
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections
RUN apt-get install -y oracle-java8-installer

# fix any issues
RUN apt-get -f install -y && apt-get upgrade -y

# Lein
RUN wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein && \
    mv lein /usr/bin && chmod a+x /usr/bin/lein && lein

# Certs for interacting with vault
ADD InstallCert/InstallCert.java /srv/InstallCert.java
RUN javac /srv/InstallCert.java
RUN find /usr -name "jre"
WORKDIR /srv
RUN echo "1" | java InstallCert thinktopic.com:8200
RUN cp jssecacerts /usr/lib/jvm/java-8-oracle/jre/lib/security

# vault
RUN wget https://releases.hashicorp.com/vault/0.6.1/vault_0.6.1_linux_amd64.zip -O vault.zip && \
    unzip vault.zip && mv vault /usr/bin

# docker machine
RUN  curl -L https://github.com/docker/machine/releases/download/v0.9.0/docker-machine-`uname -s`-`uname -m` >/tmp/docker-machine && \
     chmod +x /tmp/docker-machine && \
     cp /tmp/docker-machine /usr/local/bin/docker-machine

# docker
RUN apt install -y apt-transport-https ca-certificates && \
    curl -fsSL https://apt.dockerproject.org/gpg | apt-key add - && \
    add-apt-repository "deb https://apt.dockerproject.org/repo/ ubuntu-$(lsb_release -cs) main" && \
    apt update && apt install -y docker-engine

# upgrade to ubuntu 16.10
# exit 0 at the end because it tries to restart which fails
RUN apt-get install -y ubuntu-release-upgrader-core && \
    sed -i 's/Prompt=lts/Prompt=normal/' /etc/update-manager/release-upgrades && \
    /usr/bin/yes | do-release-upgrade -d ; exit 0

