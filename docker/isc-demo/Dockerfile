FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

#Common deps
RUN apt-get -y update && apt-get -y install vim curl xz-utils \
      wget gpg software-properties-common git libblas-dev liblapack-dev \
      gcc g++ libcurl4-openssl-dev libpython3-dev python3 python3-pip

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main" > /etc/apt/sources.list.d/llvm.list && \
    apt-get update && apt-get install -y clang-tools-9 libclang-9-dev llvm-9-dev && \
    ln -s /usr/bin/clangd-9 /usr/bin/clangd && \
    ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config

RUN python3 -m pip install cmake pyquil numpy ipopo

RUN git clone --recursive https://github.com/eclipse/xacc && cd xacc && mkdir build && cd build \
    && cmake .. -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
    && make -j2 install

RUN git clone --recursive https://code.ornl.gov/qci/qcor && cd qcor && mkdir build && cd build \
    && cmake .. -DXACC_DIR=/root/.xacc && make -j2 install

RUN echo "export PATH=$PATH:/root/.xacc/bin" >> /root/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:/root/.xacc" >> /root/.bashrc
ADD .forest_config /root/