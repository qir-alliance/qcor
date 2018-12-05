from theiaide/theia-full:next
user root
run apt-get -y update \ 
    && unlink /usr/bin/clangd \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" > \ 
                 /etc/apt/sources.list.d/llvm.list \
    && apt-get -y update && apt-get install -y clang-tools-8 cmake libssl-dev \
              python3 libpython3-dev python3-pip vim gdb gfortran libblas-dev \
              liblapack-dev pkg-config \ 
    && ln -s /usr/bin/clangd-8 /usr/bin/clangd \
    && ln -s /usr/bin/clang++-8 /usr/bin/clang++ \
    && ln -s /usr/bin/clang-8 /usr/bin/clang \
    && git clone --recursive https://github.com/eclipse/xacc \
    && cd xacc && mkdir build && cd build \
    && CC=clang CXX=clang++ cmake .. -DPYTHON_INCLUDE_DIR=/usr/include/python3.5 -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    && make -j4 install && cd ../.. \
    && git clone --recursive https://github.com/ornl-qci/xacc-ibm \
    && cd xacc-ibm && mkdir build && cd build \
    && cmake .. -DXACC_DIR=~/.xacc && make -j4 install && cd ../.. \
    && git clone https://github.com/ornl-qci/tnqvm && cd tnqvm \
    && mkdir build && cd build \
    && cmake .. -DXACC_DIR=~/.xacc && make -j4 install && cd ../.. \
    && git clone --recursive https://github.com/ornl-qci/xacc-vqe \ 
    && cd xacc-vqe && mkdir build && cd build \
    && cmake .. -DXACC_DIR=~/.xacc -DPYTHON_INCLUDE_DIR=/usr/include/python3.5 \
    && make -j4 install && cd ../..
add settings.json /home/.theia/
user xacc-theia


