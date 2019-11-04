from xacc/ubuntu:18.04
run git clone --recursive https://github.com/eclipse/xacc && cd xacc && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) install \
    && cd ../../ && git clone -b master https://github.com/ornl-qci/qcor && cd qcor && mkdir build && cd build \
    && cmake .. -DXACC_DIR=~/.xacc -DQCOR_BUILD_TESTS=TRUE \
    && make -j$(nproc) install && ctest --output-on-failure