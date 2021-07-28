FROM qcor/llvm-alpine

RUN apk add libc6-compat && git clone https://github.com/ornl-qci/qcor && cd qcor && mkdir build && cd build \
   && cmake .. -G Ninja -DLLVM_ROOT=/usr/local/aideqc/llvm -DQCOR_EXTRA_COMPILER_FLAGS="-B /usr/lib/gcc/x86_64-alpine-linux-musl/10.3.1 -L /usr/lib/gcc/x86_64-alpine-linux-musl/10.3.1" -DQCOR_EXTRA_HEADERS="/usr/include/c++/10.3.1;/usr/include/c++/10.3.1/x86_64-alpine-linux-musl" \
   && cmake --build . --target install && cd ../.. && rm -rf qcor 
ENV PYTHONPATH "${PYTHONPATH}:/root/.xacc/bin"
ENV PATH "${PATH}:/root/.xacc/bin"
