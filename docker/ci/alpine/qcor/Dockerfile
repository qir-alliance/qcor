FROM qcor/llvm-alpine as llvm_install
FROM xacc/alpine
COPY --from=llvm_install /usr/local/aideqc/llvm /usr/local/aideqc/llvm
RUN apk add libc6-compat ninja && git clone https://github.com/ornl-qci/qcor && cd qcor && mkdir build && cd build \
   && cmake .. -G Ninja -DXACC_DIR=/usr/local/aideqc/qcor -DCMAKE_INSTALL_PREFIX=/usr/local/aideqc/qcor -DLLVM_ROOT=/usr/local/aideqc/llvm -DQCOR_EXTRA_COMPILER_FLAGS="-B /usr/lib/gcc/x86_64-alpine-linux-musl/10.3.1 -L /usr/lib/gcc/x86_64-alpine-linux-musl/10.3.1" -DQCOR_EXTRA_HEADERS="/usr/include/c++/10.3.1;/usr/include/c++/10.3.1/x86_64-alpine-linux-musl" \
   && cmake --build . --target install && cd ../.. && rm -rf qcor 
ENV PYTHONPATH "${PYTHONPATH}:/usr/local/aideqc/qcor"
ENV PATH "${PATH}:/usr/local/aideqc/qcor/bin"
