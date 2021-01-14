from ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive 
run apt-get -y update && apt-get install -y gcc g++ git wget gnupg lsb-release \
            libcurl4-openssl-dev python3 \
            libpython3-dev python3-pip libblas-dev ninja-build liblapack-dev \
    && python3 -m pip install ipopo cmake \
    && wget -qO- https://aide-qc.github.io/deploy/aide_qc/debian/PUBLIC-KEY.gpg | apt-key add - \
    && wget -qO- "https://aide-qc.github.io/deploy/aide_qc/debian/$(lsb_release -cs)/aide-qc.list" | tee -a /etc/apt/sources.list.d/aide-qc.list \
    && apt-get update \
    && apt-get install aideqc-llvm 
# run git clone https://github.com/ornl-qci/llvm-project-csp llvm \
#     && cd llvm && mkdir build && cd build \
#     && cmake -G Ninja ../llvm -DCMAKE_INSTALL_PREFIX=$HOME/.llvm -DBUILD_SHARED_LIBS=TRUE -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_PROJECTS="clang;mlir" \
#     && cmake --build . --target install \
#     && ln -sf $HOME/.llvm/bin/llvm-config /usr/bin/ && cd ../../ && rm -rf /llvm /var/lib/apt/lists/*