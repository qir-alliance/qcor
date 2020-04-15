from xacc/ubuntu:18.04

run apt-get update && apt-get install -y ninja-build \
    && git clone https://github.com/hfinkel/llvm-project-csp llvm \
    && cd llvm && mkdir build && cd build \
    && cmake -G Ninja ../llvm -DCMAKE_INSTALL_PREFIX=$HOME/.llvm -DBUILD_SHARED_LIBS=TRUE -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_PROJECTS=clang \
    && cmake --build . --target install \
    && ln -sf $HOME/.llvm/bin/llvm-config /usr/bin/ && cd ../../ && rm -rf /llvm /var/lib/apt/lists/*