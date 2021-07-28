from xacc/alpine
run git clone https://github.com/ornl-qci/llvm-project-csp \
   && apk add ninja \
   && cd llvm-project-csp/ && mkdir build && cd build/ \
   && cmake -G Ninja ../llvm -DCMAKE_INSTALL_PREFIX=/usr/local/aideqc/llvm -DBUILD_SHARED_LIBS=TRUE -DLLVM_ENABLE_DUMP=TRUE -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_PROJECTS="clang;mlir" \
   && cmake --build . --target install && cd ../../ && rm -rf llvm-project-csp 
