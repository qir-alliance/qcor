# EXPERIMENTAL!!! Prototype Quantum MLIR Dialect

## Building
As of now, we need a separate MLIR install
```bash
git clone https://github.com/llvm/llvm-project.git llvm_mlir
mkdir llvm_mlir/build
cd llvm_mlir/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DBUILD_SHARED_LIBS=TRUE
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_INSTALL_PREFIX=$HOME/.mlir

cmake --build . --target install
```
Just add `-DMLIR_DIR=$HOME/.mlir/lib/cmake/mlir` to CMake call for building qcor.
