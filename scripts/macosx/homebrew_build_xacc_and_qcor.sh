!#/bin/bash

set -x

git clone --recursive https://github.com/eclipse/xacc
cd xacc && $2 . -DCMAKE_INSTALL_PREFIX=$1 -DCMAKE_CXX_COMPILER=/usr/local/opt/gcc/bin/g++-10 -DCMAKE_C_COMPILER=/usr/local/opt/gcc/bin/gcc-10 -G Ninja
$2 --build . --target install
#cd ..
#mkdir build && cd build
#cd build
#$2 .. -G Ninja -DCMAKE_INSTALL_PREFIX=$1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/local/opt/gcc/bin/g++-10 -DCMAKE_C_COMPILER=/usr/local/opt/gcc/bin/gcc-10 -DXACC_DIR=$1 -DQCOR_EXTRA_HEADERS="/usr/local/opt/gcc/include/c++/10.2.0;/usr/local/opt/gcc/include/c++/10.2.0/x86_64-apple-darwin18" -DGCC_STDCXX_PATH=/usr/local/opt/gcc/lib/gcc/10 -DLLVM_ROOT=/usr/local/opt/llvm-csp
#$2 --build . --target install
