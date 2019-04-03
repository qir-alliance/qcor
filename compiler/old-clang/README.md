Noting this here:

clang++ -std=c++11 -Xclang -load -Xclang compiler/clang/libqcor-ast-plugin.so -Xclang -add-plugin -Xclang -enable-quantum test.cpp

with args

clang++-9 -std=c++11 -Xclang -load -Xclang compiler/clang/libqcor-ast-plugin.so -Xclang -add-plugin -Xclang enable-quantum -Xclang -plugin-arg-enable-quantum -Xclang test -Xclang -v test.cpp

A better example

```bash
clang++-9 -std=c++11 -Xclang -load -Xclang compiler/clang/libqcor-ast-plugin.so
    -Xclang -add-plugin -Xclang enable-quantum
    -Xclang -plugin-arg-enable-quantum -Xclang accelerator
    -Xclang -plugin-arg-enable-quantum -Xclang local-ibm
    -Xclang -plugin-arg-enable-quantum -Xclang transform
    -Xclang -plugin-arg-enable-quantum -Xclang circuit-optimizer
    -I /root/.xacc/include/xacc -I /root/.xacc/include/cppmicroservices4
    -I /home/project/qcor/runtime -L /home/project/qcor/build/runtime
    -lqcor -L /root/.xacc/lib -lxacc test.cpp -o test
```
test.cpp looks like this

```cpp
#include "qcor.hpp"

int main(int argc, char** argv) {
  
  xacc::Initialize(argc, argv);
  
  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.execute([&]() {
      H(0);
      CX(0, 1);
      Measure(0);
      Measure(1);
    });
  });

  auto results = future.get();

  results->print();

}
```