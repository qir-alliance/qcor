Noting this here:

clang++ -std=c++11 -Xclang -load -Xclang compiler/clang/libqcor-ast-plugin.so -Xclang -add-plugin -Xclang -enable-quantum test.cpp

with args

clang++-9 -std=c++11 -Xclang -load -Xclang compiler/clang/libqcor-ast-plugin.so -Xclang -add-plugin -Xclang enable-quantum -Xclang -plugin-arg-enable-quantum -Xclang test -Xclang -v test.cpp

A better example

clang++-9 -std=c++11 -Xclang -load -Xclang compiler/clang/libqcor-ast-plugin.so
    -Xclang -add-plugin -Xclang enable-quantum
    -Xclang -plugin-arg-enable-quantum -Xclang accelerator
    -Xclang -plugin-arg-enable-quantum -Xclang tnqvm
    -Xclang -plugin-arg-enable-quantum -Xclang transform
    -Xclang -plugin-arg-enable-quantum -Xclang circuit-optimizer
    test.cpp

test.cpp looks like this

#include <stdio.h>
#include "qcor.hpp"

void foo() {

  printf("hi\n");

  qcor::submit([&](qcor::qpu_handler& qh){
    qh.vqe([&](double t0){
      X(0);
      Ry(t0,0);
      CX(1,0);
    }, 1, 1);
  });

 auto future = qcor::submit([&](qcor::qpu_handler& qh){
    qh.execute([&](double t0){
      X(2);
      Ry(t0,1);
      CX(1,2);
    });
  });

  printf("hi %d\n", future.get());
}

int main() {
    foo();
}