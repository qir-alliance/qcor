Noting this here:

clang++ -std=c++11 -Xclang -load -Xclang compiler/clang/libqcor-ast-plugin.so -Xclang -add-plugin -Xclang -enable-quantum test.cpp

with args

clang++-9 -std=c++11 -Xclang -load -Xclang compiler/clang/libqcor-ast-plugin.so -Xclang -add-plugin -Xclang enable-quantum -Xclang -plugin-arg-enable-quantum -Xclang test -Xclang -v test.cpp

test.cpp looks like this

#include <stdio.h>

void foo(int* a, int *b) {
  if (a[0] > 1) {
    b[0] = 2;
  }

  printf("hi\n");

  auto l = [&]() {
      int i = 1;
      printf("%d \n", 2);
    // X(0);
    // analog("ibm-hamiltonian-evolve");
    // autogen("uccsd",2);
  };
  l();
}

int main() {
    int a = 1;
    int b = 2;
    foo(&a, &b);
}