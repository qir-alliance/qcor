void foo(int* a, int *b) {
  if (a[0] > 1) {
    b[0] = 2;
  }

  auto l = [&]() {
    X(0);
    analog("ibm-hamiltonian-evolve");
    autogen("uccsd",2);
  };
}
