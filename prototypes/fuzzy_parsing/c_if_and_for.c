// void X(int);

void foo(int* a, int *b) {
  if (a[0] > 1) {
    b[0] = 2;
  }
//   auto x2 = []() { 
//       return 1;
//   };

  auto l = [&](int q) {
    X | 0;
    // int x = q;
  };
}

void bar(float x, float y); // just a declaration

void bang(int* a, int v) {
    int i;
    for (i = 0; i < v; ++i) {
        a[i] -= i;
    }
}