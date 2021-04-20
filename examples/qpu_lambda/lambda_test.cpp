#include "qcor.hpp"

int main(int argc, char** argv) {
  int n = argc;
  double m = 22;

  auto a = qpu_lambda([](qreg q) {
      print("n was captured, and is ", n);
      print("m was captured, and is ", m);
      for (int i = 0; i < n; i++) {
        H(q[0]);
      }
      Measure(q[0]);
  }, n, m);

  auto q = qalloc(1);
  a(q);
  q.print();

  n = 2;
  m = 33.0;
  auto r = qalloc(1);
  print("running again to show capture variables are captured by reference");
  a(r);
  r.print();
}
