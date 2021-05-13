#include <qcor_arithmetic>

__qpu__ void test_add_integer(qreg q) {
  H(q[2]); // |000> + |001>
  add_integer(q, 3); // ==> |111> + |110>
  Measure(q);
}


int main(int argc, char **argv) {
  set_shots(1024);
  auto a = qalloc(3);
  test_add_integer::print_kernel(a);
  test_add_integer(a);
  a.print();
  // Add 3 to a superposition of 0 and 4
  // => superposition of 3 and 7
  qcor_expect(a.counts().size() == 2);
  qcor_expect(a.counts()["111"] > 400);
  qcor_expect(a.counts()["110"] > 400);

  return 0;
}